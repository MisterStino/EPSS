from t3_spark.session import get_spark_session
from pyspark.sql.functions import (
    col, countDistinct, desc, expr, when, lit
)

# 1) Create Spark session and load
spark = get_spark_session("EDA_GitHub_CVE")
df = (
    spark.read
         .option("header", True)
         .option("inferSchema", True)
         .csv("data/github/raw/github_bq.csv")
)

# Quick schema + total rows
df.printSchema()
total_rows = df.count()
print(f"Total rows: {total_rows}")

# 2) Distinct CVEs and days-per-CVE
distinct_cves = df.select("cve_id").distinct().count()
print(f"Distinct CVEs: {distinct_cves}")

days_per_cve = (
    df.groupBy("cve_id")
      .count()                    # number of dates each CVE appears
      .withColumnRenamed("count", "num_days")
)
days_per_cve.describe("num_days").show()
# 50/90/99th percentiles
days_per_cve.select(
    expr("percentile_approx(num_days, array(0.5, 0.9, 0.99))")
).show(truncate=False)

# 3) Per-date “spike” pattern
#   How many CVEs have any activity on each date?
active_cves_per_date = (
    df.filter(
        # at least one event type > 0
        (col("gh_issue_opens_cnt")    > 0) |
        (col("gh_pr_opens_cnt")       > 0) |
        (col("gh_issue_comment_cnt")  > 0) |
        (col("gh_pr_review_comment_cnt") > 0) |
        (col("gh_commit_comment_cnt") > 0) |
        (col("gh_push_event_cnt")     > 0)
    )
    .groupBy("date")
    .agg(countDistinct("cve_id").alias("num_active_cves"))
    .orderBy(desc("num_active_cves"))
)
active_cves_per_date.show(10)
active_cves_per_date.describe("num_active_cves").show()

# If you see dates with extremely high num_active_cves
# (e.g. many CVEs all start on the same day), that's a red flag:
# perhaps that was the initial bulk scrape, or a BigQuery glitch.

# 4) Summary stats + outlier detection on each GitHub‐count column
count_cols = [
    "gh_issue_opens_cnt",
    "gh_pr_opens_cnt",
    "gh_issue_comment_cnt",
    "gh_pr_review_comment_cnt",
    "gh_commit_comment_cnt",
    "gh_push_event_cnt",
    "gh_repo_uniques"
]
# Basic describe()
df.select(count_cols).describe().show()

# Approximate percentiles (50/90/99) for each count column
percentiles = [0.5, 0.9, 0.99]
exprs = [
    expr(f"percentile_approx({c}, array({','.join(str(p) for p in percentiles)}))")
      .alias(f"{c}_pct[{percentiles}]")
    for c in count_cols
]
df.select(exprs).show(truncate=False)

# 5) Simple “noise” filters you might apply:
#   a) Drop any entire date where > 80% of CVEs have any activity
#      (looks like a bulk scrape, not real signal)
threshold = 0.8 * distinct_cves
bad_dates = (
    active_cves_per_date
      .filter(col("num_active_cves") > threshold)
      .select("date")
)
clean_df = df.join(bad_dates, on="date", how="left_anti")

#   b) Winsorize extreme per-CVE outliers: cap each count at its 99th percentile
caps = df.select(exprs).first().asDict()  # get the percentile values
for c in count_cols:
    pct99 = caps[f"{c}_pct[{percentiles}]"][2]
    clean_df = clean_df.withColumn(
        c,
        when(col(c) > lit(pct99), lit(pct99)).otherwise(col(c))
    )

# Show cleaned summary
clean_df.describe(count_cols).show()

# …and now you can proceed to deeper EDA or modeling.

spark.stop()