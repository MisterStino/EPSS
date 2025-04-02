import os
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import types as T
from t3_spark.session import get_spark_session

# NOOOOTEEE: PUBLICATION DATE IS NOT VALID FULLY, BECAUSE SOME OLD WERE ALREADY IN THERE AND PUBLISHED BEFORE.
# FOR PUBLUSHED LATER IT DOES SEEM TO BE 
# STILL INTERESTING TO SEE HOW IT CHANGES OVER TIME
# AND HOW IT IS STRATIFIED BY QUARTER OF PUBLICATION


def plot_quarterly_stratified(
    input_parquet="data/full_db/processed/final_full_data_parquet",
    threshold=0.7,
    output_dir="models/eda/figs/quarter_cohorts"
):
    """
    Creates two plots:
      1) Average EPSS over calendar time, lines grouped by 
         CVE's quarter of first appearance.
      2) Fraction of CVEs (in each quarter cohort) that have
         epss > threshold, for each calendar date.

    Steps:
      - Load the big parquet (cve, date, epss).
      - For each cve, find earliest date => define "quarter_of_pub".
      - Join back => each row now has "quarter_of_pub".
      - Group by (date, quarter_of_pub) to get average epss => for Plot1.
      - For Plot2, we also compute fraction above threshold => 
        we group by (date, quarter_of_pub) and do:
           count_above = sum(epss> threshold)
           fraction_above = count_above / total_in_cohort
        but we need total_in_cohort for each cohort 
        => we do a "count distinct cve" or a known total?
        Actually simpler approach: we group by (date, quarter_of_pub, cve)
        but that can be big. We'll see the steps carefully below.

    We produce two separate .png plots in `output_dir`.
    """

    # 1) Start Spark
    spark = get_spark_session()
    os.makedirs(output_dir, exist_ok=True)

    # 2) Read the big dataframe
    df = spark.read.parquet(input_parquet)

    # Ensure correct types
    df = (df
          .withColumn("cve", F.col("cve").cast(T.StringType()))
          .withColumn("date", F.col("date").cast(T.DateType()))
          .withColumn("epss", F.col("epss").cast(T.DoubleType()))
         )

    # 3) For each cve, find earliest date => define quarter
    #    We'll call it "pub_date" => min(date)
    cve_pub_df = (
        df.groupBy("cve")
          .agg(F.min("date").alias("pub_date"))
    )

    # Add a "quarter_of_pub" label, e.g. "YYYY-QN"
    # Something like year(pub_date) and quarter(pub_date)
    # Spark doesn't have a direct quarter-of-year function in older versions,
    # so we do: (month-1)//3 + 1
    cve_pub_df = (
        cve_pub_df
        .withColumn("year", F.year("pub_date"))
        .withColumn("month", F.month("pub_date"))
        # Use Python arithmetic instead of .plus(1)
        .withColumn(
            "quarter_idx", 
            ((F.col("month") - F.lit(1)) / F.lit(3)).cast(T.IntegerType()) + F.lit(1)
        )
        .withColumn(
            "quarter_of_pub", 
            F.concat_ws("-", 
                        F.col("year"), 
                        F.concat(F.lit("Q"), F.col("quarter_idx")))
        )
    )


    # We'll keep only "cve" and "quarter_of_pub"
    cve_pub_df = cve_pub_df.select("cve", "quarter_of_pub")

    # 4) Join back to original df => each row has quarter_of_pub
    df_joined = df.join(cve_pub_df, on="cve", how="inner")

    # 5) ================ PLOT 1: AVERAGE EPSS OVER CALENDAR TIME ================
    # group by date, quarter_of_pub => mean epss
    # Then we convert to pandas, we'll have (date, quarter_of_pub, avg_epss)
    daily_avg = (
        df_joined
        .groupBy("date", "quarter_of_pub")
        .agg(F.mean("epss").alias("avg_epss"))
    )

    daily_avg_pd = daily_avg.orderBy("date", "quarter_of_pub").toPandas()

    # Let's plot: x-axis = date, y-axis = avg_epss, color = quarter_of_pub
    plt.figure(figsize=(12, 7))
    # We'll group by quarter_of_pub in Pandas to create a separate line
    for qpub, subdf in daily_avg_pd.groupby("quarter_of_pub"):
        subdf = subdf.sort_values("date")
        plt.plot(subdf["date"], subdf["avg_epss"], label=qpub, marker='o', linewidth=1)

    plt.title("Average EPSS Over Time, Stratified by Quarter of Publication")
    plt.xlabel("Date")
    plt.ylabel("Average EPSS")
    plt.legend(title="Quarter of Pub", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    outpath1 = os.path.join(output_dir, "plot1_avg_epss_quarterly.png")
    plt.savefig(outpath1)
    plt.close()
    print(f"Saved Plot 1 to {outpath1}")

    # 6) =========== PLOT 2: FRACTION ABOVE THRESHOLD (0.7 by default) ============
    # We want: for each date, for each quarter_of_pub, fraction of cves that have epss>threshold
    # But how do we get "fraction"? We need to do:
    #    numerator = count of distinct cves (or just rows??) that have epss> threshold
    #    denominator = total distinct cves in that cohort
    #
    # Actually we might want the fraction among cves that appear in that day. 
    # If we define "cohort size" as the total number of cves in that quarter => that doesn't vary by date
    # So fraction = ( distinct cves in that cohort that have epss> threshold on day d ) / ( total cves in that cohort).
    # We'll do it as: 
    #    groupBy(date, quarter_of_pub, cve) => check if epss> threshold => max(1 or 0) 
    # Then group cve and sum
    # Then divide by total cves in that quarter.
    
    # 6a) get total cves in each quarter_cohort => a small DF
    cohort_sizes = (
        cve_pub_df
        .groupBy("quarter_of_pub")
        .agg(F.countDistinct("cve").alias("total_cves_in_cohort"))
    )

    # 6b) For each date, quarter_of_pub, we want distinct cves that exceed threshold => 
    # we can do something like:
    # filter by epss> threshold, then group by date, quarter_of_pub => countDistinct(cve)
    above_df = (
        df_joined
        .filter(F.col("epss") > threshold)
        .groupBy("date", "quarter_of_pub")
        .agg(F.countDistinct("cve").alias("count_above"))
    )

    # 6c) Now join "above_df" with "cohort_sizes" so we know how big each cohort is
    # We'll do a left join on quarter_of_pub
    frac_df = above_df.join(cohort_sizes, on="quarter_of_pub", how="left")
    # fraction = count_above / total_cves_in_cohort
    frac_df = frac_df.withColumn("fraction_above", F.col("count_above") / F.col("total_cves_in_cohort"))

    frac_pd = frac_df.orderBy("date", "quarter_of_pub").toPandas()

    # 6d) Plot lines => for each quarter_of_pub
    plt.figure(figsize=(12, 7))
    for qpub, subdf in frac_pd.groupby("quarter_of_pub"):
        subdf = subdf.sort_values("date")
        plt.plot(subdf["date"], subdf["fraction_above"], label=qpub, marker='o', linewidth=1)

    plt.title(f"Fraction of CVEs Above EPSS>{threshold}, By Quarter of Publication")
    plt.xlabel("Date")
    plt.ylabel("Fraction in Cohort with EPSS>threshold")
    plt.legend(title="Quarter of Pub", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    outpath2 = os.path.join(output_dir, "plot2_fraction_above_threshold_quarterly.png")
    plt.savefig(outpath2)
    plt.close()
    print(f"Saved Plot 2 to {outpath2}")

    # Done
    spark.stop()
    print("All done. Two plots created.")

if __name__ == "__main__":
    plot_quarterly_stratified()
