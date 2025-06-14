from t3_spark.session import get_spark_session
from pyspark.sql.functions import col, count, when, isnan, isnull, desc, asc

# Initialize Spark session using battle-tested approach
spark = get_spark_session()

print("=" * 80)
print("EDA: Reddit + EPSS RIGHT JOIN Result")
print("=" * 80)

# Read the parquet file
file_path = "parquet_preprocessing/input_reddit.parquet"
try:
    df = spark.read.parquet(file_path)
    print(f"✅ Successfully loaded: {file_path}")
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

print(f"\n📊 BASIC STATISTICS")
print(f"Total rows: {df.count():,}")
print(f"Total columns: {len(df.columns)}")

print(f"\n📋 COLUMN INFORMATION")
print("Column Name".ljust(25) + "Data Type".ljust(15) + "Nullable")
print("-" * 55)
for field in df.schema.fields:
    print(f"{field.name[:24].ljust(25)}{str(field.dataType)[:14].ljust(15)}{field.nullable}")

print(f"\n🔍 SAMPLE DATA (First 5 rows)")
df.show(5, truncate=False)

print(f"\n📈 NULL VALUE ANALYSIS")
null_counts = []
total_rows = df.count()
for col_name in df.columns:
    # Only use isnan for numeric columns
    col_type = dict(df.dtypes)[col_name]
    if col_type in ['double', 'float', 'int', 'bigint', 'smallint', 'tinyint']:
        null_count = df.filter(col(col_name).isNull() | isnan(col(col_name))).count()
    else:
        null_count = df.filter(col(col_name).isNull()).count()
    null_pct = (null_count / total_rows) * 100
    null_counts.append((col_name, null_count, null_pct))

print("Column Name".ljust(25) + "Null Count".ljust(15) + "Null %")
print("-" * 55)
for col_name, null_count, null_pct in sorted(null_counts, key=lambda x: x[2], reverse=True):
    print(f"{col_name[:24].ljust(25)}{str(null_count).ljust(15)}{null_pct:.1f}%")

print(f"\n🎯 KEY FIELD ANALYSIS")

# Check CVE ID distribution
print("\n📌 CVE ID Analysis:")
cve_count = df.select("cve_id").distinct().count()
print(f"Unique CVEs: {cve_count:,}")
print(f"Total rows: {total_rows:,}")
print(f"Avg EPSS records per CVE: {total_rows / cve_count:.2f}")

# Check Reddit coverage
print("\n📱 Reddit Coverage Analysis:")
reddit_non_null = df.filter(col("Post ID").isNotNull()).count()
reddit_coverage = (reddit_non_null / total_rows) * 100
print(f"Rows with Reddit data: {reddit_non_null:,} ({reddit_coverage:.1f}%)")
print(f"Rows without Reddit data: {total_rows - reddit_non_null:,} ({100-reddit_coverage:.1f}%)")

if reddit_non_null > 0:
    unique_posts = df.filter(col("Post ID").isNotNull()).select("Post ID").distinct().count()
    reddit_cves = df.filter(col("Post ID").isNotNull()).select("cve_id").distinct().count()
    print(f"Unique Reddit posts: {unique_posts:,}")
    print(f"CVEs mentioned on Reddit: {reddit_cves:,}")

# Check EPSS score distribution
print("\n📊 EPSS Score Analysis:")
if "epss" in df.columns:
    print("EPSS Score Statistics:")
    df.select("epss").describe().show()
    
    # EPSS score ranges
    epss_ranges = [
        ("Very Low (0.0-0.1)", 0.0, 0.1),
        ("Low (0.1-0.3)", 0.1, 0.3),
        ("Medium (0.3-0.7)", 0.3, 0.7),
        ("High (0.7-0.9)", 0.7, 0.9),
        ("Very High (0.9-1.0)", 0.9, 1.0)
    ]
    
    print("\nEPSS Score Distribution:")
    for label, min_val, max_val in epss_ranges:
        count = df.filter((col("epss") >= min_val) & (col("epss") < max_val)).count()
        pct = (count / total_rows) * 100
        print(f"{label.ljust(20)}: {count:,} ({pct:.1f}%)")

# Check date ranges
print("\n📅 Date Analysis:")
if "date" in df.columns:
    print("EPSS Date Range:")
    date_stats = df.select("date").describe()
    date_stats.show()
    
    # Check date distribution
    min_date = df.select("date").agg({"date": "min"}).collect()[0][0]
    max_date = df.select("date").agg({"date": "max"}).collect()[0][0]
    print(f"EPSS date range: {min_date} to {max_date}")

# Check for data quality issues
print(f"\n⚠️  DATA QUALITY CHECK")
issues = []

# Check for completely null rows
null_rows = df.filter(col("cve_id").isNull()).count()
if null_rows > 0:
    issues.append(f"Rows with null CVE IDs: {null_rows}")

# Check CVE ID format
invalid_cve_format = df.filter(~col("cve_id").rlike("^cve-\\d{4}-\\d+$")).count()
if invalid_cve_format > 0:
    issues.append(f"Invalid CVE ID format: {invalid_cve_format}")

# Check for duplicate CVE-date combinations
duplicate_check = df.groupBy("cve_id", "date").count().filter(col("count") > 1).count()
if duplicate_check > 0:
    issues.append(f"Duplicate CVE-date combinations: {duplicate_check}")

if issues:
    print("❌ Issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ No major data quality issues detected")

print(f"\n📈 COMPARISON WITH PREVIOUS ANALYSIS")
print("Previous (LEFT JOIN):")
print("  - Total rows: 5,035,210")
print("  - Unique CVEs: 1,973")
print("  - Reddit-centric dataset")
print("\nCurrent (RIGHT JOIN):")
print(f"  - Total rows: {total_rows:,}")
print(f"  - Unique CVEs: {cve_count:,}")
print("  - EPSS-centric dataset")

print(f"\n📈 SUMMARY")
print(f"✅ File loaded successfully: {file_path}")
print(f"✅ Total records: {total_rows:,}")
print(f"✅ Unique CVEs: {cve_count:,}")
print(f"✅ Reddit coverage: {reddit_coverage:.1f}%")
print(f"✅ Columns: {len(df.columns)}")

# Clean up
spark.catalog.clearCache()
print(f"\n✅ Analysis complete - Spark cache cleared") 