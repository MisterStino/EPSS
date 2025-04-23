from pyspark.sql.functions import col, log, lit, when
from t3_spark.session import get_spark_session


def transform_epss(input_path: str, output_path: str, transform: str = "inverted_log") -> None:
    """
    Reads a Parquet dataset at `input_path`, applies a probability-to-scale transform
    to the 'epss' column, and writes the result to `output_path`. Supports:
      - 'inverted_log': y = -log(p)
      - 'cloglog':       y = log(-log(1 - p))
      - 'logit':         y = log(p / (1 - p))

    Zero or one probabilities are floored to avoid infinite values.

    Args:
        input_path:   Path to the input Parquet file or directory.
        output_path:  Path where the transformed Parquet will be written.
        transform:    One of 'inverted_log', 'cloglog', 'logit'.
    """
    # 1) start Spark
    spark = get_spark_session()

    # 2) load data
    df = spark.read.parquet(input_path)

    # 3) constants for clipping
    eps_min = 1e-6
    one_minus_eps_min = 1.0 - eps_min

    # 4) clip p into (eps_min, 1 - eps_min)
    p = col("epss")
    p_clipped = when(p < eps_min, lit(eps_min)) \
                .when(p > one_minus_eps_min, lit(one_minus_eps_min)) \
                .otherwise(p)

    # 5) choose transform
    if transform == "inverted_log":
        # y = -log(p)
        expr = -log(p_clipped)
    elif transform == "cloglog":
        # y = log(-log(1 - p))
        one_minus_p = 1 - p_clipped
        h = -log(one_minus_p)       # hazard = -log(1 - p)
        expr = log(h)               # cloglog = log(hazard)
    elif transform == "logit":
        # y = log(p / (1 - p))
        one_minus_p = 1 - p_clipped
        expr = log(p_clipped / one_minus_p)
    else:
        raise ValueError(f"Unknown transform '{transform}'. Choose from inverted_log, cloglog, logit.")

    # 6) apply transformation
    df_transformed = df.withColumn("epss", expr)

    # 7) write output
    df_transformed.write.mode("overwrite").parquet(output_path)

    # 8) stop Spark
    spark.stop()


if __name__ == "__main__":
    # hard-coded paths for this run
    input_path = "data/full_db/sampled/final_full_data_sampled.parquet"
    # generate three sets
    transform_epss(input_path,
                   "data/full_db/ml-sets-sampled/log-scaled/full",       
                   transform="inverted_log")
    transform_epss(input_path,
                   "data/full_db/ml-sets-sampled/cloglog-scaled/full",
                   transform="cloglog")
    transform_epss(input_path,
                   "data/full_db/ml-sets-sampled/logit-scaled/full",
                   transform="logit")
