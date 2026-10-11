"""PySpark re-implementation of a subset of the features (for big-data scale, 110M+ clients in production).
`tests/test_spark_parity.py` checks that Spark and pandas produce identical values."""
from __future__ import annotations
from pyspark.sql import SparkSession, functions as F

PAY = [f"PAY_{i}" for i in range(1, 7)]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]


def get_spark():
    return (SparkSession.builder.master("local[2]").appName("credit-features")
            .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "4").getOrCreate())


def spark_features(sdf):
    n_late = sum(F.when(F.col(c) > 0, 1).otherwise(0) for c in PAY)
    max_delay = F.greatest(*[F.col(c) for c in PAY])
    util = [F.col(b) / F.greatest(F.col("LIMIT_BAL"), F.lit(1.0)) for b in BILL]
    util_mean = sum(util) / 6
    return sdf.select("client_id",
                      n_late.alias("n_late_months"),
                      max_delay.alias("max_delay"),
                      util_mean.alias("util_mean"),
                      util[0].alias("util_last"))
