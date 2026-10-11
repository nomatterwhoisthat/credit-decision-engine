import numpy as np
from src import data, features
from src.spark_features import get_spark, spark_features


def test_spark_matches_pandas():
    df = data.load().head(2000)
    spark = get_spark()
    out = spark_features(spark.createDataFrame(df)).toPandas().sort_values("client_id").reset_index(drop=True)
    X = features.build_features(df).reset_index(drop=True)
    for c in ["n_late_months", "max_delay", "util_mean", "util_last"]:
        assert np.allclose(out[c].to_numpy(float), X[c].to_numpy(float), atol=1e-9), c
