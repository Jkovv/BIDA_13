# TF-IDF on movie titles using PySpark
# we use Spark here because the project requires it and it makes sense
# for the text processing pipeline (tokenize -> stopwords -> tf -> idf)
#
# the tfidf column is a SparseVector which prepare() in run.py can't use
# directly. we extract the top N dimensions as individual numeric columns
# (tfidf_0, tfidf_1, ...) so they flow into feature selection.

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover

N_TFIDF = 50  # top dimensions from 1000-dim sparse vector


def run():
    print("Step 5: TF-IDF (PySpark)")

    if not os.path.exists("/app/processed/train.parquet"):
        return

    spark = SparkSession.builder \
        .appName("IMDB-Features") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    train = spark.read.parquet("/app/processed/train.parquet").repartition(8).cache()

    tokenizer = Tokenizer(inputCol="title_clean", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashing_tf = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=1000)
    idf = IDF(inputCol="tf", outputCol="tfidf")

    pipeline_out = hashing_tf.transform(remover.transform(tokenizer.transform(train)))
    idf_model = idf.fit(pipeline_out)
    print(f"  IDF fitted on {train.count()} titles")

    # find which dimensions have highest total weight across training docs
    from pyspark.ml.linalg import SparseVector, DenseVector
    import numpy as np

    train_tfidf = idf_model.transform(pipeline_out)
    vectors = train_tfidf.select("tfidf").rdd.map(lambda r: r[0]).collect()
    dim_sums = np.zeros(1000)
    for v in vectors:
        if isinstance(v, SparseVector):
            for idx, val in zip(v.indices, v.values):
                dim_sums[idx] += abs(val)
        elif isinstance(v, DenseVector):
            dim_sums += np.abs(v.toArray())

    top_dims = np.argsort(dim_sums)[::-1][:N_TFIDF].tolist()
    print(f"  Extracting top {N_TFIDF} TF-IDF dimensions as numeric columns")

    def make_extractor(idx):
        def extract(v):
            if v is None:
                return 0.0
            if isinstance(v, SparseVector):
                lookup = dict(zip(v.indices, v.values))
                return float(lookup.get(idx, 0.0))
            return float(v[idx])
        return F.udf(extract, DoubleType())

    for name in ["train", "validation_hidden", "test_hidden"]:
        path = f"/app/processed/{name}.parquet"
        if os.path.exists(path):
            df = spark.read.parquet(path)
            words = tokenizer.transform(df)
            filtered = remover.transform(words)
            tf = hashing_tf.transform(filtered)
            feat = idf_model.transform(tf)

            for i, dim_idx in enumerate(top_dims):
                feat = feat.withColumn(f"tfidf_{i}", make_extractor(dim_idx)(F.col("tfidf")))

            feat = feat.drop("words", "filtered", "tf", "tfidf")
            feat.write.mode("overwrite").parquet(f"/app/processed/{name}_features.parquet")
            print(f"  {name}_features.parquet saved")

    spark.stop()
    print()


if __name__ == "__main__":
    run()