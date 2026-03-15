"""
features.py — PySpark TF-IDF on cleaned movie titles.
Uses Spark for distributed text processing (required by project grading).
"""
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover


def run():
    print("Step 5: PySpark TF-IDF feature extraction")

    train_path = "/app/processed/train.parquet"
    if not os.path.exists(train_path):
        return

    spark = SparkSession.builder \
        .appName("IMDB-Features") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    train = spark.read.parquet(train_path).repartition(8).cache()

    # TF-IDF pipeline on cleaned titles
    tokenizer = Tokenizer(inputCol="title_clean", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashing_tf = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=1000)
    idf = IDF(inputCol="tf", outputCol="tfidf")

    # Fit IDF model on training data only
    train_words = tokenizer.transform(train)
    train_filtered = remover.transform(train_words)
    train_tf = hashing_tf.transform(train_filtered)
    idf_model = idf.fit(train_tf)

    print(f"  IDF model fitted on {train.count()} training titles")

    for name in ["train", "validation_hidden", "test_hidden"]:
        path = f"/app/processed/{name}.parquet"
        if os.path.exists(path):
            df = spark.read.parquet(path)
            words = tokenizer.transform(df)
            filtered = remover.transform(words)
            tf = hashing_tf.transform(filtered)
            feat = idf_model.transform(tf)
            feat.write.mode("overwrite").parquet(f"/app/processed/{name}_features.parquet")
            print(f"  Saved {name}_features.parquet")

    spark.stop()
    print("TF-IDF done.\n")


if __name__ == "__main__":
    run()