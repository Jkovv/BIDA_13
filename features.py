# TF-IDF on movie titles using PySpark
# we use Spark here because the project requires it and it makes sense
# for the text processing pipeline (tokenize -> stopwords -> tf -> idf)

import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover


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

    # fit IDF on training titles only, then transform all splits
    pipeline_out = hashing_tf.transform(remover.transform(tokenizer.transform(train)))
    idf_model = idf.fit(pipeline_out)
    print(f"  IDF fitted on {train.count()} titles")

    for name in ["train", "validation_hidden", "test_hidden"]:
        path = f"/app/processed/{name}.parquet"
        if os.path.exists(path):
            df = spark.read.parquet(path)
            out = idf_model.transform(hashing_tf.transform(remover.transform(tokenizer.transform(df))))
            out.write.mode("overwrite").parquet(f"/app/processed/{name}_features.parquet")
            print(f"  {name}_features.parquet saved")

    spark.stop()
    print()


if __name__ == "__main__":
    run()