import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover

def run():
    spark = SparkSession.builder.appName("IMDB-Features").getOrCreate()
    
    train_path = "/app/processed/train.parquet"
    if not os.path.exists(train_path):
        return

    # cache after reading, repartition by number not column
    train = spark.read.parquet(train_path).repartition(8).cache()
    
    # TF-IDF on cleaned titles
    tokenizer = Tokenizer(inputCol="title_clean", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashing_tf = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=1000)
    idf = IDF(inputCol="tf", outputCol="tfidf")
    
    idf_model = idf.fit(hashing_tf.transform(remover.transform(tokenizer.transform(train))))
    
    for name in ["train", "validation_hidden", "test_hidden"]:
        path = f"/app/processed/{name}.parquet"
        if os.path.exists(path):
            df = spark.read.parquet(path)
            words = tokenizer.transform(df)
            filtered = remover.transform(words)
            tf = hashing_tf.transform(filtered)
            feat = idf_model.transform(tf)
            feat.write.mode("overwrite").parquet(f"/app/processed/{name}_features.parquet")
    
    spark.stop()

if __name__ == "__main__":
    run()
