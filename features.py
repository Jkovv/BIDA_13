import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover

def run():
    # Spark Session for distributed computing (Week 4 RDDs)
    spark = SparkSession.builder.appName("IMDB-Features").getOrCreate()
    
    # RDD Caching to avoid re-reading inputs (Resilient Distributed Datasets)
    train_path = "/app/processed/train.parquet"
    if not os.path.exists(train_path):
        return
        
    train = spark.read.parquet(train_path).repartition("tconst").cache()
    
    # NLP Pipeline: Converting normalized titles into numeric TF-IDF vectors
    tokenizer = Tokenizer(inputCol="title_clean", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashing_tf = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=1000)
    idf = IDF(inputCol="tf", outputCol="tfidf")
    
    pipeline = idf.fit(hashing_tf.transform(remover.transform(tokenizer.transform(train))))
    
    for name in ["train", "validation_hidden", "test_hidden"]:
        path = f"/app/processed/{name}.parquet"
        if os.path.exists(path):
            df = spark.read.parquet(path)
            feat = pipeline.transform(hashing_tf.transform(remover.transform(tokenizer.transform(df))))
            feat.write.mode("overwrite").parquet(f"/app/processed/{name}_features.parquet")
    
    spark.stop()

if __name__ == "__main__":
    run()