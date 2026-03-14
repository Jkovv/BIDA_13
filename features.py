import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover

PROCESSED_DIR = "/app/processed"

def run(input_dir=PROCESSED_DIR):
    spark = SparkSession.builder.appName("IMDB-Features-Max").getOrCreate()
    train = spark.read.parquet(f"{input_dir}/train.parquet").repartition("tconst").cache()
    labels = train.select("tconst", F.col("label").cast("double").alias("label"))
    global_mean = labels.select(F.avg("label")).collect()[0][0]

    def get_crew_features(path, prefix):
        if not os.path.exists(path): return None
        df = spark.read.parquet(path)
        stats = df.join(labels, "tconst").groupBy("person_id").agg(F.count("*").alias("c"), F.avg("label").alias("r"))
        res = df
        for m in [0, 5, 15, 30]:
            col = f"{prefix}_score_m{m}"
            smoothed = stats.withColumn(col, (F.col("r") * F.col("c") + (m * global_mean)) / (F.col("c") + m))
            res = res.join(smoothed.select("person_id", col), "person_id")
        return res.groupBy("tconst").agg(*[F.avg(f"{prefix}_score_m{m}").alias(f"{prefix}_score_m{m}") for m in [0, 5, 15, 30]])

    dir_features = get_crew_features(f"{input_dir}/directing.parquet", "dir")
    wrt_features = get_crew_features(f"{input_dir}/writing.parquet", "wrt")

    tokenizer = Tokenizer(inputCol="title_clean", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    # Increased to 2000 features for higher accuracy push
    hashing_tf = HashingTF(inputCol="filtered", outputCol="tf", numFeatures=2000)
    idf = IDF(inputCol="tf", outputCol="tfidf")
    
    idf_model = idf.fit(hashing_tf.transform(remover.transform(tokenizer.transform(train))))

    for n in ["train", "validation", "test"]:
        df = spark.read.parquet(f"{input_dir}/{n}.parquet").repartition("tconst")
        final = df
        if dir_features: final = final.join(dir_features, "tconst", "left")
        if wrt_features: final = final.join(wrt_features, "tconst", "left")
        final = final.fillna(global_mean)
        processed = idf_model.transform(hashing_tf.transform(remover.transform(tokenizer.transform(final))))
        processed.write.mode("overwrite").parquet(f"{input_dir}/{n}_final.parquet")
    spark.stop()

if __name__ == "__main__":
    run()