"""
we divided this process into 2 phases:
Phase 1 (DuckDB): crew count joins (star schema), temporal features, title stats.
Phase 2 (PySpark): director/writer success rates with leave-one-out, TF-IDF.
"""
import duckdb
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

PROCESSED_DIR = "/app/processed"

def run_duckdb_features(input_dir=PROCESSED_DIR):
    con = duckdb.connect()

    con.execute(f"CREATE TABLE directing AS SELECT * FROM read_parquet('{input_dir}/directing.parquet')")
    con.execute(f"CREATE TABLE writing AS SELECT * FROM read_parquet('{input_dir}/writing.parquet')")
    
    # pre-clean crew tables for joins
    con.execute("DELETE FROM directing WHERE director_id = '\\N' OR director_id IS NULL")
    con.execute("DELETE FROM writing WHERE writer_id = '\\N' OR writer_id IS NULL")

    # star-schema aggregation: calculating crew sizes per movie
    con.execute("""
        CREATE TABLE crew_counts AS
        SELECT
            COALESCE(d.tconst, w.tconst) AS tconst,
            COALESCE(d.n_directors, 0) AS n_directors,
            COALESCE(w.n_writers, 0) AS n_writers,
            COALESCE(d.n_directors, 0) + COALESCE(w.n_writers, 0) AS crew_size,
            COALESCE(d.n_directors, 0) > 1 AS is_multi_director
        FROM (SELECT tconst, COUNT(DISTINCT director_id) AS n_directors FROM directing GROUP BY tconst) d
        FULL OUTER JOIN (SELECT tconst, COUNT(DISTINCT writer_id) AS n_writers FROM writing GROUP BY tconst) w
        ON d.tconst = w.tconst
    """)

    # reusability
    for src, dst in [("train.parquet", "train_feat.parquet"),
                      ("validation.parquet", "validation_feat.parquet"),
                      ("test.parquet", "test_feat.parquet")]:
        path = os.path.join(input_dir, src)
        if not os.path.exists(path):
            continue
            
        tbl = src.replace(".parquet", "")
        con.execute(f"""
            CREATE OR REPLACE TABLE {tbl} AS
            SELECT s.*,
                COALESCE(c.n_directors, 0) AS n_directors,
                COALESCE(c.n_writers, 0) AS n_writers,
                COALESCE(c.crew_size, 0) AS crew_size,
                COALESCE(c.is_multi_director, FALSE) AS is_multi_director,
                2024 - s.year AS movie_age,
                LENGTH(s.primaryTitle) AS title_length,
                LENGTH(s.primaryTitle) - LENGTH(REPLACE(s.primaryTitle, ' ', '')) + 1 AS title_word_count
            FROM read_parquet('{path}') s
            LEFT JOIN crew_counts c ON s.tconst = c.tconst
        """)
        
        out = os.path.join(input_dir, dst)
        con.execute(f"COPY {tbl} TO '{out}' (FORMAT PARQUET)")
        print(f"  Generated DuckDB features for {dst}")

    con.close()

def run_spark_features(input_dir=PROCESSED_DIR):
    spark = SparkSession.builder \
        .appName("IMDB-Features-Spark") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    movies = spark.read.parquet(os.path.join(input_dir, "train_feat.parquet"))
    directing = spark.read.parquet(os.path.join(input_dir, "directing.parquet")).filter(F.col("director_id") != "\\N")
    writing = spark.read.parquet(os.path.join(input_dir, "writing.parquet")).filter(F.col("writer_id") != "\\N")

    labels = movies.select("tconst", "label").filter(F.col("label").isNotNull())
    
    dir_lab = directing.join(F.broadcast(labels), "tconst", "inner")
    dir_stats = dir_lab.groupBy("director_id").agg(
        F.sum(F.col("label").cast("int")).alias("dir_success"),
        F.count("*").alias("dir_movies")
    )
    dir_feat = dir_lab.join(dir_stats, "director_id") \
        .withColumn("dir_loo", (F.col("dir_success") - F.col("label").cast("int")) / 
                    F.greatest(F.col("dir_movies") - 1, F.lit(1)))
    dir_movie = dir_feat.groupBy("tconst").agg(F.avg("dir_loo").alias("director_avg_success"),
                                              F.max("dir_movies").alias("director_max_experience"))

    wrt_lab = writing.join(F.broadcast(labels), "tconst", "inner")
    wrt_stats = wrt_lab.groupBy("writer_id").agg(
        F.sum(F.col("label").cast("int")).alias("wrt_success"),
        F.count("*").alias("wrt_movies")
    )
    wrt_feat = wrt_lab.join(wrt_stats, "writer_id") \
        .withColumn("wrt_loo", (F.col("wrt_success") - F.col("label").cast("int")) / 
                    F.greatest(F.col("wrt_movies") - 1, F.lit(1)))
    wrt_movie = wrt_feat.groupBy("tconst").agg(F.avg("wrt_loo").alias("writer_avg_success"),
                                              F.max("wrt_movies").alias("writer_max_experience"))

    # TF-IDF on titles 
    tokenizer = Tokenizer(inputCol="primaryTitle", outputCol="title_tokens")
    hashing = HashingTF(inputCol="title_tokens", outputCol="title_tf", numFeatures=100)
    idf = IDF(inputCol="title_tf", outputCol="title_tfidf")

    train_tokenized = tokenizer.transform(movies)
    train_hashed = hashing.transform(train_tokenized)
    idf_model = idf.fit(train_hashed)
    train_tfidf = idf_model.transform(train_hashed)

    defaults = {"director_avg_success": 0.5, "director_max_experience": 0,
                "writer_avg_success": 0.5, "writer_max_experience": 0}

    train_final = train_tfidf.join(dir_movie, "tconst", "left").join(wrt_movie, "tconst", "left").fillna(defaults)
    train_final.write.mode("overwrite").parquet(os.path.join(input_dir, "train_final.parquet"))

    for split in ["validation", "test"]:
        feat_path = os.path.join(input_dir, f"{split}_feat.parquet")
        if not os.path.exists(feat_path): continue
        
        df = spark.read.parquet(feat_path)
        df_tfidf = idf_model.transform(hashing.transform(tokenizer.transform(df)))
        df_final = df_tfidf.join(dir_movie, "tconst", "left").join(wrt_movie, "tconst", "left").fillna(defaults)
        
        df_final.write.mode("overwrite").parquet(os.path.join(input_dir, f"{split}_final.parquet"))
        print(f"  Finished {split}_final.parquet")

    spark.stop()

def run(input_dir=PROCESSED_DIR):
    print("Starting Feature Engineering...")
    run_duckdb_features(input_dir)
    run_spark_features(input_dir)
    print("Feature Engineering completed successfully.")

if __name__ == "__main__":
    run()
