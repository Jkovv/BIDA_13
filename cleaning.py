import duckdb
import unicodedata
import json
import os
import pandas as pd

DATA_DIR = "/app/imdb"           
OUTPUT_DIR = "/app/processed"      

def strip_diacritics(text):
    if text is None: return None
    nfkd = unicodedata.normalize('NFKD', str(text))
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def run(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    con = duckdb.connect()
    con.create_function("strip_diacritics", strip_diacritics, ['VARCHAR'], 'VARCHAR')

    print("Step 1: Loading disaggregated project data...")
    con.execute(f"CREATE TABLE movies_raw AS SELECT * FROM read_csv_auto('{data_dir}/train-*.csv', header=true)")
    
    for split in ["validation_hidden", "test_hidden"]:
        path = os.path.join(data_dir, f"{split}.csv")
        if os.path.exists(path):
            con.execute(f"CREATE TABLE {split}_raw AS SELECT * FROM read_csv_auto('{path}', header=true)")

    for js_file, table, id_key in [("directing.json", "directing", "director"), 
                                   ("writing.json", "writing", "writer")]:
        path = os.path.join(data_dir, js_file)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                df = pd.DataFrame(data).rename(columns={"movie": "tconst", id_key: "person_id"})
                con.register(f"{table}_view", df)
                con.execute(f"CREATE TABLE {table} AS SELECT * FROM {table}_view WHERE person_id != '\\N'")
                con.execute(f"COPY {table} TO '{output_dir}/{table}.parquet' (FORMAT PARQUET)")

    print("Step 2: Normalization and Temporal Imputation...")
    for src, dst, has_label in [("movies_raw", "movies_clean", True), 
                                ("validation_hidden_raw", "validation_clean", False),
                                ("test_hidden_raw", "test_clean", False)]:
        if src not in [r[0] for r in con.execute("SHOW TABLES").fetchall()]: continue
        l_col = ", label" if has_label else ""
        con.execute(f"""
            CREATE TABLE {dst} AS SELECT 
                tconst, strip_diacritics(primaryTitle) as title_clean,
                primaryTitle != originalTitle AS is_foreign,
                LENGTH(primaryTitle) as title_len,
                TRY_CAST(COALESCE(NULLIF(startYear, '\\N'), NULLIF(endYear, '\\N')) AS INTEGER) AS year,
                TRY_CAST(NULLIF(runtimeMinutes, '\\N') AS INTEGER) AS runtime,
                numVotes {l_col}
            FROM {src}
        """)

    for src, dst in [("movies_clean", "train"), ("validation_clean", "validation"), ("test_clean", "test")]:
        con.execute(f"""
            CREATE TABLE {dst} AS SELECT *,
                (year / 10 * 10) AS decade,
                COALESCE(numVotes, MEDIAN(numVotes) OVER (PARTITION BY (year/10*10))) AS votes_filled,
                COALESCE(runtime, CAST(MEDIAN(runtime) OVER (PARTITION BY (year/10*10)) AS INTEGER)) AS runtime_filled,
                LOG(COALESCE(numVotes, 1) + 1) AS log_votes,
                numVotes / GREATEST(runtime, 1) as vote_density
            FROM {src}
        """)
        con.execute(f"COPY {dst} TO '{output_dir}/{dst}.parquet' (FORMAT PARQUET)")
    con.close()

if __name__ == "__main__":
    run()