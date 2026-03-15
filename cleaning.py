import duckdb
import os
import unicodedata

def clean_txt(text):
    if text is None: return None
    # lowercase, trim, remove diacritics
    nfkd = unicodedata.normalize('NFKD', str(text).lower().strip())
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def run():
    con = duckdb.connect()
    con.create_function("clean_txt", clean_txt, ['VARCHAR'], 'VARCHAR')

    # Load all training CSVs
    con.execute("CREATE TABLE raw AS SELECT * FROM read_csv_auto('/app/imdb/train-*.csv')")
    
    # Winsorize runtime and cap votes to handle outliers
    con.execute("""
        CREATE TABLE clean_base AS SELECT DISTINCT
            tconst, clean_txt(primaryTitle) as title_clean,
            TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) as year,
            LEAST(GREATEST(TRY_CAST(NULLIF(runtimeMinutes, '\\N') AS INTEGER), 1), 210) as runtime,
            LEAST(numVotes, 400000) as votes,
            label
        FROM raw
    """)

    # Impute missing runtime with decade median, compute log_votes
    con.execute("""
        CREATE TABLE final_export AS SELECT
            tconst, title_clean, year, votes, label,
            COALESCE(runtime, MEDIAN(runtime) OVER (PARTITION BY (year/10*10))) as runtime,
            LOG(votes + 1) as log_votes
        FROM clean_base
    """)
    
    os.makedirs('/app/processed', exist_ok=True)
    con.execute("COPY final_export TO '/app/processed/train.parquet' (FORMAT PARQUET)")
    
    # Process hidden sets - include log_votes so run.py can use it
    for split in ["validation_hidden", "test_hidden"]:
        path = f"/app/imdb/{split}.csv"
        if os.path.exists(path):
            con.execute(f"CREATE TABLE {split}_raw AS SELECT * FROM read_csv_auto('{path}')")
            con.execute(f"""
                COPY (SELECT
                    tconst,
                    clean_txt(primaryTitle) as title_clean,
                    TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) as year,
                    LEAST(GREATEST(TRY_CAST(NULLIF(runtimeMinutes, '\\N') AS INTEGER), 1), 210) as runtime,
                    LEAST(numVotes, 400000) as votes,
                    LOG(LEAST(numVotes, 400000) + 1) as log_votes
                FROM {split}_raw) TO '/app/processed/{split}.parquet' (FORMAT PARQUET)
            """)
    con.close()

if __name__ == "__main__":
    run()
