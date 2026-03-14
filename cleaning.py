import duckdb
import os
import unicodedata

def clean_txt(text):
    if text is None: return None
    # Entity Resolution: lowercase, trim, and remove diacritics (Week 5)
    nfkd = unicodedata.normalize('NFKD', str(text).lower().strip())
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def run():
    con = duckdb.connect()
    con.create_function("clean_txt", clean_txt, ['VARCHAR'], 'VARCHAR')

    # Load disaggregated CSV files
    con.execute("CREATE TABLE raw AS SELECT * FROM read_csv_auto('/app/imdb/train-*.csv')")
    
    # Robust Centers (Week 5): Capping outliers (Winsorization) to mitigate synthetic errors
    con.execute("""
        CREATE TABLE clean_base AS SELECT DISTINCT
            tconst, clean_txt(primaryTitle) as title_clean,
            TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) as year,
            LEAST(GREATEST(TRY_CAST(NULLIF(runtimeMinutes, '\\N') AS INTEGER), 1), 210) as runtime,
            LEAST(numVotes, 400000) as votes,
            label
        FROM raw
    """)

    # Missing Value Imputation: Decade-based medians for quantitative cleaning
    con.execute("""
        CREATE TABLE final_export AS SELECT *,
            COALESCE(runtime, MEDIAN(runtime) OVER (PARTITION BY (year/10*10))) as runtime_f,
            LOG(votes + 1) as log_votes
        FROM clean_base
    """)
    
    os.makedirs('/app/processed', exist_ok=True)
    con.execute("COPY final_export TO '/app/processed/train.parquet' (FORMAT PARQUET)")
    
    # Clean hidden sets for prediction
    for split in ["validation_hidden", "test_hidden"]:
        path = f"/app/imdb/{split}.csv"
        if os.path.exists(path):
            con.execute(f"CREATE TABLE {split}_raw AS SELECT * FROM read_csv_auto('{path}')")
            con.execute(f"""
                COPY (SELECT tconst, clean_txt(primaryTitle) as title_clean,
                TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) as year,
                LEAST(GREATEST(TRY_CAST(NULLIF(runtimeMinutes, '\\N') AS INTEGER), 1), 210) as runtime,
                LEAST(numVotes, 400000) as votes
                FROM {split}_raw) TO '/app/processed/{split}.parquet' (FORMAT PARQUET)
            """)
    con.close()

if __name__ == "__main__":
    run()