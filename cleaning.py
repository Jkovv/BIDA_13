import duckdb
import os
import unicodedata
import pandas as pd
import ftfy


def clean_txt(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    fixed = ftfy.fix_text(str(text))
    nfkd = unicodedata.normalize('NFKD', fixed.lower().strip())
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def process_split(con, table_name, has_label):
    label_select = ", label" if has_label else ""

    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name}_typed AS
        SELECT
            tconst, primaryTitle, originalTitle,
            TRY_CAST(NULLIF(startYear, '\\N') AS INTEGER) AS year_raw,
            TRY_CAST(NULLIF(endYear, '\\N') AS INTEGER) AS endYear_raw,
            LEAST(GREATEST(TRY_CAST(NULLIF(runtimeMinutes, '\\N') AS INTEGER), 1), 210) AS runtime_raw,
            TRY_CAST(numVotes AS DOUBLE) AS votes_raw
            {label_select}
        FROM {table_name}
    """)

    # impute missing runtime/votes with decade median using window functions
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name}_imputed AS
        SELECT *,
            COALESCE(year_raw, endYear_raw) AS year,
            COALESCE(runtime_raw,
                CAST(MEDIAN(runtime_raw) OVER (
                    PARTITION BY (COALESCE(year_raw, endYear_raw) / 10 * 10)
                ) AS INTEGER)
            ) AS runtime,
            LEAST(COALESCE(votes_raw,
                MEDIAN(votes_raw) OVER (
                    PARTITION BY (COALESCE(year_raw, endYear_raw) / 10 * 10)
                )), 400000) AS votes
        FROM {table_name}_typed
    """)

    # derive features - still in DuckDB because it's just math on columns
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name}_feat AS
        SELECT
            tconst, primaryTitle, originalTitle, year, runtime, votes,
            LN(votes + 1) AS log_votes,
            GREATEST(2026 - year, 1) AS film_age,
            votes / GREATEST(runtime, 1) AS votes_per_minute,
            CASE WHEN runtime < 90 THEN 1 ELSE 0 END AS runtime_short,
            LN(votes + 1) * runtime AS votes_x_runtime,
            CASE WHEN originalTitle IS NOT NULL
                  AND originalTitle != ''
                  AND originalTitle != primaryTitle
                 THEN 1 ELSE 0 END AS is_foreign
            {label_select}
        FROM {table_name}_imputed
    """)

    # only thing we need pandas for: ftfy encoding repair (can't do this in SQL)
    df = con.execute(f"SELECT * FROM {table_name}_feat").fetchdf()
    df["title_clean"] = df["primaryTitle"].apply(clean_txt)
    df = df.drop(columns=["primaryTitle", "originalTitle"], errors="ignore")
    return df


def run():
    con = duckdb.connect()

    print("Step 1: Ingesting raw data with DuckDB")
    con.execute("CREATE TABLE raw AS SELECT * FROM read_csv_auto('/app/imdb/train-*.csv', header=true, ignore_errors=true)")

    n_raw = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    n_unique = con.execute("SELECT COUNT(DISTINCT tconst) FROM raw").fetchone()[0]
    print(f"  {n_raw} rows, {n_unique} unique movies (duplicated {n_raw // n_unique}x)")

    # every movie appears 3x with identical data across the CSVs
    con.execute("CREATE TABLE deduped AS SELECT DISTINCT * FROM raw")
    print(f"  {con.execute('SELECT COUNT(*) FROM deduped').fetchone()[0]} after dedup")

    # quick data quality check
    nulls = con.execute("""
        SELECT
            COUNT(*) FILTER (WHERE NULLIF(startYear, '\\N') IS NULL) AS null_year,
            COUNT(*) FILTER (WHERE NULLIF(runtimeMinutes, '\\N') IS NULL) AS null_rt,
            COUNT(*) FILTER (WHERE numVotes IS NULL) AS null_votes
        FROM deduped
    """).fetchone()
    print(f"  Nulls: year={nulls[0]}, runtime={nulls[1]}, votes={nulls[2]}")

    print("\nStep 2: Cleaning & features")
    train = process_split(con, "deduped", has_label=True)
    os.makedirs('/app/processed', exist_ok=True)
    train.to_parquet('/app/processed/train.parquet', index=False)
    print(f"  train.parquet: {len(train)} rows, {len(train.columns)} cols")

    for split in ["validation_hidden", "test_hidden"]:
        path = f"/app/imdb/{split}.csv"
        if os.path.exists(path):
            con.execute(f"CREATE TABLE {split}_raw AS SELECT * FROM read_csv_auto('{path}', header=true, ignore_errors=true)")
            result = process_split(con, f"{split}_raw", has_label=False)
            result.to_parquet(f'/app/processed/{split}.parquet', index=False)
            print(f"  {split}.parquet: {len(result)} rows")

    con.close()
    print("Done.\n")


if __name__ == "__main__":
    run()