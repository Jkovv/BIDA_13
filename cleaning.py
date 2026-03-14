import duckdb
import unicodedata
import json
import os
import pandas as pd

DATA_DIR = "/app/imdb"           
OUTPUT_DIR = "/app/processed"      

def strip_diacritics(text):
    if text is None:
        return None
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def print_scorecard(con, table_name, label):
    n = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"\n  [{label}] {n} rows")
    cols = con.execute(
        f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}'"
    ).fetchall()
    for (col,) in cols:
        nulls = con.execute(
            f'SELECT COUNT(*) FROM {table_name} WHERE "{col}" IS NULL'
        ).fetchone()[0]
        pct = 100 * (1 - nulls / n) if n > 0 else 0
        flag = " (!)" if pct < 95 else ""
        print(f"    {col:30s} {pct:5.1f}% complete{flag}")


def load_json_to_df(filepath, col1_name, col2_name, key1, key2):
    with open(filepath) as f:
        data = json.load(f)
    
    # case 1: list of objects 
    if isinstance(data, list):
        return pd.DataFrame({
            col1_name: [row.get(key1) for row in data],
            col2_name: [row.get(key2) for row in data]
        })
    
    # case 2: dict with keys
    if isinstance(data, dict):
        val1 = data.get(key1, {})
        val2 = data.get(key2, {})
        
        # dict values to list if needed
        if isinstance(val1, dict):
            val1 = list(val1.values())
        if isinstance(val2, dict):
            val2 = list(val2.values())
        
        return pd.DataFrame({col1_name: val1, col2_name: val2})
    
    raise ValueError(f"Unexpected JSON structure in {filepath}")


def run(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    con = duckdb.connect()
    con.create_function(
        "strip_diacritics", strip_diacritics,
        ['VARCHAR'], 'VARCHAR'
    )

    # ingest 
    print("Step 1: Ingesting raw data")

    train_path = os.path.join(data_dir, "train-*.csv")
    con.execute(f"""
        CREATE TABLE movies_raw AS
        SELECT * FROM read_csv_auto('{train_path}', header=true, ignore_errors=true)
    """)
    print(f"  {con.execute('SELECT COUNT(*) FROM movies_raw').fetchone()[0]} training movies")

    for split in ["validation_hidden", "test_hidden"]:
        path = os.path.join(data_dir, f"{split}.csv")
        if os.path.exists(path):
            con.execute(f"""
                CREATE TABLE {split}_raw AS
                SELECT * FROM read_csv_auto('{path}', header=true, ignore_errors=true)
            """)
            print(f"  {con.execute(f'SELECT COUNT(*) FROM {split}_raw').fetchone()[0]} {split} rows")

    # loading jsons
    df_directing = load_json_to_df(
        os.path.join(data_dir, "directing.json"),
        "tconst", "director_id", "movie", "director"
    )
    con.register("directing_view", df_directing)
    con.execute("CREATE TABLE directing AS SELECT * FROM directing_view")
    print(f"  {len(df_directing)} directing assignments")

    df_writing = load_json_to_df(
        os.path.join(data_dir, "writing.json"),
        "tconst", "writer_id", "movie", "writer"
    )
    con.register("writing_view", df_writing)
    con.execute("CREATE TABLE writing AS SELECT * FROM writing_view")
    print(f"  {len(df_writing)} writing assignments")

    # UVR key detection
    print("\n  Candidate key detection (UVR):")
    for (col,) in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name='movies_raw'"
    ).fetchall():
        uvr = con.execute(
            f'SELECT COUNT(DISTINCT "{col}")::FLOAT / COUNT(*) FROM movies_raw'
        ).fetchone()[0]
        tag = "<-- candidate key" if uvr and uvr > 0.95 else ""
        print(f"    {col}: {uvr:.4f} {tag}")

    # clean 
    print("\nStep 2: Cleaning")

    for src, dst, has_label in [
        ("movies_raw", "movies_clean", True),
        ("validation_hidden_raw", "validation_clean", False),
        ("test_hidden_raw", "test_clean", False),
    ]:
        if src not in [r[0] for r in con.execute("SHOW TABLES").fetchall()]:
            continue
        label_col = ", label" if has_label else ""

        con.execute(f"""
            CREATE TABLE {dst} AS
            SELECT
                tconst,
                strip_diacritics(primaryTitle) AS primaryTitle,
                primaryTitle AS primaryTitle_raw,
                COALESCE(originalTitle, strip_diacritics(primaryTitle)) AS originalTitle,
                CASE WHEN originalTitle IS NOT NULL
                      AND originalTitle != strip_diacritics(primaryTitle)
                     THEN TRUE ELSE FALSE END AS is_foreign,
                CASE WHEN primaryTitle != strip_diacritics(primaryTitle)
                     THEN TRUE ELSE FALSE END AS is_title_corrupted,
                TRY_CAST(COALESCE(NULLIF(startYear, '\\N'), NULLIF(endYear, '\\N')) AS INTEGER) AS year,
                TRY_CAST(NULLIF(runtimeMinutes, '\\N') AS INTEGER) AS runtimeMinutes,
                numVotes
                {label_col}
            FROM {src}
        """)
        nc = con.execute(f"SELECT COUNT(*) FROM {dst} WHERE is_title_corrupted").fetchone()[0]
        print(f"  {dst}: {nc} titles cleaned")

    print_scorecard(con, "movies_clean", "After cleaning")

    # impute
    print("\nStep 3: Stats and imputation")

    stats = con.execute("""
        SELECT AVG(numVotes), MEDIAN(numVotes),
               QUANTILE_CONT(numVotes, 0.25),
               QUANTILE_CONT(numVotes, 0.75)
        FROM movies_clean WHERE numVotes IS NOT NULL
    """).fetchone()
    mean, median, q1, q3 = stats
    print(f"  numVotes: mean={mean:,.0f}, median={median:,.0f}, "
          f"ratio={mean/median:.1f}x, IQR={q3-q1:,.0f}")

    # pre-calculate percentiles for winsorization
    runtime_percentiles = con.execute("""
        SELECT QUANTILE_CONT(runtimeMinutes, 0.05),
               QUANTILE_CONT(runtimeMinutes, 0.95)
        FROM movies_clean WHERE runtimeMinutes IS NOT NULL
    """).fetchone()
    p05, p95 = runtime_percentiles

    for src, dst in [
        ("movies_clean", "movies"),
        ("validation_clean", "validation"),
        ("test_clean", "test"),
    ]:
        if src not in [r[0] for r in con.execute("SHOW TABLES").fetchall()]:
            continue

        con.execute(f"""
            CREATE TABLE {dst} AS
            WITH imputed AS (
                SELECT *,
                    (year / 10 * 10) AS decade,
                    COALESCE(numVotes, MEDIAN(numVotes) OVER (PARTITION BY (year / 10 * 10))) AS numVotes_imputed,
                    COALESCE(runtimeMinutes, CAST(MEDIAN(runtimeMinutes) OVER (PARTITION BY (year / 10 * 10)) AS INTEGER)) AS runtimeMinutes_imputed,
                    numVotes IS NULL AS numVotes_was_imputed,
                    runtimeMinutes IS NULL AS runtime_was_imputed
                FROM {src}
            )
            SELECT *,
                LN(numVotes_imputed + 1) AS log_numVotes,
                GREATEST(LEAST(runtimeMinutes_imputed, {p95}), {p05}) AS runtimeMinutes_winsorised,
                numVotes_imputed / GREATEST(runtimeMinutes_imputed, 1) AS votes_per_minute
            FROM imputed
        """)

    nv = con.execute("SELECT COUNT(*) FROM movies WHERE numVotes_was_imputed").fetchone()[0]
    print(f"  Imputed {nv} missing numVotes")
    print_scorecard(con, "movies", "After imputation")

    # save
    print("\nSaving parquet checkpoints")
    for table, filename in [
        ("movies", "train.parquet"),
        ("validation", "validation.parquet"),
        ("test", "test.parquet"),
        ("directing", "directing.parquet"),
        ("writing", "writing.parquet"),
    ]:
        if table in [r[0] for r in con.execute("SHOW TABLES").fetchall()]:
            out = os.path.join(output_dir, filename)
            con.execute(f"COPY {table} TO '{out}' (FORMAT PARQUET)")
            print(f"  {out}")

    con.close()
    print("Cleaning done")

if __name__ == "__main__":
    run()
