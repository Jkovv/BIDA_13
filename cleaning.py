"""
cleaning.py — DuckDB ingestion + pandas cleaning.
Produces train.parquet, validation_hidden.parquet, test_hidden.parquet
"""
import duckdb
import os
import re
import math
import unicodedata
import pandas as pd
import ftfy


def clean_txt(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    fixed = ftfy.fix_text(str(text))
    nfkd = unicodedata.normalize('NFKD', fixed.lower().strip())
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def engineer_features(df, has_label):
    """Apply all cleaning + feature engineering in pandas."""

    df["title_clean"] = df["primaryTitle"].apply(clean_txt)

    # Parse year (use endYear as fallback)
    df["startYear"] = df["startYear"].replace(r"\N", None)
    df["endYear"] = df["endYear"].replace(r"\N", None)
    df["year"] = pd.to_numeric(df["startYear"], errors="coerce").fillna(
        pd.to_numeric(df["endYear"], errors="coerce")
    ).astype("Int64")

    # Runtime: parse + winsorise
    df["runtimeMinutes"] = df["runtimeMinutes"].replace(r"\N", None)
    df["runtime_raw"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")
    df["runtime"] = df["runtime_raw"].clip(lower=1, upper=210)

    # Impute missing runtime with decade median
    df["decade"] = (df["year"] // 10 * 10)
    decade_rt_median = df.groupby("decade")["runtime"].transform("median")
    df["runtime"] = df["runtime"].fillna(decade_rt_median)

    # Votes: cap outliers
    df["votes"] = pd.to_numeric(df["numVotes"], errors="coerce").clip(upper=400000)

    # Impute missing votes with decade median
    decade_vote_median = df.groupby("decade")["votes"].transform("median")
    df["votes"] = df["votes"].fillna(decade_vote_median)

    # Log votes — strongest single predictor
    df["log_votes"] = df["votes"].apply(lambda x: math.log(x + 1) if pd.notna(x) else 0)

    # Vote density: votes / film_age — detects cult classics
    df["vote_density"] = None
    mask = df["year"].notna() & (df["year"] < 2026)
    df.loc[mask, "vote_density"] = df.loc[mask, "votes"] / (2026 - df.loc[mask, "year"].astype(float))

    # Era buckets (survivorship bias is real for older films)
    def era(y):
        if pd.isna(y): return "contemporary"
        if y < 1930: return "silent"
        if y < 1950: return "golden_age"
        if y < 1970: return "classic"
        if y < 1990: return "new_hollywood"
        if y < 2005: return "modern"
        return "contemporary"
    df["era"] = df["year"].apply(era)

    # Foreign film flag — foreign films have 60.7% True rate vs 47.7% domestic
    df["originalTitle_clean"] = df["originalTitle"].apply(clean_txt)
    df["is_foreign"] = (
        df["originalTitle"].notna()
        & (df["originalTitle"] != "")
        & (df["originalTitle"] != df["primaryTitle"])
    ).astype(int)

    # Title-based features
    df["title_length"] = df["title_clean"].str.len().fillna(0).astype(int)
    df["title_word_count"] = df["title_clean"].str.split().str.len().fillna(0).astype(int)

    # Sequel indicator — sequels have much lower True rate
    df["is_sequel"] = df["primaryTitle"].apply(
        lambda t: int(bool(re.search(r'\b(II|III|IV|V|VI|VII|VIII|2|3|4|5|6|7|8|Part|Chapter|Vol)\b', str(t))))
    )

    # Votes missing flag — missingness itself is a signal
    df["votes_missing"] = df["numVotes"].isna().astype(int) | (df["numVotes"] == "").astype(int)

    # Title was corrupted (had diacritics errors) — correlates with True label
    df["title_corrupted"] = (df["primaryTitle"] != df["primaryTitle"].apply(
        lambda t: ''.join(c for c in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(c)) if pd.notna(t) else t
    )).astype(int)

    # Votes per minute of runtime
    df["votes_per_minute"] = df["votes"] / df["runtime"].clip(lower=1)

    keep = ["tconst", "title_clean", "year", "runtime", "votes", "log_votes",
            "vote_density", "era", "is_foreign", "title_length", "title_word_count",
            "is_sequel", "votes_missing", "title_corrupted", "votes_per_minute"]
    if has_label:
        keep.append("label")
    return df[keep]


def run():
    con = duckdb.connect()

    # --- Ingest with DuckDB (fast CSV glob reading) ---
    print("Step 1: Ingesting raw data with DuckDB")
    con.execute("CREATE TABLE raw AS SELECT * FROM read_csv_auto('/app/imdb/train-*.csv')")
    n_raw = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]
    n_unique = con.execute("SELECT COUNT(DISTINCT tconst) FROM raw").fetchone()[0]
    print(f"  {n_raw} raw rows, {n_unique} unique movies (each appears {n_raw//n_unique}x)")

    # Deduplicate (every movie appears 3x with identical data)
    con.execute("CREATE TABLE deduped AS SELECT DISTINCT * FROM raw")
    n_deduped = con.execute("SELECT COUNT(*) FROM deduped").fetchone()[0]
    print(f"  {n_deduped} rows after dedup")

    # --- Clean in pandas ---
    print("\nStep 2: Cleaning & feature engineering")
    df_train = con.execute("SELECT * FROM deduped").fetchdf()
    train = engineer_features(df_train, has_label=True)

    os.makedirs('/app/processed', exist_ok=True)
    train.to_parquet('/app/processed/train.parquet', index=False)
    print(f"  Saved train.parquet ({len(train)} rows, {len(train.columns)} cols)")

    # Process hidden sets
    for split in ["validation_hidden", "test_hidden"]:
        path = f"/app/imdb/{split}.csv"
        if os.path.exists(path):
            con.execute(f"CREATE TABLE {split}_raw AS SELECT * FROM read_csv_auto('{path}')")
            df = con.execute(f"SELECT * FROM {split}_raw").fetchdf()
            result = engineer_features(df, has_label=False)
            result.to_parquet(f'/app/processed/{split}.parquet', index=False)
            print(f"  Saved {split}.parquet ({len(result)} rows)")

    con.close()
    print("Cleaning done.\n")


if __name__ == "__main__":
    run()