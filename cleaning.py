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
    """Pull raw data, clean in pandas, push back to DuckDB."""
    df = con.execute(f"SELECT * FROM {table_name}").fetchdf()

    df["title_clean"] = df["primaryTitle"].apply(clean_txt)
    df["year"] = pd.to_numeric(df["startYear"].replace("\\N", None), errors="coerce").astype("Int64")
    df["runtime_raw"] = pd.to_numeric(df["runtimeMinutes"].replace("\\N", None), errors="coerce")
    df["runtime"] = df["runtime_raw"].clip(lower=1, upper=210).astype("Int64")
    df["votes"] = df["numVotes"].clip(upper=400000)
    df["log_votes"] = (df["votes"] + 1).apply(lambda x: __import__('math').log(x))

    # Impute missing runtime with decade median
    df["decade"] = (df["year"] // 10 * 10)
    decade_median = df.groupby("decade")["runtime"].transform("median")
    df["runtime"] = df["runtime"].fillna(decade_median).astype("Int64")

    # Vote density: engagement relative to film age
    df["vote_density"] = None
    mask = df["year"].notna() & (df["year"] < 2026)
    df.loc[mask, "vote_density"] = df.loc[mask, "votes"] * 1.0 / (2026 - df.loc[mask, "year"].astype(float))

    # Era buckets
    def era(y):
        if pd.isna(y): return "contemporary"
        if y < 1930: return "silent"
        if y < 1950: return "golden_age"
        if y < 1970: return "classic"
        if y < 1990: return "new_hollywood"
        if y < 2005: return "modern"
        return "contemporary"
    df["era"] = df["year"].apply(era)

    keep = ["tconst", "title_clean", "year", "runtime", "votes", "log_votes", "vote_density", "era"]
    if has_label:
        keep.append("label")
    return df[keep]

def run():
    con = duckdb.connect()

    # Load all training CSVs
    con.execute("CREATE TABLE raw AS SELECT * FROM read_csv_auto('/app/imdb/train-*.csv')")
    print(f"  {con.execute('SELECT COUNT(*) FROM raw').fetchone()[0]} training movies")

    train = process_split(con, "raw", has_label=True)
    os.makedirs('/app/processed', exist_ok=True)
    train.to_parquet('/app/processed/train.parquet', index=False)
    print(f"  Saved train.parquet ({len(train)} rows)")

    # Process hidden sets
    for split in ["validation_hidden", "test_hidden"]:
        path = f"/app/imdb/{split}.csv"
        if os.path.exists(path):
            con.execute(f"CREATE TABLE {split}_raw AS SELECT * FROM read_csv_auto('{path}')")
            df = process_split(con, f"{split}_raw", has_label=False)
            df.to_parquet(f'/app/processed/{split}.parquet', index=False)
            print(f"  Saved {split}.parquet ({len(df)} rows)")

    con.close()
    print("Cleaning done.")

if __name__ == "__main__":
    run()