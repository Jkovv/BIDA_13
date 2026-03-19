import os
import pandas as pd

# The Oscar Award dataset (1927-2025) — Kaggle: unanimad/the-oscar-award
# place the_oscar_award.csv in /app/imdb/ before running
# direct tconst join where available, title+year fallback for the rest
OSCAR_PATH = "/app/imdb/the_oscar_award.csv"


def load_oscar_features(train_tconsts):
    df = pd.read_csv(OSCAR_PATH)

    # normalize columns - dataset has slight variations depending on version
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # winner column is True/False or 1/0 depending on version
    if "winner" in df.columns:
        df["won"] = df["winner"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["won"] = False

    # aggregate per film: total nominations and wins
    # some rows have imdb id, others don't — handle both
    has_tconst = "imdb_id" in df.columns or "tconst" in df.columns
    id_col = "tconst" if "tconst" in df.columns else ("imdb_id" if "imdb_id" in df.columns else None)

    if id_col:
        # clean up the id - sometimes stored as tt1234567, sometimes as 1234567
        df[id_col] = df[id_col].astype(str).str.strip()
        df[id_col] = df[id_col].apply(
            lambda x: f"tt{int(x):07d}" if x.isdigit() else x
        )
        agg = df.groupby(id_col).agg(
            oscar_nominations=("won", "count"),
            oscar_wins=("won", "sum")
        ).reset_index()
        agg = agg.rename(columns={id_col: "tconst"})
        print(f"  Oscar data: {len(agg)} films with tconst, joining directly")
    else:
        # no tconst column - fall back to title+year join
        # less reliable but still catches most films
        if "film" not in df.columns:
            print("  Warning: unexpected Oscar CSV format, skipping")
            return None

        df["film_clean"] = df["film"].str.lower().str.strip()
        agg = df.groupby("film_clean").agg(
            oscar_nominations=("won", "count"),
            oscar_wins=("won", "sum")
        ).reset_index()
        print(f"  Oscar data: {len(agg)} films (title join — less precise)")
        return agg  # caller handles title join

    agg["oscar_wins"] = agg["oscar_wins"].astype(int)
    return agg


def run():
    if not os.path.exists("/app/processed/train.parquet"):
        return

    if not os.path.exists(OSCAR_PATH):
        print("Step 4b: Oscar awards — file not found, skipping")
        print("  (download the_oscar_award.csv from kaggle.com/datasets/unanimad/the-oscar-award)")
        print("  (place in imdb/ folder and rebuild)")
        return

    print("Step 4b: Oscar awards enrichment")
    oscar = load_oscar_features(None)
    if oscar is None:
        return

    for name in ["train", "validation_hidden", "test_hidden"]:
        path = f"/app/processed/{name}.parquet"
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        df = df.drop(columns=["oscar_nominations", "oscar_wins"], errors="ignore")

        if "tconst" in oscar.columns:
            df = df.merge(oscar, on="tconst", how="left")
        else:
            # title fallback
            df["film_clean"] = df["title_clean"].str.lower().str.strip()
            df = df.merge(oscar, on="film_clean", how="left")
            df = df.drop(columns=["film_clean"])

        # films with no Oscar history get 0
        df["oscar_nominations"] = df["oscar_nominations"].fillna(0).astype(int)
        df["oscar_wins"] = df["oscar_wins"].fillna(0).astype(int)

        matched = (df["oscar_nominations"] > 0).sum()
        df.to_parquet(path, index=False)
        print(f"  {name}: {matched} films with Oscar history")

    print()


if __name__ == "__main__":
    run()
