import json
import pandas as pd
import numpy as np
import os

def load_directing(path):
    with open(path) as f:
        d = json.load(f)
    # pandas-exported format: {"movie": {"0": "tt...", ...}, "director": {"0": "nm...", ...}}
    return pd.DataFrame({"tconst": d["movie"].values(), "director_id": d["director"].values()})

def load_writing(path):
    with open(path) as f:
        w = json.load(f)
    # list of dicts: [{"movie": "tt...", "writer": "nm..."}, ...]
    return pd.DataFrame({"tconst": [x["movie"] for x in w], "writer_id": [x["writer"] for x in w]})

def bayesian_prestige(df, id_col, label_col, k=5):
    # Bayesian smoothing: prestige = (n / (n + k)) * person_mean + (k / (n + k)) * global_mean
    global_mean = df[label_col].mean()
    stats = df.groupby(id_col)[label_col].agg(["mean", "count"]).reset_index()
    stats.columns = [id_col, "person_mean", "n"]
    stats["prestige"] = (stats["n"] / (stats["n"] + k)) * stats["person_mean"] + \
                        (k / (stats["n"] + k)) * global_mean
    return stats[[id_col, "prestige"]], global_mean

def run():
    train_path = "/app/processed/train.parquet"
    if not os.path.exists(train_path):
        return

    train = pd.read_parquet(train_path)
    train["label_int"] = (train["label"] == "True") | (train["label"] == True)
    train["label_int"] = train["label_int"].astype(int)

    directing = load_directing("/app/imdb/directing.json")
    writing = load_writing("/app/imdb/writing.json")

    # one prestige score per movie: average across co-directors/co-writers
    train_dir = train.merge(directing, on="tconst", how="left")
    train_wri = train.merge(writing, on="tconst", how="left")

    dir_prestige, dir_global = bayesian_prestige(
        train_dir.dropna(subset=["director_id"]), "director_id", "label_int"
    )
    wri_prestige, wri_global = bayesian_prestige(
        train_wri.dropna(subset=["writer_id"]), "writer_id", "label_int"
    )

    def add_prestige(df):
        # director prestige: mean prestige score across all directors of the film
        dir_scores = directing.merge(dir_prestige, on="director_id", how="left")
        dir_scores["prestige"] = dir_scores["prestige"].fillna(dir_global)
        dir_mean = dir_scores.groupby("tconst")["prestige"].mean().reset_index()
        dir_mean.columns = ["tconst", "director_prestige"]

        # writer prestige: mean prestige score across all writers of the film
        wri_scores = writing.merge(wri_prestige, on="writer_id", how="left")
        wri_scores["prestige"] = wri_scores["prestige"].fillna(wri_global)
        wri_mean = wri_scores.groupby("tconst")["prestige"].mean().reset_index()
        wri_mean.columns = ["tconst", "writer_prestige"]

        df = df.merge(dir_mean, on="tconst", how="left")
        df = df.merge(wri_mean, on="tconst", how="left")
        df["director_prestige"] = df["director_prestige"].fillna(dir_global)
        df["writer_prestige"] = df["writer_prestige"].fillna(wri_global)
        return df

    train = add_prestige(train)
    train.drop(columns=["label_int"], inplace=True)
    train.to_parquet(train_path, index=False)

    for split in ["validation_hidden", "test_hidden"]:
        path = f"/app/processed/{split}.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = add_prestige(df)
            df.to_parquet(path, index=False)

    print(f"Prestige scores added. Dir global mean: {dir_global:.3f}, Wri global mean: {wri_global:.3f}")

if __name__ == "__main__":
    run()
