"""
prestige.py — Bayesian-smoothed director/writer prestige scores.
Computes on training labels, applies to all splits.

NOTE: This script adds prestige to the parquet files for the full pipeline.
However, run.py RECOMPUTES prestige inside each CV fold to avoid label leakage
during cross-validation. The prestige columns written here are only used for
the final model training + predictions (where using full training data is correct).
"""
import json
import pandas as pd
import numpy as np
import os


def load_directing(path):
    with open(path) as f:
        d = json.load(f)
    return pd.DataFrame({"tconst": list(d["movie"].values()), "director_id": list(d["director"].values())})


def load_writing(path):
    with open(path) as f:
        w = json.load(f)
    return pd.DataFrame({"tconst": [x["movie"] for x in w], "writer_id": [x["writer"] for x in w]})


def bayesian_prestige(df, id_col, label_col, k=20):
    """Bayesian smoothed mean: shrinks toward global mean with strength k.
    A person needs k+ films before their own mean dominates."""
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

    print("Step 3: Computing prestige scores")
    train = pd.read_parquet(train_path)
    train["label_int"] = ((train["label"] == "True") | (train["label"] == True)).astype(int)

    directing = load_directing("/app/imdb/directing.json")
    writing = load_writing("/app/imdb/writing.json")

    # Also compute director/writer frequency features
    dir_count = directing.groupby("tconst").size().reset_index(name="n_directors")
    wri_count = writing.groupby("tconst").size().reset_index(name="n_writers")

    train_dir = train.merge(directing, on="tconst", how="left")
    train_wri = train.merge(writing, on="tconst", how="left")

    dir_prestige, dir_global = bayesian_prestige(
        train_dir.dropna(subset=["director_id"]), "director_id", "label_int", k=20
    )
    wri_prestige, wri_global = bayesian_prestige(
        train_wri.dropna(subset=["writer_id"]), "writer_id", "label_int", k=20
    )

    # Also compute director experience (number of movies they directed in training set)
    dir_exp = directing.merge(train[["tconst"]], on="tconst", how="inner")
    dir_exp = dir_exp.groupby("director_id").size().reset_index(name="dir_experience")
    wri_exp = writing.merge(train[["tconst"]], on="tconst", how="inner")
    wri_exp = wri_exp.groupby("writer_id").size().reset_index(name="wri_experience")

    def add_prestige(df):
        # Director prestige: average across all directors of the movie
        dir_scores = directing.merge(dir_prestige, on="director_id", how="left")
        dir_scores["prestige"] = dir_scores["prestige"].fillna(dir_global)
        dir_mean = dir_scores.groupby("tconst")["prestige"].mean().reset_index()
        dir_mean.columns = ["tconst", "director_prestige"]

        # Writer prestige
        wri_scores = writing.merge(wri_prestige, on="writer_id", how="left")
        wri_scores["prestige"] = wri_scores["prestige"].fillna(wri_global)
        wri_mean = wri_scores.groupby("tconst")["prestige"].mean().reset_index()
        wri_mean.columns = ["tconst", "writer_prestige"]

        # Director experience: max experience among directors
        dir_exp_scores = directing.merge(dir_exp, on="director_id", how="left")
        dir_exp_scores["dir_experience"] = dir_exp_scores["dir_experience"].fillna(0)
        dir_exp_max = dir_exp_scores.groupby("tconst")["dir_experience"].max().reset_index()

        # Writer experience
        wri_exp_scores = writing.merge(wri_exp, on="writer_id", how="left")
        wri_exp_scores["wri_experience"] = wri_exp_scores["wri_experience"].fillna(0)
        wri_exp_max = wri_exp_scores.groupby("tconst")["wri_experience"].max().reset_index()

        df = df.merge(dir_mean, on="tconst", how="left")
        df = df.merge(wri_mean, on="tconst", how="left")
        df = df.merge(dir_count, on="tconst", how="left")
        df = df.merge(wri_count, on="tconst", how="left")
        df = df.merge(dir_exp_max, on="tconst", how="left")
        df = df.merge(wri_exp_max, on="tconst", how="left")

        df["director_prestige"] = df["director_prestige"].fillna(dir_global)
        df["writer_prestige"] = df["writer_prestige"].fillna(wri_global)
        df["n_directors"] = df["n_directors"].fillna(1).astype(int)
        df["n_writers"] = df["n_writers"].fillna(1).astype(int)
        df["dir_experience"] = df["dir_experience"].fillna(0).astype(int)
        df["wri_experience"] = df["wri_experience"].fillna(0).astype(int)
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

    print(f"  Dir global mean: {dir_global:.3f}, Wri global mean: {wri_global:.3f}")
    print(f"  Added: director_prestige, writer_prestige, n_directors, n_writers, dir_experience, wri_experience")


if __name__ == "__main__":
    run()