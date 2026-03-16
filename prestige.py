import json
import pandas as pd
import os


def load_directing(path="/app/imdb/directing.json"):
    with open(path) as f:
        d = json.load(f)
    return pd.DataFrame({"tconst": list(d["movie"].values()),
                          "director_id": list(d["director"].values())})

def load_writing(path="/app/imdb/writing.json"):
    with open(path) as f:
        w = json.load(f)
    return pd.DataFrame({"tconst": [x["movie"] for x in w],
                          "writer_id": [x["writer"] for x in w]})


def bayesian_prestige(df, id_col, label_col, k=20):
    """k=20 means a director needs 20+ films before their own mean dominates
    over the global mean. prevents single-film directors from getting prestige == their label"""
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

    print("Step 3: Prestige scores")
    train = pd.read_parquet(train_path)
    train["label_int"] = ((train["label"] == "True") | (train["label"] == True)).astype(int)

    directing = load_directing()
    writing = load_writing()

    # how many directors/writers per movie (structural, no labels involved)
    dir_count = directing.groupby("tconst").size().reset_index(name="n_directors")
    wri_count = writing.groupby("tconst").size().reset_index(name="n_writers")

    train_dir = train.merge(directing, on="tconst", how="left")
    train_wri = train.merge(writing, on="tconst", how="left")

    dir_prestige, dir_global = bayesian_prestige(
        train_dir.dropna(subset=["director_id"]), "director_id", "label_int")
    wri_prestige, wri_global = bayesian_prestige(
        train_wri.dropna(subset=["writer_id"]), "writer_id", "label_int")

    def add_prestige(df):
        # average prestige across all directors of the movie
        ds = directing.merge(dir_prestige, on="director_id", how="left")
        ds["prestige"] = ds["prestige"].fillna(dir_global)
        dm = ds.groupby("tconst")["prestige"].mean().reset_index()
        dm.columns = ["tconst", "director_prestige"]

        ws = writing.merge(wri_prestige, on="writer_id", how="left")
        ws["prestige"] = ws["prestige"].fillna(wri_global)
        wm = ws.groupby("tconst")["prestige"].mean().reset_index()
        wm.columns = ["tconst", "writer_prestige"]

        df = df.merge(dm, on="tconst", how="left")
        df = df.merge(wm, on="tconst", how="left")
        df = df.merge(dir_count, on="tconst", how="left")
        df = df.merge(wri_count, on="tconst", how="left")

        df["director_prestige"] = df["director_prestige"].fillna(dir_global)
        df["writer_prestige"] = df["writer_prestige"].fillna(wri_global)
        df["n_directors"] = df["n_directors"].fillna(1).astype(int)
        df["n_writers"] = df["n_writers"].fillna(1).astype(int)
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

    print(f"  dir global: {dir_global:.3f}, wri global: {wri_global:.3f}")

    # NOTE: run.py recomputes prestige inside each CV fold to avoid leaking
    # validation labels into these scores. the columns here are only used
    # for the final model training + test predictions.


if __name__ == "__main__":
    run()