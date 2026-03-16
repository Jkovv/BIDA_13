import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np

# MovieLens 25M - academic dataset from GroupLens (University of Minnesota)
# NOT from IMDB, so it's allowed as additional data
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ML_ZIP = "/app/imdb/ml-25m.zip"
ML_DIR = "/app/imdb/ml-25m"


def download_and_extract():
    if os.path.exists(ML_DIR):
        print("  ml-25m already extracted.")
        return
    if not os.path.exists(ML_ZIP):
        print("  Downloading MovieLens 25M (~262MB)...")
        urllib.request.urlretrieve(ML_URL, ML_ZIP)
        print("  Done.")
    print("  Extracting...")
    with zipfile.ZipFile(ML_ZIP, 'r') as z:
        z.extractall("/app/imdb/")
    print("  Extracted.")


def tconst_from_imdbid(imdb_id):
    """MovieLens stores imdbId as integer (e.g. 114709), we need tt0114709"""
    return f"tt{int(imdb_id):07d}"


def load_movielens_features():
    # links.csv maps movieId -> imdbId (our join key)
    links = pd.read_csv(os.path.join(ML_DIR, "links.csv"))
    links["tconst"] = links["imdbId"].apply(tconst_from_imdbid)

    # movies.csv has genres
    movies = pd.read_csv(os.path.join(ML_DIR, "movies.csv"))
    movies = movies.merge(links[["movieId", "tconst"]], on="movieId", how="inner")

    # one-hot encode genres
    top_genres = ['Drama', 'Comedy', 'Action', 'Romance', 'Thriller',
                  'Crime', 'Adventure', 'Horror', 'Documentary', 'Animation',
                  'Sci-Fi', 'Mystery', 'Fantasy', 'War', 'Musical',
                  'Western', 'Film-Noir', 'Children', 'IMAX']
    for g in top_genres:
        col = f'genre_{g.lower().replace("-", "_")}'
        movies[col] = movies['genres'].str.contains(g, na=False).astype(int)
    movies['genre_count'] = movies['genres'].str.count('\\|').add(1).fillna(0).astype(int)

    # aggregate ratings per movie from 25M user ratings
    # this is MovieLens community ratings, NOT IMDB ratings - different user base
    ratings = pd.read_csv(os.path.join(ML_DIR, "ratings.csv"))
    agg = ratings.groupby("movieId").agg(
        ml_rating_mean=("rating", "mean"),
        ml_rating_std=("rating", "std"),
        ml_rating_count=("rating", "count"),
        ml_rating_median=("rating", "median"),
    ).reset_index()
    agg["ml_rating_std"] = agg["ml_rating_std"].fillna(0)
    # high variance in ratings = controversial movie
    # high count = popular on MovieLens
    agg["ml_log_count"] = np.log1p(agg["ml_rating_count"])

    movies = movies.merge(agg, on="movieId", how="left")

    # user tags - aggregate tag counts per movie
    tags = pd.read_csv(os.path.join(ML_DIR, "tags.csv"))
    tag_counts = tags.groupby("movieId").size().reset_index(name="ml_tag_count")
    movies = movies.merge(tag_counts, on="movieId", how="left")
    movies["ml_tag_count"] = movies["ml_tag_count"].fillna(0).astype(int)

    # select final columns
    genre_cols = [c for c in movies.columns if c.startswith('genre_')]
    ml_cols = ["ml_rating_mean", "ml_rating_std", "ml_rating_count",
               "ml_rating_median", "ml_log_count", "ml_tag_count"]
    keep = ["tconst"] + genre_cols + ml_cols
    return movies[keep]


def run():
    if not os.path.exists("/app/processed/train.parquet"):
        return

    print("Step 4: MovieLens enrichment")
    download_and_extract()

    ml_features = load_movielens_features()
    new_cols = [c for c in ml_features.columns if c != "tconst"]
    print(f"  {len(ml_features)} movies in MovieLens, {len(new_cols)} features")

    matched_total = 0
    for name in ["train", "validation_hidden", "test_hidden"]:
        path = f"/app/processed/{name}.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = df.drop(columns=[c for c in new_cols if c in df.columns], errors="ignore")
            before = len(df)
            df = df.merge(ml_features, on="tconst", how="left")

            matched = df[new_cols[0]].notna().sum()
            matched_total += matched

            # fill missing: genres=0, ratings=median, counts=0
            for col in new_cols:
                if col.startswith("genre_"):
                    df[col] = df[col].fillna(0).astype(int)
                elif col in ["ml_tag_count", "ml_rating_count"]:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].median())

            df.to_parquet(path, index=False)
            print(f"  {name}: matched {matched}/{before}")

    print(f"  Total matched: {matched_total}\n")


if __name__ == "__main__":
    run()