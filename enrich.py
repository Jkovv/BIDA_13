import os
import gzip
import urllib.request
import pandas as pd

BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
BASICS_PATH = "/app/imdb/title.basics.tsv.gz"
# NOT using title.ratings - that contains averageRating which is basically
# the label signal. using it would be data leakage.


def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"  Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
    else:
        print(f"  {os.path.basename(path)} already cached.")


def load_genres():
    with gzip.open(BASICS_PATH, 'rt', encoding='utf-8') as f:
        basics = pd.read_csv(f, sep='\t', usecols=['tconst', 'genres', 'titleType'],
                             na_values='\\N', low_memory=False)

    top_genres = ['Drama', 'Comedy', 'Action', 'Romance', 'Thriller',
                  'Crime', 'Adventure', 'Horror', 'Documentary', 'Animation',
                  'Sci-Fi', 'Mystery', 'Fantasy', 'War', 'Biography',
                  'Musical', 'Western', 'Family', 'History']
    for g in top_genres:
        col = f'genre_{g.lower().replace("-", "_")}'
        basics[col] = basics['genres'].str.contains(g, na=False).astype(int)

    basics['genre_count'] = basics['genres'].str.count(',').add(1).fillna(0).astype(int)
    basics['is_movie'] = (basics['titleType'] == 'movie').astype(int)
    basics['is_short'] = (basics['titleType'] == 'short').astype(int)
    basics['is_tvmovie'] = (basics['titleType'] == 'tvMovie').astype(int)

    genre_cols = [c for c in basics.columns if c.startswith('genre_')]
    return basics[['tconst'] + genre_cols + ['is_movie', 'is_short', 'is_tvmovie']]


def run():
    if not os.path.exists("/app/processed/train.parquet"):
        return

    print("Step 4: Genre enrichment")
    download_if_missing(BASICS_URL, BASICS_PATH)
    genres = load_genres()
    new_cols = [c for c in genres.columns if c != "tconst"]

    for name in ["train", "validation_hidden", "test_hidden"]:
        path = f"/app/processed/{name}.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = df.drop(columns=[c for c in new_cols if c in df.columns], errors="ignore")
            df = df.merge(genres, on="tconst", how="left")
            for col in new_cols:
                df[col] = df[col].fillna(0).astype(int)
            df.to_parquet(path, index=False)

    print(f"  Added {len(new_cols)} genre/type features\n")


if __name__ == "__main__":
    run()