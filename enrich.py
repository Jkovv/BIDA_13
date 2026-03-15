import os
import gzip
import urllib.request
import pandas as pd

BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
BASICS_PATH = "/app/imdb/title.basics.tsv.gz"

def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"  Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        print("  Done.")
    else:
        print(f"  {os.path.basename(path)} already cached.")

def load_genres():
    with gzip.open(BASICS_PATH, 'rt', encoding='utf-8') as f:
        basics = pd.read_csv(f, sep='\t', usecols=['tconst', 'genres'], na_values='\\N', low_memory=False)
    top_genres = ['Drama', 'Comedy', 'Action', 'Romance', 'Thriller',
                  'Crime', 'Adventure', 'Horror', 'Documentary', 'Animation']
    for g in top_genres:
        basics[f'genre_{g.lower()}'] = basics['genres'].str.contains(g, na=False).astype(int)
    basics['genre_count'] = basics['genres'].str.count(',').add(1).fillna(0).astype(int)
    return basics[['tconst'] + [f'genre_{g.lower()}' for g in top_genres] + ['genre_count']]

def run():
    train_path = "/app/processed/train.parquet"
    if not os.path.exists(train_path):
        return

    download_if_missing(BASICS_URL, BASICS_PATH)
    genres = load_genres()

    for name in ["train", "validation_hidden", "test_hidden"]:
        path = f"/app/processed/{name}.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = df.merge(genres, on="tconst", how="left")
            genre_cols = [c for c in df.columns if c.startswith('genre_')]
            df[genre_cols] = df[genre_cols].fillna(0).astype(int)
            df['genre_count'] = df['genre_count'].fillna(0).astype(int)
            df.to_parquet(path, index=False)

    print(f"  Genres joined. Coverage: {genres.shape[0]} titles from IMDB basics.")
    # NOTE: title.ratings.tsv.gz intentionally NOT used — averageRating is the
    # signal the labels are derived from, so using it would be data leakage.

if __name__ == "__main__":
    run()