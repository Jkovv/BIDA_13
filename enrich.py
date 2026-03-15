import os
import gzip
import urllib.request
import pandas as pd

BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
BASICS_PATH = "/app/imdb/title.basics.tsv.gz"

def download_basics():
    if not os.path.exists(BASICS_PATH):
        print("Downloading title.basics.tsv.gz from IMDB...")
        urllib.request.urlretrieve(BASICS_URL, BASICS_PATH)
        print("Download complete.")
    else:
        print("title.basics.tsv.gz already present, skipping download.")

def load_genres():
    with gzip.open(BASICS_PATH, 'rt', encoding='utf-8') as f:
        basics = pd.read_csv(f, sep='\t', usecols=['tconst', 'genres'], na_values='\\N', low_memory=False)
    # genres is comma-separated e.g. "Drama,Romance" - one-hot encode top genres
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

    download_basics()
    genres = load_genres()

    for name in ["train", "validation_hidden", "test_hidden"]:
        path = f"/app/processed/{name}.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = df.merge(genres, on="tconst", how="left")
            # fill missing genres with 0 (movie not in IMDB basics)
            genre_cols = [c for c in df.columns if c.startswith('genre_')]
            df[genre_cols] = df[genre_cols].fillna(0).astype(int)
            df.to_parquet(path, index=False)

    print(f"Genres joined. {genres.shape[0]} titles in basics, top genres: {[g.lower() for g in ['Drama','Comedy','Action','Romance','Thriller']]}")

if __name__ == "__main__":
    run()
