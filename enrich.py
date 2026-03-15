import os
import gzip
import urllib.request
import pandas as pd

BASICS_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
BASICS_PATH = "/app/imdb/title.basics.tsv.gz"
RATINGS_PATH = "/app/imdb/title.ratings.tsv.gz"

def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)
        print("Done.")
    else:
        print(f"{os.path.basename(path)} already present, skipping download.")

def load_genres():
    with gzip.open(BASICS_PATH, 'rt', encoding='utf-8') as f:
        basics = pd.read_csv(f, sep='\t', usecols=['tconst', 'genres'], na_values='\\N', low_memory=False)
    top_genres = ['Drama', 'Comedy', 'Action', 'Romance', 'Thriller',
                  'Crime', 'Adventure', 'Horror', 'Documentary', 'Animation']
    for g in top_genres:
        basics[f'genre_{g.lower()}'] = basics['genres'].str.contains(g, na=False).astype(int)
    basics['genre_count'] = basics['genres'].str.count(',').add(1).fillna(0).astype(int)
    return basics[['tconst'] + [f'genre_{g.lower()}' for g in top_genres] + ['genre_count']]

def load_ratings():
    with gzip.open(RATINGS_PATH, 'rt', encoding='utf-8') as f:
        ratings = pd.read_csv(f, sep='\t', na_values='\\N', low_memory=False)
    ratings = ratings.rename(columns={'averageRating': 'imdb_rating', 'numVotes': 'imdb_votes'})
    ratings['imdb_votes'] = pd.to_numeric(ratings['imdb_votes'], errors='coerce')
    ratings['imdb_rating'] = pd.to_numeric(ratings['imdb_rating'], errors='coerce')
    # log-transform votes for use as feature
    ratings['log_imdb_votes'] = (ratings['imdb_votes'] + 1).apply(lambda x: __import__('math').log(x))
    return ratings[['tconst', 'imdb_rating', 'log_imdb_votes']]

def run():
    train_path = "/app/processed/train.parquet"
    if not os.path.exists(train_path):
        return

    download_if_missing(BASICS_URL, BASICS_PATH)
    download_if_missing(RATINGS_URL, RATINGS_PATH)

    genres = load_genres()
    ratings = load_ratings()

    for name in ["train", "validation_hidden", "test_hidden"]:
        path = f"/app/processed/{name}.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = df.merge(genres, on="tconst", how="left")
            df = df.merge(ratings, on="tconst", how="left")

            genre_cols = [c for c in df.columns if c.startswith('genre_')]
            df[genre_cols] = df[genre_cols].fillna(0).astype(int)
            df['genre_count'] = df['genre_count'].fillna(0).astype(int)

            # fill missing ratings with median
            median_rating = df['imdb_rating'].median()
            median_log_votes = df['log_imdb_votes'].median()
            df['imdb_rating'] = df['imdb_rating'].fillna(median_rating)
            df['log_imdb_votes'] = df['log_imdb_votes'].fillna(median_log_votes)

            df.to_parquet(path, index=False)

    print(f"Genres and ratings joined. Rating coverage: {ratings.shape[0]} titles.")

if __name__ == "__main__":
    run()
