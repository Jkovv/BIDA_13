import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np

# MovieLens 25M - GroupLens, University of Minnesota
# different source/userbase from IMDB — allowed as external data
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ML_ZIP = "/app/imdb/ml-25m.zip"
ML_DIR = "/app/imdb/ml-25m"

# hand-picked genome tags that are predictive of quality/prestige
# covers: artistic merit, audience reception, cultural impact, tone
GENOME_TAGS = [
    "thought-provoking", "masterpiece", "atmospheric", "cult film",
    "visually appealing", "feel-good", "boring", "predictable",
    "original", "oscar (best picture)", "critically acclaimed",
    "dark", "twist ending", "funny", "violent", "slow",
    "cinematography", "surreal", "inspiring", "dialogue",
]


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
    return f"tt{int(imdb_id):07d}"


def load_genome_features(links):
    """Load tag genome: relevance scores for selected tags per movie.
    genome-scores.csv has (movieId, tagId, relevance) for ~1000 tags x 10k movies.
    relevance is a continuous [0,1] score — how well the tag describes the movie."""
    genome_scores = pd.read_csv(os.path.join(ML_DIR, "genome-scores.csv"))
    genome_tags = pd.read_csv(os.path.join(ML_DIR, "genome-tags.csv"))

    # filter to our selected tags only
    selected_tags = genome_tags[genome_tags["tag"].str.lower().isin(
        [t.lower() for t in GENOME_TAGS]
    )].copy()
    selected_tags["col_name"] = "genome_" + selected_tags["tag"].str.lower() \
        .str.replace(" ", "_").str.replace("-", "_").str.replace("(", "").str.replace(")", "")

    filtered = genome_scores[genome_scores["tagId"].isin(selected_tags["tagId"])]
    filtered = filtered.merge(selected_tags[["tagId", "col_name"]], on="tagId")

    # pivot: one row per movie, one col per tag
    pivoted = filtered.pivot_table(index="movieId", columns="col_name",
                                   values="relevance", aggfunc="first").reset_index()
    pivoted.columns.name = None

    # join tconst
    pivoted = pivoted.merge(links[["movieId", "tconst"]], on="movieId", how="inner")
    pivoted = pivoted.drop(columns=["movieId"])

    print(f"  Genome: {len(pivoted)} movies, {len(pivoted.columns)-1} tag features")
    return pivoted


def load_movielens_features():
    links = pd.read_csv(os.path.join(ML_DIR, "links.csv"))
    links["tconst"] = links["imdbId"].apply(tconst_from_imdbid)

    movies = pd.read_csv(os.path.join(ML_DIR, "movies.csv"))
    movies = movies.merge(links[["movieId", "tconst"]], on="movieId", how="inner")

    # genres from MovieLens
    top_genres = ['Drama', 'Comedy', 'Action', 'Romance', 'Thriller',
                  'Crime', 'Adventure', 'Horror', 'Documentary', 'Animation',
                  'Sci-Fi', 'Mystery', 'Fantasy', 'War', 'Musical',
                  'Western', 'Film-Noir', 'Children', 'IMAX']
    for g in top_genres:
        col = f'genre_{g.lower().replace("-", "_")}'
        movies[col] = movies['genres'].str.contains(g, na=False).astype(int)
    movies['genre_count'] = movies['genres'].str.count('\\|').add(1).fillna(0).astype(int)

    # aggregate ratings from 25M user ratings
    ratings = pd.read_csv(os.path.join(ML_DIR, "ratings.csv"))
    agg = ratings.groupby("movieId").agg(
        ml_rating_mean=("rating", "mean"),
        ml_rating_std=("rating", "std"),
        ml_rating_count=("rating", "count"),
        ml_rating_median=("rating", "median"),
    ).reset_index()
    agg["ml_rating_std"] = agg["ml_rating_std"].fillna(0)
    agg["ml_log_count"] = np.log1p(agg["ml_rating_count"])
    movies = movies.merge(agg, on="movieId", how="left")

    # user tag counts
    tags = pd.read_csv(os.path.join(ML_DIR, "tags.csv"))
    tag_counts = tags.groupby("movieId").size().reset_index(name="ml_tag_count")
    movies = movies.merge(tag_counts, on="movieId", how="left")
    movies["ml_tag_count"] = movies["ml_tag_count"].fillna(0).astype(int)

    # genome features
    genome = load_genome_features(links)
    movies = movies.merge(genome, on="tconst", how="left")

    genre_cols = [c for c in movies.columns if c.startswith('genre_')]
    genome_cols = [c for c in movies.columns if c.startswith('genome_')]
    ml_cols = ["ml_rating_mean", "ml_rating_std", "ml_rating_count",
               "ml_rating_median", "ml_log_count", "ml_tag_count"]
    keep = ["tconst"] + genre_cols + ml_cols + genome_cols
    return movies[keep]


def run():
    if not os.path.exists("/app/processed/train.parquet"):
        return

    print("Step 4: MovieLens enrichment (ratings + genres + genome)")
    download_and_extract()

    ml_features = load_movielens_features()
    new_cols = [c for c in ml_features.columns if c != "tconst"]
    genome_cols = [c for c in new_cols if c.startswith("genome_")]
    print(f"  {len(ml_features)} movies total, {len(new_cols)} features "
          f"({len(genome_cols)} genome tags)")

    for name in ["train", "validation_hidden", "test_hidden"]:
        path = f"/app/processed/{name}.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df = df.drop(columns=[c for c in new_cols if c in df.columns], errors="ignore")
            before = len(df)
            df = df.merge(ml_features, on="tconst", how="left")

            matched = df["ml_rating_mean"].notna().sum()

            for col in new_cols:
                if col.startswith("genre_"):
                    df[col] = df[col].fillna(0).astype(int)
                elif col in ["ml_tag_count", "ml_rating_count"]:
                    df[col] = df[col].fillna(0)
                elif col.startswith("genome_"):
                    # fill missing genome scores with column median
                    # (movies not in genome get average tag relevance)
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].median())

            df.to_parquet(path, index=False)
            print(f"  {name}: matched {matched}/{before} movies")

    print()


if __name__ == "__main__":
    run()
