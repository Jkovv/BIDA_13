import os
import zipfile
import urllib.request
import json
import pandas as pd
import numpy as np

# movielens
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ML_ZIP = "/app/imdb/ml-25m.zip"
ML_DIR = "/app/imdb/ml-25m"

# hand-picked MovieLens tag relevance features
# MovieLens assigns continuous [0,1] relevance scores per movie for ~1000
# user-generated tags. we pick ones relevant to quality/reception.
MOVIELENS_TAGS = [
    "thought-provoking", "masterpiece", "atmospheric", "cult film",
    "visually appealing", "feel-good", "boring", "predictable",
    "original", "oscar (best picture)", "critically acclaimed",
    "dark", "twist ending", "funny", "violent", "slow",
    "cinematography", "surreal", "inspiring", "dialogue",
]

# from scipy import stats
# N_SELECTED_TAGS = 50
#
# def select_tags_by_correlation(train_tconsts_labels, links, top_n=N_SELECTED_TAGS):
#     scores = pd.read_csv(os.path.join(ML_DIR, "genome-scores.csv"))
#     tags = pd.read_csv(os.path.join(ML_DIR, "genome-tags.csv"))
#     scores = scores.merge(links[["movieId", "tconst"]], on="movieId", how="inner")
#     pivoted = scores.pivot_table(index="tconst", columns="tagId", values="relevance", aggfunc="first")
#     train_data = train_tconsts_labels.join(pivoted, how="inner")
#     y = train_data["label_int"].values
#     correlations = {}
#     for tag_id in [c for c in train_data.columns if c != "label_int"]:
#         x = train_data[tag_id].values
#         if x.std() > 0:
#             r, _ = stats.pointbiserialr(y, x)
#             correlations[tag_id] = abs(r)
#     top_tag_ids = sorted(correlations, key=correlations.get, reverse=True)[:top_n]
#     tag_name_map = tags.set_index("tagId")["tag"].to_dict()
#     selected = [(tid, tag_name_map[tid]) for tid in top_tag_ids if tid in tag_name_map]
#     return selected
# --

# Bechdel Test (bechdeltest.com) 
# measures female representation in fiction (score 0-3)
# independent of IMDB, joinable via imdb_id
BECHDEL_URL = "https://bechdeltest.com/api/v1/getAllMovies"
BECHDEL_CACHE = "/app/imdb/bechdel_all.json"


def download_and_extract_ml():
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
    try:
        return f"tt{int(imdb_id):07d}"
    except (ValueError, TypeError):
        return None


def load_tag_relevance_features(links):
    """Two approaches combined, both independently validated by statistical pipeline:
    1) 19 hand-picked tags as individual features (mltag_boring, etc.)
    2) PCA on ALL 1,128 tags → 30 components (mltag_pc_0, etc.)
    
    Hand-picked tags give direct split-friendly signal (mltag_boring=0.92).
    PCA components capture cross-tag quality patterns.
    Spearman redundancy (step 4) confirms no individual tag correlates >0.85
    with any PCA component — they provide complementary information."""
    from sklearn.decomposition import PCA

    genome_scores = pd.read_csv(os.path.join(ML_DIR, "genome-scores.csv"))
    genome_tags = pd.read_csv(os.path.join(ML_DIR, "genome-tags.csv"))

    # pivot ALL tags into a wide matrix (movieId x tagId)
    all_pivoted = genome_scores.pivot_table(
        index="movieId", columns="tagId", values="relevance", aggfunc="first"
    )
    all_pivoted = all_pivoted.fillna(0)
    print(f"  Full tag matrix: {all_pivoted.shape[0]} movies x {all_pivoted.shape[1]} tags")

    # PCA on the full matrix
    N_COMPONENTS = 30
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    components = pca.fit_transform(all_pivoted.values)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {N_COMPONENTS} components, {explained:.1%} variance explained")

    pca_df = pd.DataFrame(
        components,
        columns=[f"mltag_pc_{i}" for i in range(N_COMPONENTS)],
        index=all_pivoted.index
    ).reset_index()

    # 19 hand-picked tags as individual features
    selected_tags = genome_tags[genome_tags["tag"].str.lower().isin(
        [t.lower() for t in MOVIELENS_TAGS]
    )].copy()
    selected_tags["col_name"] = "mltag_" + selected_tags["tag"].str.lower() \
        .str.replace(" ", "_").str.replace("-", "_").str.replace("(", "").str.replace(")", "")

    filtered = genome_scores[genome_scores["tagId"].isin(selected_tags["tagId"])]
    filtered = filtered.merge(selected_tags[["tagId", "col_name"]], on="tagId")
    hand_picked = filtered.pivot_table(
        index="movieId", columns="col_name", values="relevance", aggfunc="first"
    ).reset_index()
    hand_picked.columns.name = None

    # merge PCA + hand-picked
    result = pca_df.merge(hand_picked, on="movieId", how="outer")
    result = result.merge(links[["movieId", "tconst"]], on="movieId", how="inner")
    result = result.drop(columns=["movieId"])

    n_features = len([c for c in result.columns if c != "tconst"])
    print(f"  Tag features: {len(result)} movies, {n_features} features "
          f"(19 hand-picked + {N_COMPONENTS} PCA)")
    return result


def load_movielens_features():
    links = pd.read_csv(os.path.join(ML_DIR, "links.csv"))
    links["tconst"] = links["imdbId"].apply(tconst_from_imdbid)

    movies = pd.read_csv(os.path.join(ML_DIR, "movies.csv"))
    movies = movies.merge(links[["movieId", "tconst"]], on="movieId", how="inner")

    top_genres = ['Drama', 'Comedy', 'Action', 'Romance', 'Thriller',
                  'Crime', 'Adventure', 'Horror', 'Documentary', 'Animation',
                  'Sci-Fi', 'Mystery', 'Fantasy', 'War', 'Musical',
                  'Western', 'Film-Noir', 'Children', 'IMAX']
    for g in top_genres:
        col = f'genre_{g.lower().replace("-", "_")}'
        movies[col] = movies['genres'].str.contains(g, na=False).astype(int)
    movies['genre_count'] = movies['genres'].str.count('\\|').add(1).fillna(0).astype(int)

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

    tags = pd.read_csv(os.path.join(ML_DIR, "tags.csv"))
    tag_counts = tags.groupby("movieId").size().reset_index(name="ml_tag_count")
    movies = movies.merge(tag_counts, on="movieId", how="left")
    movies["ml_tag_count"] = movies["ml_tag_count"].fillna(0).astype(int)

    tag_rel = load_tag_relevance_features(links)
    movies = movies.merge(tag_rel, on="tconst", how="left")

    genre_cols = [c for c in movies.columns if c.startswith('genre_')]
    mltag_cols = [c for c in movies.columns if c.startswith('mltag_')]
    ml_cols = ["ml_rating_mean", "ml_rating_std", "ml_rating_count",
               "ml_rating_median", "ml_log_count", "ml_tag_count"]
    keep = ["tconst"] + genre_cols + ml_cols + mltag_cols
    return movies[keep]


def load_bechdel_features():
    """Bechdel test scores (0-3) from bechdeltest.com."""
    if not os.path.exists(BECHDEL_CACHE):
        print("  Downloading Bechdel test data...")
        try:
            req = urllib.request.Request(BECHDEL_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            with open(BECHDEL_CACHE, 'w') as f:
                json.dump(data, f)
            print(f"  Downloaded {len(data)} movies")
        except Exception as e:
            print(f"  Bechdel download failed: {e}, skipping")
            return None
    else:
        print("  Bechdel data cached.")
        with open(BECHDEL_CACHE) as f:
            data = json.load(f)

    df = pd.DataFrame(data)
    df["tconst"] = df["imdbid"].apply(tconst_from_imdbid)
    df["bechdel_score"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(int)
    df["bechdel_pass"] = (df["bechdel_score"] == 3).astype(int)
    return df[["tconst", "bechdel_score", "bechdel_pass"]].drop_duplicates(subset=["tconst"])


# TMDB community ratings — different user base from both IMDB and MovieLens
TMDB_PATH = "/app/imdb/tmdb_metadata.csv"

def load_tmdb_ratings():
    """extract just vote_average and vote_count from TMDB metadata.
    NOT budget/revenue (those failed MI due to 31% missing). the ratings
    have much better coverage."""
    if not os.path.exists(TMDB_PATH):
        print("  tmdb_metadata.csv not found, skipping TMDB ratings")
        return None

    df = pd.read_csv(TMDB_PATH, low_memory=False)
    if "imdb_id" not in df.columns:
        print("  TMDB CSV missing imdb_id, skipping")
        return None

    df["tconst"] = df["imdb_id"].astype(str).str.strip()
    df["tmdb_vote_avg"] = pd.to_numeric(df.get("vote_average"), errors="coerce")
    df["tmdb_vote_count"] = pd.to_numeric(df.get("vote_count"), errors="coerce")
    df["tmdb_log_votes"] = np.log1p(df["tmdb_vote_count"])

    # drop movies with no ratings at all (vote_count == 0 or NaN)
    # these would get median-imputed anyway but keeping them as NaN
    # lets the join handle it cleanly
    df.loc[df["tmdb_vote_count"] < 1, ["tmdb_vote_avg", "tmdb_vote_count", "tmdb_log_votes"]] = np.nan

    out = df[["tconst", "tmdb_vote_avg", "tmdb_vote_count", "tmdb_log_votes"]].drop_duplicates(subset=["tconst"])
    has_ratings = out["tmdb_vote_avg"].notna().sum()
    print(f"  TMDB ratings: {has_ratings}/{len(out)} movies with votes")
    return out


def run():
    if not os.path.exists("/app/processed/train.parquet"):
        return

    print("Step 4: External data enrichment")

    # MovieLens
    download_and_extract_ml()
    ml_features = load_movielens_features()
    ml_cols = [c for c in ml_features.columns if c != "tconst"]

    # Bechdel
    bechdel_features = load_bechdel_features()

    # TMDB ratings (not budget/revenue — those failed)
    tmdb_features = load_tmdb_ratings()

    for name in ["train", "validation_hidden", "test_hidden"]:
        path = f"/app/processed/{name}.parquet"
        if not os.path.exists(path):
            continue

        df = pd.read_parquet(path)
        before = len(df)

        # MovieLens
        df = df.drop(columns=[c for c in ml_cols if c in df.columns], errors="ignore")
        df = df.merge(ml_features, on="tconst", how="left")
        for col in ml_cols:
            if col.startswith("genre_"):
                df[col] = df[col].fillna(0).astype(int)
            elif col in ["ml_tag_count", "ml_rating_count"]:
                df[col] = df[col].fillna(0)
            elif col.startswith("mltag_"):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].median())
        ml_matched = df["ml_rating_mean"].notna().sum()

        # Bechdel
        bechdel_matched = 0
        if bechdel_features is not None:
            bech_cols = ["bechdel_score", "bechdel_pass"]
            df = df.drop(columns=[c for c in bech_cols if c in df.columns], errors="ignore")
            df = df.merge(bechdel_features, on="tconst", how="left")
            df["bechdel_score"] = df["bechdel_score"].fillna(0).astype(int)
            df["bechdel_pass"] = df["bechdel_pass"].fillna(0).astype(int)
            bechdel_matched = (df["bechdel_score"] > 0).sum()

        # TMDB ratings - median imputation for missing (NOT zero-fill)
        tmdb_matched = 0
        if tmdb_features is not None:
            tmdb_cols = ["tmdb_vote_avg", "tmdb_vote_count", "tmdb_log_votes"]
            df = df.drop(columns=[c for c in tmdb_cols if c in df.columns], errors="ignore")
            df = df.merge(tmdb_features, on="tconst", how="left")
            for col in tmdb_cols:
                df[col] = df[col].fillna(df[col].median())
            tmdb_matched = (df["tmdb_vote_count"] > 0).sum()

        df.to_parquet(path, index=False)
        print(f"  {name}: ML={ml_matched}/{before}, Bechdel={bechdel_matched}/{before}"
              f", TMDB={tmdb_matched}/{before}")

    # failed enrichments documented for poster:
    # - Oscar awards: 11% match, failed MW+MI tests (awards.py)
    # - TSPDT 1000: 2.6% match, same (prestige_lists.py)
    # - Criterion: 3.4% match, same (prestige_lists.py)
    # - Director-DP loyalty: 10% match, failed MI (loyalty.py)
    # - TMDB budget/revenue/popularity/language: 31% missing budget, all failed permutation MI
    # - TMDB vote_average/vote_count: tested separately (different coverage than budget)
    print()


if __name__ == "__main__":
    run()
