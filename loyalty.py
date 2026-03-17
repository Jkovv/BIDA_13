# loyalty.py — not in pipeline, kept for analysis and poster
# director-DP (cinematographer) collaboration count from TMDB credits
#   TMDB dataset — https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
#   files needed: tmdb_credits.csv (needs unzipping) + tmdb_metadata.csv
#
# hypothesis: recurring director-DP pairs develop refined visual grammar → higher ratings
# result: matched ~10% of training movies, failed permutation MI test — too sparse
# the signal exists (Scorsese-Ballhaus, Nolan-Pfister type pairings do score higher)
# but not enough examples in our 8k movie dataset to be statistically reliable
#
# to re-enable: add to Dockerfile CMD and place tmdb files in imdb/

# import os
# import pandas as pd
# import ast
#
# CREDITS_PATH = "/app/imdb/tmdb_credits.csv"
# METADATA_PATH = "/app/imdb/tmdb_metadata.csv"
#
#
# def tconst_from_imdbid(imdb_id):
#     try:
#         s = str(imdb_id).strip()
#         if s.startswith("tt"):
#             return s
#         if s in ("nan", ""):
#             return None
#         return f"tt{int(float(s)):07d}"
#     except (ValueError, TypeError):
#         return None
#
#
# def extract_crew_role(crew_str, job_name):
#     try:
#         crew = ast.literal_eval(str(crew_str))
#         return [p["name"] for p in crew if p.get("job", "").lower() == job_name.lower()]
#     except (ValueError, SyntaxError):
#         return []
#
#
# def build_loyalty_features(credits, metadata):
#     metadata = metadata[["id", "imdb_id", "release_date"]].copy()
#     metadata["tconst"] = metadata["imdb_id"].apply(tconst_from_imdbid)
#     metadata["id"] = metadata["id"].astype(str)
#     metadata = metadata.dropna(subset=["tconst"])
#     credits["id"] = credits["id"].astype(str)
#     credits = credits.merge(metadata, on="id", how="inner")
#     credits = credits.dropna(subset=["tconst"])
#     credits["year"] = pd.to_datetime(credits["release_date"], errors="coerce").dt.year
#     credits["directors"] = credits["crew"].apply(lambda x: extract_crew_role(x, "Director"))
#     credits["dps"] = credits["crew"].apply(lambda x: extract_crew_role(x, "Director of Photography"))
#     rows = []
#     for _, row in credits.iterrows():
#         if not row["directors"] or not row["dps"]:
#             continue
#         for d in row["directors"]:
#             for dp in row["dps"]:
#                 rows.append({"tconst": row["tconst"], "year": row["year"], "director": d, "dp": dp})
#     collab_df = pd.DataFrame(rows).dropna().sort_values("year")
#     results = []
#     for _, row in credits.iterrows():
#         if not row["directors"] or not row["dps"]:
#             results.append({"tconst": row["tconst"], "dir_dp_collabs": 0})
#             continue
#         director = row["directors"][0]
#         prior = collab_df[
#             (collab_df["director"] == director) &
#             (collab_df["dp"].isin(row["dps"])) &
#             (collab_df["year"] < row["year"]) &
#             (collab_df["tconst"] != row["tconst"])
#         ]
#         results.append({"tconst": row["tconst"], "dir_dp_collabs": len(prior)})
#     return pd.DataFrame(results).drop_duplicates(subset=["tconst"])
#
#
# def run():
#     if not os.path.exists("/app/processed/train.parquet"):
#         return
#     if not os.path.exists(CREDITS_PATH) or not os.path.exists(METADATA_PATH):
#         print("Step 4d: Director-DP loyalty — files not found, skipping")
#         return
#     print("Step 4d: Director-DP loyalty enrichment")
#     credits = pd.read_csv(CREDITS_PATH)
#     metadata = pd.read_csv(METADATA_PATH, low_memory=False)
#     loyalty = build_loyalty_features(credits, metadata)
#     for name in ["train", "validation_hidden", "test_hidden"]:
#         path = f"/app/processed/{name}.parquet"
#         if not os.path.exists(path):
#             continue
#         df = pd.read_parquet(path)
#         df = df.drop(columns=["dir_dp_collabs"], errors="ignore")
#         df = df.merge(loyalty, on="tconst", how="left")
#         df["dir_dp_collabs"] = df["dir_dp_collabs"].fillna(0).astype(int)
#         df.to_parquet(path, index=False)
#
#
# if __name__ == "__main__":
#     run()
