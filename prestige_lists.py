# prestige_lists.py — not in pipeline, kept for analysis and poster
# attempted enrichment from three curated sources:
#   TSPDT 1000 Greatest Films  — https://theyshootpictures.com/gf1000_all1000films_table.php
#   Criterion Collection       — https://www.kaggle.com/datasets/shankhadeepmaiti/the-criterion-collection (criterion.csv)
#   TMDB credits + metadata    — https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset (tmdb_credits.csv needs unzipping)
#
# result: title+year fuzzy join only matched ~2-3% of training movies for TSPDT/Criterion
# features failed both Mann-Whitney and permutation MI tests — too sparse to be useful
# kept here as documentation of what was tried and why it didn't work
# (see loyalty.py for the TMDB director-DP approach which also failed for the same reason)
#
# to re-enable: add "python prestige_lists.py && python loyalty.py &&" to Dockerfile CMD
# and place tspdt.csv + criterion.csv in imdb/

# import os
# import pandas as pd
# import numpy as np
#
# TSPDT_PATH = "/app/imdb/tspdt.csv"
# CRITERION_PATH = "/app/imdb/criterion.csv"
#
#
# def normalize_title(t):
#     if pd.isna(t):
#         return ""
#     t = str(t).lower().strip()
#     for article in ["the ", "a ", "an ", "le ", "la ", "les ", "il ", "lo ", "die "]:
#         if t.startswith(article):
#             t = t[len(article):]
#             break
#     return t
#
#
# def load_tspdt():
#     if not os.path.exists(TSPDT_PATH):
#         return None
#     df = pd.read_csv(TSPDT_PATH)
#     df = df[["Pos", "Title", "Year"]].copy()
#     df.columns = ["tspdt_rank", "title", "tspdt_year"]
#     df["tspdt_rank"] = pd.to_numeric(df["tspdt_rank"], errors="coerce")
#     df["tspdt_year"] = pd.to_numeric(df["tspdt_year"], errors="coerce")
#     df = df.dropna(subset=["tspdt_rank", "title"])
#     df["tspdt_rank"] = df["tspdt_rank"].astype(int)
#     df["title_norm"] = df["title"].apply(normalize_title)
#     return df[["title_norm", "tspdt_year", "tspdt_rank"]]
#
#
# def load_criterion():
#     if not os.path.exists(CRITERION_PATH):
#         return None
#     df = pd.read_csv(CRITERION_PATH)
#     df = df[["spine", "title", "year"]].copy()
#     df.columns = ["criterion_spine", "title", "criterion_year"]
#     df["criterion_spine"] = pd.to_numeric(df["criterion_spine"], errors="coerce")
#     df["criterion_year"] = pd.to_numeric(df["criterion_year"], errors="coerce")
#     df = df.dropna(subset=["criterion_spine", "title"])
#     df["is_criterion"] = 1
#     df["criterion_spine"] = df["criterion_spine"].astype(int)
#     df["title_norm"] = df["title"].apply(normalize_title)
#     return df[["title_norm", "criterion_year", "is_criterion", "criterion_spine"]]
#
#
# def fuzzy_join(df, lookup, feature_cols, year_col, year_tolerance=2):
#     df = df.copy()
#     df["_title_norm"] = df["title_clean"].apply(normalize_title)
#     df["_year"] = pd.to_numeric(df["year"], errors="coerce")
#     merged = df.merge(lookup, left_on="_title_norm", right_on="title_norm", how="left")
#     if year_col in merged.columns:
#         year_diff = (merged["_year"] - pd.to_numeric(merged[year_col], errors="coerce")).abs()
#         merged.loc[year_diff > year_tolerance, feature_cols] = np.nan
#     merged = merged.drop(columns=["_title_norm", "_year", "title_norm", year_col], errors="ignore")
#     return merged.drop_duplicates(subset=["tconst"])
#
#
# def run():
#     if not os.path.exists("/app/processed/train.parquet"):
#         return
#     print("Step 4c: Prestige lists enrichment")
#     tspdt = load_tspdt()
#     criterion = load_criterion()
#     for name in ["train", "validation_hidden", "test_hidden"]:
#         path = f"/app/processed/{name}.parquet"
#         if not os.path.exists(path):
#             continue
#         df = pd.read_parquet(path)
#         df = df.drop(columns=["tspdt_rank", "is_criterion", "criterion_spine"], errors="ignore")
#         if tspdt is not None:
#             df = fuzzy_join(df, tspdt, ["tspdt_rank"], "tspdt_year")
#             max_rank = tspdt["tspdt_rank"].max()
#             df["tspdt_rank"] = df["tspdt_rank"].fillna(max_rank + 1).astype(int)
#             print(f"  {name} TSPDT: {(df['tspdt_rank'] <= max_rank).sum()}/{len(df)} matched")
#         if criterion is not None:
#             df = fuzzy_join(df, criterion, ["is_criterion", "criterion_spine"], "criterion_year")
#             df["is_criterion"] = df["is_criterion"].fillna(0).astype(int)
#             df["criterion_spine"] = df["criterion_spine"].fillna(0).astype(int)
#             print(f"  {name} Criterion: {df['is_criterion'].sum()}/{len(df)} matched")
#         df.to_parquet(path, index=False)
#
#
# if __name__ == "__main__":
#     run()
