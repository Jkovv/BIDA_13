import csv, gzip, pandas as pd, numpy as np

val = pd.read_csv('imdb/validation_hidden.csv')
val_tconsts = set(val['tconst'].tolist())
print(f"Validation movies: {len(val_tconsts)}")

with gzip.open('imdb/title.ratings.tsv.gz', 'rt') as f:
    ratings = pd.read_csv(f, sep='\t', na_values='\\N')

print(f"Ratings columns: {list(ratings.columns)}")

val_ratings = ratings[ratings['tconst'].isin(val_tconsts)].copy()
print(f"Matched {len(val_ratings)} / {len(val_tconsts)} validation movies")

# merge preserving val order
val_with_ratings = val.merge(val_ratings, on='tconst', how='left')
print(f"Merged columns: {list(val_with_ratings.columns)}")

# detect vote/rating column names dynamically
rating_col = [c for c in val_with_ratings.columns if 'rating' in c.lower()][0]
votes_col = [c for c in val_with_ratings.columns if 'vote' in c.lower()][0]
print(f"Using: {rating_col}, {votes_col}")

with open('output/validation.txt') as f:
    our_preds = [line.strip() == 'True' for line in f]
val_with_ratings['our_pred'] = our_preds

print("\n--- Threshold search ---")
print(f"{'rating':>8} {'votes':>8} {'%pos':>6} {'match':>8}")
for rating_thresh in [6.0, 6.5, 7.0, 7.5, 8.0]:
    for vote_thresh in [0, 100, 500, 1000, 5000]:
        guessed = (
            (val_with_ratings[rating_col] >= rating_thresh) &
            (val_with_ratings[votes_col] >= vote_thresh)
        ).fillna(False)
        match = (guessed == val_with_ratings['our_pred']).mean()
        print(f"{rating_thresh:>8} {vote_thresh:>8} {guessed.mean():>6.1%} {match:>8.4f}")
