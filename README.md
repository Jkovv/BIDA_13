# IMDB Movie Rating Classifier - Group 13

Binary classification pipeline to predict whether a movie is highly rated on IMDB.

**Validation accuracy: 88.59%** | [IMDB Leaderboard](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/competitions/imdb)

---

### Pipeline Overview

```
cleaning.py     DuckDB   Raw CSVs → cleaned parquet (unicode fix, winsorization, log_votes, film_age, vote_density)
prestige.py     Pandas   Director/writer Bayesian prestige scores (k=20), recomputed per CV fold in run.py
enrich.py       Pandas   MovieLens 25M ratings + genres + genome tag features joined via tconst
features.py     PySpark  TF-IDF on cleaned titles → feature parquet
run.py          sklearn  4-stage feature selection + model competition (XGB/LGBM/CatBoost/RF/GBM) + grid search → predictions
```

**Not in pipeline (kept for analysis):**
```
awards.py          Oscar nominations/wins — title join matched ~11%, too sparse for feature selection
prestige_lists.py  TSPDT rank + Criterion Collection — title join matched ~2-3%, same issue
loyalty.py         Director-DP collaboration count from TMDB credits — matched ~10%, failed MI test
```

### How to Run

**Requirements:** Docker Desktop running

```bash
docker compose up
```

Or manually:

```bash
docker build -t imdb-project .
docker run -it \
  -v "$(pwd)/imdb:/app/imdb" \
  -v "$(pwd)/output:/app/output" \
  imdb-project
```

Predictions are saved to `output/validation.txt` and `output/test.txt`.

### Data

Place the following files in the `imdb/` folder before running:

```
imdb/
  train-*.csv
  validation_hidden.csv
  test_hidden.csv
  directing.json
  writing.json
```

MovieLens 25M (~262MB) is downloaded automatically on first run and cached after.

Additional datasets used for analysis (not required to run pipeline):
- TSPDT 1000 Greatest Films: https://theyshootpictures.com/gf1000_all1000films_table.php → save as `tspdt.csv`
- Criterion Collection: https://www.kaggle.com/datasets/shankhadeepmaiti/the-criterion-collection → `criterion.csv`
- TMDB credits + metadata: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset → `tmdb_credits.csv` (needs unzipping) + `tmdb_metadata.csv`

### Features Used

| Feature | Source | Description |
|---|---|---|
| `log_votes` | train CSV | log1p(numVotes) |
| `runtime` | train CSV | winsorized to [1, 210] min |
| `film_age` | train CSV | 2026 - release year |
| `votes_x_runtime` | train CSV | interaction term |
| `writer_prestige` | writing.json | Bayesian-smoothed writer label rate (k=20) |
| `ml_rating_mean` | MovieLens 25M | mean user rating across 25M ratings |
| `ml_rating_std` | MovieLens 25M | rating variance — polarizing films score lower |
| `ml_log_count` | MovieLens 25M | log number of ratings |
| `ml_tag_count` | MovieLens 25M | number of user tags |
| `genre_*` | MovieLens 25M | one-hot genre flags (drama, horror, sci-fi, etc.) |
| `genome_*` | MovieLens tag genome | continuous [0,1] relevance scores for 19 semantic tags |

### Feature Selection Pipeline (run.py)

Four-stage statistical selection before any model sees the data:
1. **Mann-Whitney U + Benjamini-Hochberg FDR** (alpha=0.05) — univariate significance
2. **Permutation mutual information + BH** (alpha=0.1) — nonlinear signal check
3. **Partial Spearman** — drops interaction features redundant with their components
4. **Spearman redundancy pruning** (|rho| > 0.85) — removes collinear features

### Notes

**Local CV accuracy (~88%) is trustworthy** — prestige scores are recomputed inside each CV fold so validation labels don't leak into the features.

**Genome tag selection:** hand-picked 19 tags covering artistic merit, tone, and audience reception (boring, predictable, masterpiece, atmospheric, etc.). We also implemented data-driven selection via point-biserial correlation across all 1,128 genome tags (top 50 by |r|) — same validation accuracy, kept commented in `enrich.py` for reference.

**Failed enrichments (Analysis talking points):**
- Oscar awards, TSPDT rank, Criterion Collection: institutionally meaningful signals but title+year fuzzy join coverage was too low (2–11%) to pass statistical significance tests on our 8k movie dataset
- Director-DP loyalty (TMDB): signal exists in principle but insufficient training examples with prior collaborations
- These failures are themselves informative — the pipeline's statistical gatekeeping correctly identified that sparse features hurt more than they help

### Submitting predictions

Each line in `output/validation.txt` and `output/test.txt` is either `True` or `False` for the corresponding row in the hidden CSV files.

[Submission server](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/)
