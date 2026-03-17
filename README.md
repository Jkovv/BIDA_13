# IMDB Movie Rating Classifier - Group 13

Binary classification pipeline to predict whether a movie is highly rated on IMDB.

**Validation accuracy: 88.59%** | [IMDB Leaderboard](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/competitions/imdb)

---

### Pipeline Overview

```
cleaning.py     DuckDB   Raw CSVs -> cleaned parquet (unicode fix, winsorization, log_votes, film_age)
prestige.py     Pandas   Director/writer Bayesian prestige scores (k=20), recomputed per CV fold in run.py
enrich.py       Pandas   MovieLens 25M ratings + genres + genome tag features joined via tconst
features.py     PySpark  TF-IDF on cleaned titles -> feature parquet
run.py          sklearn  4-stage feature selection + model competition (XGB/LGBM/CatBoost/RF/GBM) + grid search -> predictions
```

Not in pipeline (kept for analysis):
```
awards.py          Oscar nominations/wins -- title join matched ~11%, too sparse for feature selection
prestige_lists.py  TSPDT rank + Criterion Collection -- title join matched ~2-3%, same issue
loyalty.py         Director-DP collaboration count from TMDB credits -- matched ~10%, failed MI test
```

---

### How to Run

Requirements: Docker Desktop running

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

---

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

Additional datasets used for analysis (not required to run the pipeline):
- TSPDT 1000 Greatest Films: https://theyshootpictures.com/gf1000_all1000films_table.php -> save as `tspdt.csv`
- Criterion Collection: https://www.kaggle.com/datasets/shankhadeepmaiti/the-criterion-collection -> `criterion.csv`
- TMDB credits + metadata: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset -> `tmdb_credits.csv` (needs unzipping) + `tmdb_metadata.csv`

---

### Features Used

| Feature | Source | Description |
|---|---|---|
| `log_votes` | train CSV | log1p(numVotes) |
| `runtime` | train CSV | winsorized to [1, 210] min |
| `film_age` | train CSV | 2026 - release year |
| `votes_x_runtime` | train CSV | interaction term |
| `writer_prestige` | writing.json | Bayesian-smoothed writer label rate (k=20) |
| `ml_rating_mean` | MovieLens 25M | mean user rating across 25M ratings |
| `ml_rating_std` | MovieLens 25M | rating variance (polarizing films score lower) |
| `ml_log_count` | MovieLens 25M | log number of ratings |
| `ml_tag_count` | MovieLens 25M | number of user tags |
| `genre_*` | MovieLens 25M | one-hot genre flags (drama, horror, sci-fi, etc.) |
| `genome_*` | MovieLens tag genome | continuous [0,1] relevance scores for 19 semantic tags |

---

### Feature Selection Pipeline (run.py)

Four-stage statistical selection runs before any model sees the data:
1. **Mann-Whitney U + Benjamini-Hochberg FDR** (alpha=0.05) -- univariate significance per feature
2. **Permutation mutual information + BH** (alpha=0.1) -- catches nonlinear signal that Mann-Whitney misses
3. **Partial Spearman** -- drops interaction features that are redundant with their components
4. **Spearman redundancy pruning** (|rho| > 0.85) -- removes collinear features, keeps one representative

---

### Notes

**Local CV (~88%) is trustworthy** -- prestige scores are recomputed inside each CV fold so validation labels never leak into the prestige features.

**Genome tag selection:** we hand-picked 19 tags covering artistic merit, tone, and audience reception (boring, predictable, masterpiece, atmospheric, etc.). We also implemented a data-driven version that computes point-biserial correlation across all 1,128 genome tags and takes the top 50 by absolute correlation -- same validation accuracy (88.59%), kept commented out in `enrich.py` for reference.

**Failed enrichments (what we tried and why it did not work):**

- Oscar awards, TSPDT rank, Criterion Collection: strong signals in theory but title+year fuzzy join coverage was too low (2-11% of training movies). Features failed both Mann-Whitney and permutation MI tests. Sparse features like these hurt more than they help on an 8k movie dataset.
- Director-DP loyalty (from TMDB credits): the signal exists in principle (recurring director-cinematographer pairings do correlate with quality) but not enough training examples with prior collaborations to pass the statistical tests.
- TMDB metadata (budget, revenue, popularity, original_language): budget and revenue were missing for ~31% of training movies (studios often do not disclose). All four features failed the permutation MI test and were dropped by the selection pipeline.
- Stacking (meta-learner): trained a logistic regression on out-of-fold probabilities from all 5 base models. The meta-learner collapsed to RF-dominant weights (RF: +3.27, XGB: -0.35), meaning the models were not diverse enough for stacking to add value. Reverted to single champion model.

These failures are documented in the repo (`awards.py`, `prestige_lists.py`, `loyalty.py`) as examples of features that the pipeline's statistical gatekeeping correctly rejected.

---

### Submitting Predictions

Each line in `output/validation.txt` and `output/test.txt` is either `True` or `False` for the corresponding row in the hidden CSV files.

[Submission server](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/)
