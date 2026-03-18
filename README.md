# IMDB Movie Rating Classifier - Group 13

Binary classification pipeline to predict whether a movie is highly rated on IMDB.

**Validation accuracy: 91.31%** | [IMDB Leaderboard](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/competitions/imdb)

---

### Pipeline Overview

```
cleaning.py     DuckDB   Raw CSVs -> cleaned parquet (unicode fix, winsorization, log_votes, film_age)
prestige.py     Pandas   Director/writer Bayesian prestige scores (k=20), recomputed per CV fold in run.py
enrich.py       Pandas   MovieLens 25M ratings + genres + tag relevance (19 hand-picked + 30 PCA components)
features.py     PySpark  TF-IDF on cleaned titles, extracts top 50 dimensions as numeric columns
run.py          sklearn  4-stage feature selection + model competition (RF/ET) + threshold tuning -> predictions
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
| `mltag_*` | MovieLens tag genome | continuous [0,1] relevance scores for 19 hand-picked semantic tags |
| `mltag_pc_*` | MovieLens tag genome | 30 PCA components over all 1,128 genome tag scores |
| `tfidf_*` | PySpark TF-IDF | top scoring dimensions from title TF-IDF (those that pass feature selection) |

---

### Feature Selection Pipeline (run.py)

Four-stage nonparametric statistical selection runs before any model sees the data:

1. **Mann-Whitney U + Benjamini-Hochberg FDR** (alpha=0.05) -- nonparametric group difference test, no normality assumption. BH correction controls false discovery rate across all features tested simultaneously.
2. **Permutation mutual information + BH** (alpha=0.1) -- real MI compared against a null distribution from 100 shuffled-label runs. A feature only passes if its MI significantly exceeds what you get by chance.
3. **Partial Spearman** -- for derived/interaction features, tests whether they add conditional signal beyond their components. Caught `votes_per_minute` as redundant with `log_votes` and `runtime`.
4. **Spearman redundancy pruning** (|rho| > 0.85) -- pairwise rank correlation to remove collinear features, keeps the one with higher MI.

---

### Score Progression

| Score | What changed |
|---|---|
| 73.2% | Baseline with basic features |
| 79.2% | Director/writer Bayesian prestige scores + MovieLens genres |
| 88.6% | MovieLens tag genome (mltag_boring, mltag_predictable strongest signals) |
| 88.9% | PCA on all 1,128 genome tags -- 26/30 components survived feature selection |
| 91.1% | ExtraTrees replaced Random Forest as champion model + TF-IDF extraction fixed |
| 91.3% | Decision threshold tuned to 0.46 on OOF probabilities (was 0.5 default) |

---

### Notes

**Local CV (~89%) is trustworthy** -- prestige scores are recomputed inside each CV fold so validation labels never leak into the prestige features.

**Tag genome features:** MovieLens assigns continuous [0,1] relevance scores per movie across ~1,128 user-generated tags. We use these two ways: 19 hand-picked interpretable tags (mltag_boring, mltag_predictable, mltag_masterpiece, etc.) plus PCA on the full 1,128-tag matrix (30 components, 26 survived feature selection). The PCA components capture quality patterns across tags that no single tag captures alone -- mltag_pc_0 had MI=0.177, as strong as mltag_boring.

**TF-IDF fix:** features.py previously computed a PySpark sparse vector that prepare() in run.py never extracted into the feature matrix. Fixed to extract the top 50 most informative dimensions as individual numeric columns. Most fail feature selection (title text is weak signal on 2-4 word movie titles) but the extraction is now correct.

**Why ExtraTrees beats Random Forest here:** ET randomizes split thresholds rather than searching for the optimal split at each node. With 60+ features including many correlated PCA components and tag scores, this extra randomness reduces overfitting and increases tree diversity within the ensemble.

**Threshold tuning:** after model selection, we collect out-of-fold probabilities from the champion model and sweep thresholds from 0.30 to 0.70 to find the one maximizing OOF accuracy. ET's default threshold of 0.5 gave 0.8914 OOF, while 0.46 gave 0.8937 -- the model is slightly conservative by default. This +0.0023 local improvement translated to +0.002 on the server (91.1% -> 91.3%).

**Data-driven tag selection (commented out in enrich.py):** we also implemented correlation-based selection across all 1,128 tags using point-biserial correlation against training labels, top 50 by absolute r. Produced the same 88.59% server score as the hand-picked 19 but is more principled -- kept commented for reference.

**Failed enrichments (what we tried and why it did not work):**

- Oscar awards, TSPDT rank, Criterion Collection: strong signals in theory but title+year fuzzy join coverage was too low (2-11% of training movies). Features failed both Mann-Whitney and permutation MI tests. Sparse features hurt more than they help on an 8k movie dataset.
- Director-DP loyalty (TMDB credits): signal exists in principle but not enough training examples with prior collaborations to pass the statistical tests.
- TMDB metadata (budget, revenue, popularity, original_language): budget missing for ~31% of movies. All four features failed permutation MI.
- Stacking (meta-learner): logistic regression on out-of-fold probabilities from all base models collapsed to ET/RF-dominant weights, models not diverse enough for stacking to add value.
- Bechdel test (bechdeltest.com): API returned 410 Gone during the project period, zero matches.

These failures are documented in the repo (`awards.py`, `prestige_lists.py`, `loyalty.py`) as examples of features that the pipeline's statistical gatekeeping correctly rejected.

---

### Submitting Predictions

Each line in `output/validation.txt` and `output/test.txt` is either `True` or `False` for the corresponding row in the hidden CSV files.

[Submission server](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/)
