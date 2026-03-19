# IMDB Movie Rating Classifier - Group 13
[IMDB Leaderboard](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/competitions/imdb)

---

Binary classification pipeline to predict whether a movie is "high-rated" on IMDB. We combine raw IMDB metadata with external features from MovieLens 25M, run everything through a nonparametric statistical gatekeeping pipeline, and let an ExtraTrees model do the rest.

**Final Local CV: 89.84%** | **Best Server Accuracy: 91.31%**

---

### 1. Pipeline Architecture

```
cleaning.py     DuckDB    CSV ingestion, dedup, unicode fix, winsorization, imputation
prestige.py     Pandas    Bayesian prestige scores for directors/writers (k=20)
enrich.py       Pandas    MovieLens 25M: ratings, genres, 19 tag scores + 30 PCA components
features.py     PySpark   TF-IDF on movie titles, extracts top 50 dimensions
run.py          sklearn   Feature selection → Optuna ET tuning → threshold tuning → predictions
```

**Why these tools?** DuckDB handles the CSV glob ingestion, deduplication, and window-function imputation much faster than pandas on 24k raw rows. PySpark is used for the TF-IDF pipeline because it's a natural fit for the tokenize→stopwords→TF→IDF chain and the project requires Spark usage. Pandas handles the MovieLens joins and prestige computation where row-level logic (like Bayesian smoothing) is more readable than SQL.

Not in the pipeline (kept for analysis and the poster):
```
awards.py          Oscar nominations/wins — matched ~11%, too sparse
prestige_lists.py  TSPDT rank + Criterion Collection — matched 2-3%, same
loyalty.py         Director-DP collaboration count from TMDB — matched ~10%, failed MI
```

---

### 2. Data Cleaning (cleaning.py)

The raw training data has every movie duplicated 3x across the CSV files — we handle this with `SELECT DISTINCT` in DuckDB. Other issues we found and fixed:

* **Unicode corruption:** titles like "Amélie" stored as "AmÃ©lie". Fixed with `ftfy.fix_text()` in pandas (can't do this in SQL).
* **Missing years (~10%):** imputed via `COALESCE(startYear, endYear)`, remaining NULLs left as-is.
* **Extreme runtimes:** winsorized to [1, 210] minutes using `LEAST(GREATEST(...))` in DuckDB. A handful of movies had runtimes over 5000 min.
* **Missing votes (~10%):** imputed with decade-median using `MEDIAN() OVER (PARTITION BY decade)` window function.
* **Derived features** computed in DuckDB: `log_votes`, `film_age` (2026 − year), `votes_per_minute`, `runtime_short` (< 90 min flag), `votes_x_runtime` (interaction), `is_foreign` (originalTitle ≠ primaryTitle).

---

### 3. External Data Enrichment (enrich.py)

We enrich every movie with features from **MovieLens 25M** (GroupLens, University of Minnesota) — this is NOT IMDB data, it's a completely separate user base with independent ratings.

**What we extract:**
* **Rating aggregates:** mean, std, median, count, log_count, tag_count - captures both quality and popularity from a different audience than IMDB.
* **19 genre flags** from MovieLens genre tags (Drama, Comedy, Horror, etc.) plus a genre_count feature.
* **19 hand-picked tag relevance scores:** MovieLens assigns continuous [0,1] relevance to ~1,128 user-generated tags per movie. We pick tags relevant to quality perception: `boring`, `predictable`, `masterpiece`, `atmospheric`, `cinematography`, `thought-provoking`, etc. These turned out to be the single biggest accuracy boost (+9%).
* **30 PCA components** on the full 1,128-tag matrix: captures cross-tag patterns that no individual tag captures alone. `mltag_pc_0` had MI=0.177, as strong as `mltag_boring`.

We also attempt to download **Bechdel Test** scores from bechdeltest.com (female representation, score 0-3), but the API has been returning 410 Gone throughout the project period.

---

### 4. Feature Selection (run.py)

With ~70 candidate features on only 8k movies, aggressive selection is essential. We use a 4-stage nonparametric pipeline — no normality assumptions anywhere:

1. **Mann-Whitney U + Benjamini-Hochberg** (FDR α=0.05) - tests whether feature distributions differ between True/False classes. BH correction prevents false discoveries across 70+ simultaneous tests.

2. **Permutation Mutual Information + BH** (FDR α=0.10) - computes real MI, then shuffles labels 100 times to build a null distribution. A feature only passes if its MI significantly exceeds what you'd get by chance. This is the strictest gate - most genres and all TF-IDF dimensions fail here.

3. **Partial Spearman** - for features we know are derived from others (like `votes_x_runtime` from `log_votes` × `runtime`), tests whether the derived feature adds conditional signal beyond its components. This caught `votes_per_minute` as fully redundant.

4. **Spearman Redundancy Pruning** (|ρ| > 0.85) — pairwise rank correlation between surviving features. If two are too correlated, drop the one with lower MI. This is why `director_prestige` gets dropped (ρ=0.851 with `writer_prestige`).

Typically ~60 features enter and ~35 survive. The exact set varies slightly between runs due to MI permutation randomness.

---

### 5. Prestige Scores (prestige.py / run.py)

Directors and writers get a Bayesian-smoothed label rate:

$$\text{prestige} = \frac{n}{n + k} \cdot \bar{x}_{\text{person}} + \frac{k}{n + k} \cdot \bar{x}_{\text{global}}$$

With $k=20$, a director needs 20+ films before their personal mean dominates over the global mean. Prevents a one-hit-wonder from getting prestige=1.0.

**Leak prevention:** `prestige.py` computes scores once for the final model, but `run.py` recomputes them inside each CV fold using only the training fold's labels. Validation labels never touch the prestige calculation during evaluation.

---

### 6. Model Selection & Tuning (run.py)

We tested six models across many runs: XGBoost, LightGBM, CatBoost, Random Forest, GradientBoosting, and ExtraTrees.

**ExtraTrees won every single run**, beating RF by ~2% consistently. ET randomizes split thresholds rather than optimizing them, which provides better regularization when dealing with 60+ features including many correlated PCA components.

**Optuna tuning** (80 TPE trials) searches over: `n_estimators` [300–1500], `max_depth` [10–24 or None], `min_samples_leaf` [1–5], `max_features` [sqrt, log2, 0.3–0.7]. Most reasonable configs give ~89% CV, but Optuna finds marginally better combos than random search.

**Threshold tuning:** ET's default 0.5 probability threshold is slightly conservative. We sweep out-of-fold probabilities from 0.30 to 0.70 and usually land around 0.44–0.48. This gave +0.2% on the server (91.1% → 91.3%).

---

### 7. Score Progression

| Local CV | Server | What changed |
|:---|:---|:---|
| 73.2% | — | Baseline: runtime + votes only |
| 79.2% | — | Added Bayesian prestige + MovieLens genres |
| 88.6% | 88.59% | MovieLens tag genome — boring and predictable were the strongest signals |
| 88.9% | 88.90% | PCA on all 1,128 tags — 26/30 components survived selection |
| 89.1% | 91.10% | Switched from Random Forest to ExtraTrees |
| 89.37% | 91.31% | Threshold tuning on OOF probabilities (0.46 instead of 0.5) |
| **89.84%** | **91.31%** | Optuna-tuned ET (80 trials), 60 features |

Local CV is consistently ~2% below the server score because the final model trains on 100% of the data.

---

### 8. What We Tried and Why It Failed

We ran a lot of experiments. Here's everything that didn't make the final pipeline and why the statistical tests rejected it:

**Oscar Awards** (`awards.py`): from Kaggle (unanimad/the-oscar-award). Only 11% of training movies had any Oscar history. With 89% zeros, both `oscar_nominations` and `oscar_wins` were indistinguishable from noise in the permutation MI test.

**TSPDT 1000 Greatest Films** (`prestige_lists.py`): scraped from theyshootpictures.com. Title+year fuzzy join matched only 2.6% of training movies. Same sparsity problem.

**Criterion Collection** (`prestige_lists.py`): from Kaggle (shankhadeepmaiti). 3.4% match rate. Too sparse.

**TMDB Budget/Revenue**: from Kaggle (rounakbanik/the-movies-dataset). Budget missing for ~31% of movies. Even with the filled values, all four features (budget, revenue, popularity, original_language) failed permutation MI. We also tried inflation-adjusting by decade — the missing data problem is the bottleneck, not the scale.

**Director-DP Loyalty** (`loyalty.py`): built from TMDB credits. Counted prior collaborations between a director and their cinematographer. The Scorsese–Ballhaus type signal is real, but only 10% of movies had prior collaborations. Failed MI.

**PageRank on Director-Writer Graph**: bipartite graph from directing.json + writing.json, computed PageRank centrality per person. The resulting features correlated ρ > 0.85 with Bayesian prestige — both just capture "this person is associated with good movies." Pruned in the redundancy step.

**NMF instead of PCA** (60 components): Non-Negative Matrix Factorization on the tag genome. Theoretically better for non-negative [0,1] data, but performed worse on the server. PCA's orthogonal components work better with ExtraTrees.

**More PCA components** (50, 80): diminishing returns past 30. Later components explain less variance and mostly add noise.

**Stacking / Meta-learner**: logistic regression on out-of-fold probabilities from all base models. Collapsed to giving ET ~95% of the weight. Not enough model diversity for stacking to help.

**Bechdel Test** (bechdeltest.com): API endpoint returned HTTP 410 Gone throughout the project period. Zero matches.

**TF-IDF title features**: 49/50 dimensions failed permutation MI across multiple runs. Movie titles are 2-4 words - there just isn't enough text for meaningful signal. The extraction is correct now (it was broken earlier, never reaching the model), but the features themselves are noise.

**Correlation-based tag selection** (commented out in `enrich.py`): instead of hand-picking 19 tags, we took the top 50 by point-biserial correlation with training labels. Same 88.59% server score — kept as a reference implementation.

All failed experiments are documented in the repo with comments explaining why the statistical pipeline rejected them.

---

### 9. Design Decisions

**Why DuckDB for cleaning?** The raw data is 24k rows across multiple CSVs with duplicates, missing values, and type issues. DuckDB's `read_csv_auto` with glob patterns, `SELECT DISTINCT`, `TRY_CAST`, `NULLIF`, and window functions handle all of this in a few SQL statements. The only thing we need pandas for is `ftfy.fix_text()`.

**Why PySpark for TF-IDF?** The tokenize→stopwords→HashingTF→IDF pipeline maps naturally to Spark's ML API. We fit IDF on training data only and apply to all splits. The results turned out to be nearly useless (titles are too short), but the implementation is sound.

**Why not Spark for everything?** On 8k movies, Spark's overhead isn't justified. DuckDB is faster for SQL-style work, pandas is more readable for join-heavy enrichment.

**Why hand-pick tags AND use PCA?** The 19 hand-picked tags pass all 4 statistical tests independently. They're not cherry-picked guesses — they're validated by the same pipeline. PCA components and individual tags aren't redundant (no tag correlates > 0.85 with any single component), so the model gets both direct signal (`mltag_boring = 0.92`) and decomposed patterns (`mltag_pc_0`). Removing the hand-picked tags drops server accuracy by ~0.3%.

---

### 10. How to Run

**Data setup:** place in `imdb/` before running:
```
train-*.csv, validation_hidden.csv, test_hidden.csv
directing.json, writing.json
```

MovieLens 25M (~262MB) downloads automatically on first run and gets cached.

```bash
docker compose up
```

Or manually:
```bash
docker build -t imdb-project .
docker run -it -v "$(pwd)/imdb:/app/imdb" -v "$(pwd)/output:/app/output" imdb-project
```

Output: `output/validation.txt` and `output/test.txt` — one `True`/`False` per line matching the hidden CSV row order.
