# IMDB Movie Rating Classifier - Group 13
[IMDB Leaderboard](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/competitions/imdb)

--- 

This project implements a high-performance binary classification pipeline to predict whether a movie is "high-rated" on IMDB. By combining raw IMDB data with extensive external features from MovieLens 25M and the Bechdel Test, the system achieves a state-of-the-art balance between complexity and statistical rigor.

**Final Local CV: 89.84%** | **Best Server Accuracy: 91.31%**

---

### 1. Pipeline Architecture

The pipeline is distributed across specialized modules to ensure efficiency and scalability:

* **`cleaning.py` (DuckDB):** Handles heavy lifting for data ingestion. It performs unicode normalization (`ftfy`), winsorization of runtime and votes, and median imputation of missing values partitioned by decade.
* **`prestige.py` (Pandas):** Generates Bayesian-smoothed prestige scores for directors and writers ($k=20$). Note that `run.py` recomputes these within each CV fold to prevent label leakage.
* **`enrich.py` (Pandas):** Integrates MovieLens 25M ratings and tag genome data. It extracts 19 hand-picked semantic tags and 30 PCA components from the 1,128-tag genome matrix to capture latent quality signals.
* **`features.py` (PySpark):** Uses Spark to process movie titles through a TF-IDF pipeline, extracting the top 50 most informative dimensions as numeric features.
* **`run.py` (sklearn/Optuna):** The core execution engine. It runs a 4-stage statistical feature selection process, an 80-trial Optuna hyperparameter study, and probability threshold tuning.

---

### 2. Feature Selection: Statistical Gatekeeping

To prevent overfitting on an 8k movie dataset, all features must pass a rigorous nonparametric selection process in `run.py`:

1.  **Mann-Whitney U + Benjamini-Hochberg (FDR 0.05):** A nonparametric test for significant group differences between high and low-rated movies.
2.  **Permutation Mutual Information (FDR 0.10):** Compares real MI against a null distribution from 100 label shuffles to ensure signals are not random.
3.  **Partial Spearman Correlation:** Filters interaction terms (like `votes_x_runtime`) to ensure they provide conditional signal beyond their base components.
4.  **Spearman Redundancy Pruning:** Pairwise rank correlation checks ($|\rho| > 0.85$) remove collinear features, keeping the one with higher Mutual Information.

---

### 3. Experiment Log & Progression

| Local CV | Configuration | Server Score |
| :--- | :--- | :--- |
| 73.2% | Baseline with basic IMDB features | - |
| 79.2% | Bayesian Prestige + MovieLens Genres | - |
| 88.6% | MovieLens Tag Genome (19 hand-picked tags) | 88.59% |
| 89.1% | ExtraTrees (ET) + Fixed TF-IDF Extraction | 91.10% |
| 89.37% | ET + Threshold Tuning (0.46) | 91.31% |
| **89.84%** | **Final: Optuna Optimized ET (500 est, depth 16) + 60 Features** | 91.31% |

---

### 4. Final Model Configuration

The current "Champion" model is an **ExtraTreesClassifier**. Its use of random split thresholds provides superior regularization over Random Forest when handling high-dimensional PCA and tag features.

* **Best Parameters (Optuna Trial 75+):** `n_estimators: 500`, `max_depth: 16`, `min_samples_leaf: 3`, `max_features: 0.3`.
* **Optimal Threshold:** 0.50.
* **Selected Features (60):** Includes `log_votes`, `film_age`, `writer_prestige`, `ml_rating_mean`, 19 semantic tags (e.g., `mltag_atmospheric`), and 25+ PCA components (e.g., `mltag_pc_0`).

---

### 5. Failed Enrichments (Analysis)

The following were analyzed but correctly rejected by our statistical pipeline:
* **Oscar Awards & Prestige Lists (TSPDT/Criterion):** Strong signals but low coverage (2-11% match); failed permutation MI tests.
* **Director-DP Loyalty:** Signal existed for elite pairings, but insufficient training examples to generalize.
* **TMDB Metadata:** High percentage of missing budget data (~31%); failed MI gatekeeping.
* **Graph PageRank:** Highly redundant with existing Bayesian prestige scores ($\rho > 0.85$).

---

### 6. How to Run

1.  **Data Setup:** Place `train-*.csv`, `directing.json`, and `writing.json` in the `/app/imdb/` directory.
2.  **Environment:** Ensure Docker is installed.
3.  **Execution:**
    ```bash
    docker compose up
    ```
4.  **Outputs:** Predictions are saved to `/app/output/validation.txt` and `/app/output/test.txt`.

[Submission server](http://big-data-competitions.westeurope.cloudapp.azure.com:8080/)
