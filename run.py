"""
run.py — Leak-free model training pipeline.

Key design: prestige scores are recomputed INSIDE each CV fold so that
validation fold labels never leak into prestige features. For final
predictions, prestige is computed on the full training set.
"""
import joblib
import json
import pandas as pd
import numpy as np
import os
import random
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

ERA_ORDER = ['silent', 'golden_age', 'classic', 'new_hollywood', 'modern', 'contemporary']

# ---- Prestige helpers (same logic as prestige.py, but callable per-fold) ----

def load_directing(path="/app/imdb/directing.json"):
    with open(path) as f:
        d = json.load(f)
    return pd.DataFrame({"tconst": list(d["movie"].values()),
                          "director_id": list(d["director"].values())})

def load_writing(path="/app/imdb/writing.json"):
    with open(path) as f:
        w = json.load(f)
    return pd.DataFrame({"tconst": [x["movie"] for x in w],
                          "writer_id": [x["writer"] for x in w]})

# Load once globally (these are static metadata, no labels)
DIRECTING = None
WRITING = None
DIR_COUNT = None
WRI_COUNT = None

def init_crew_data():
    global DIRECTING, WRITING, DIR_COUNT, WRI_COUNT
    if DIRECTING is None:
        DIRECTING = load_directing()
        WRITING = load_writing()
        DIR_COUNT = DIRECTING.groupby("tconst").size().reset_index(name="n_directors")
        WRI_COUNT = WRITING.groupby("tconst").size().reset_index(name="n_writers")

def bayesian_prestige(df, id_col, label_col, k=20):
    global_mean = df[label_col].mean()
    stats = df.groupby(id_col)[label_col].agg(["mean", "count"]).reset_index()
    stats.columns = [id_col, "person_mean", "n"]
    stats["prestige"] = (stats["n"] / (stats["n"] + k)) * stats["person_mean"] + \
                        (k / (stats["n"] + k)) * global_mean
    return stats[[id_col, "prestige"]], global_mean

def compute_prestige_features(train_df, target_df):
    """Compute prestige from train_df labels, apply to target_df rows.
    train_df must have 'tconst' and 'label_int' columns.
    target_df must have 'tconst' column.
    Returns target_df with prestige columns added.
    """
    init_crew_data()

    train_dir = train_df.merge(DIRECTING, on="tconst", how="left")
    train_wri = train_df.merge(WRITING, on="tconst", how="left")

    dir_prestige, dir_global = bayesian_prestige(
        train_dir.dropna(subset=["director_id"]), "director_id", "label_int", k=20)
    wri_prestige, wri_global = bayesian_prestige(
        train_wri.dropna(subset=["writer_id"]), "writer_id", "label_int", k=20)

    # Director experience based on train_df only
    dir_exp = DIRECTING.merge(train_df[["tconst"]], on="tconst", how="inner")
    dir_exp = dir_exp.groupby("director_id").size().reset_index(name="dir_experience")
    wri_exp = WRITING.merge(train_df[["tconst"]], on="tconst", how="inner")
    wri_exp = wri_exp.groupby("writer_id").size().reset_index(name="wri_experience")

    # Build lookup tables
    dir_scores = DIRECTING.merge(dir_prestige, on="director_id", how="left")
    dir_scores["prestige"] = dir_scores["prestige"].fillna(dir_global)
    dir_mean = dir_scores.groupby("tconst")["prestige"].mean().reset_index()
    dir_mean.columns = ["tconst", "director_prestige"]

    wri_scores = WRITING.merge(wri_prestige, on="writer_id", how="left")
    wri_scores["prestige"] = wri_scores["prestige"].fillna(wri_global)
    wri_mean = wri_scores.groupby("tconst")["prestige"].mean().reset_index()
    wri_mean.columns = ["tconst", "writer_prestige"]

    dir_exp_scores = DIRECTING.merge(dir_exp, on="director_id", how="left")
    dir_exp_scores["dir_experience"] = dir_exp_scores["dir_experience"].fillna(0)
    dir_exp_max = dir_exp_scores.groupby("tconst")["dir_experience"].max().reset_index()

    wri_exp_scores = WRITING.merge(wri_exp, on="writer_id", how="left")
    wri_exp_scores["wri_experience"] = wri_exp_scores["wri_experience"].fillna(0)
    wri_exp_max = wri_exp_scores.groupby("tconst")["wri_experience"].max().reset_index()

    # Drop existing prestige columns from target to avoid _x/_y dupes
    prestige_cols = ["director_prestige", "writer_prestige", "n_directors",
                     "n_writers", "dir_experience", "wri_experience"]
    out = target_df.drop(columns=[c for c in prestige_cols if c in target_df.columns], errors="ignore")

    out = out.merge(dir_mean, on="tconst", how="left")
    out = out.merge(wri_mean, on="tconst", how="left")
    out = out.merge(DIR_COUNT, on="tconst", how="left")
    out = out.merge(WRI_COUNT, on="tconst", how="left")
    out = out.merge(dir_exp_max, on="tconst", how="left")
    out = out.merge(wri_exp_max, on="tconst", how="left")

    out["director_prestige"] = out["director_prestige"].fillna(dir_global)
    out["writer_prestige"] = out["writer_prestige"].fillna(wri_global)
    out["n_directors"] = out["n_directors"].fillna(1).astype(int)
    out["n_writers"] = out["n_writers"].fillna(1).astype(int)
    out["dir_experience"] = out["dir_experience"].fillna(0).astype(int)
    out["wri_experience"] = out["wri_experience"].fillna(0).astype(int)

    return out

# ---- Feature preparation ----

PRESTIGE_COLS = ["director_prestige", "writer_prestige",
                 "n_directors", "n_writers", "dir_experience", "wri_experience"]

def prepare(df):
    """Build feature matrix X from dataframe. Returns (X, y_or_None)."""
    num_cols = [
        "year", "runtime", "log_votes", "vote_density", "votes_per_minute",
        "is_foreign", "title_length", "title_word_count",
        "is_sequel", "votes_missing", "title_corrupted",
        # prestige
        "director_prestige", "writer_prestige",
        "n_directors", "n_writers", "dir_experience", "wri_experience",
        # enrich
        "is_movie", "is_short", "is_tvmovie",
    ]

    X_num = df[[c for c in num_cols if c in df.columns]].reset_index(drop=True)
    X_num = X_num.apply(pd.to_numeric, errors='coerce').fillna(0)

    if "era" in df.columns:
        X_num["era_ord"] = df["era"].map(
            {e: i for i, e in enumerate(ERA_ORDER)}
        ).fillna(0).astype(int).values

    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    X_genre = df[genre_cols].reset_index(drop=True).fillna(0).astype(int) if genre_cols else pd.DataFrame()

    X = pd.concat([X_num, X_genre], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    X.columns = X.columns.astype(str)

    has_label = "label" in df.columns
    if has_label:
        y = LabelEncoder().fit_transform(df["label"].astype(str))
    else:
        y = None

    return X.astype(np.float64), y

# ---- Leak-free cross-validation ----

def cv_score_leakfree(clf_factory, full_df, skf):
    """
    For each fold:
      1. Split full_df into train_fold / val_fold by index
      2. Compute prestige from train_fold labels only
      3. Apply prestige to both folds
      4. Build feature matrices
      5. Train & score
    """
    # Encode labels once for splitting
    y_all = LabelEncoder().fit_transform(full_df["label"].astype(str))
    indices = np.arange(len(full_df))
    scores = []

    for fold_i, (tr_idx, val_idx) in enumerate(skf.split(indices, y_all)):
        train_fold_df = full_df.iloc[tr_idx].copy().reset_index(drop=True)
        val_fold_df = full_df.iloc[val_idx].copy().reset_index(drop=True)

        # Compute label_int on train fold only
        train_fold_df["label_int"] = ((train_fold_df["label"] == "True") |
                                       (train_fold_df["label"] == True)).astype(int)

        # Compute prestige from train fold, apply to both
        train_fold_df = compute_prestige_features(train_fold_df, train_fold_df)
        val_fold_df = compute_prestige_features(train_fold_df, val_fold_df)

        # Drop helper columns
        if "label_int" in train_fold_df.columns:
            train_fold_df = train_fold_df.drop(columns=["label_int"])

        X_tr, y_tr = prepare(train_fold_df)
        X_val, y_val = prepare(val_fold_df)

        clf = clf_factory()
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        scores.append(acc)

    return np.mean(scores), np.std(scores)


def run_benchmark():
    print("Step 6: Model training & prediction (leak-free CV)")

    train_feat_path = "/app/processed/train_features.parquet"
    if not os.path.exists(train_feat_path):
        print("  ERROR: train_features.parquet not found.")
        return

    full_df = pd.read_parquet(train_feat_path)

    # Quick feature count check (prestige columns may or may not be present yet)
    X_tmp, y_tmp = prepare(full_df)
    print(f"  Base features (before per-fold prestige): {X_tmp.shape[1]} cols")
    print(f"  Training rows: {len(full_df)}")
    print(f"  Label balance: {y_tmp.mean():.3f} positive rate")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---- Phase 1: Model competition (leak-free) ----
    model_factories = {
        'xgb':  lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       random_state=42, eval_metric='logloss'),
        'lgbm': lambda: LGBMClassifier(n_estimators=300, max_depth=4, random_state=42, verbose=-1),
        'cat':  lambda: CatBoostClassifier(iterations=300, depth=4, verbose=False, random_seed=42),
        'rf':   lambda: RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'ada':  lambda: AdaBoostClassifier(n_estimators=100, random_state=42),
        'gbm':  lambda: GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                                    random_state=42),
    }

    best_name, best_acc = None, 0
    all_scores = {}
    print("\n  --- Phase 1: Model Competition (leak-free CV) ---")
    for name, factory in model_factories.items():
        mean_acc, std_acc = cv_score_leakfree(factory, full_df, skf)
        all_scores[name] = mean_acc
        print(f"    {name.upper():5s}: {mean_acc:.4f} (+/- {std_acc:.4f})")
        if mean_acc > best_acc:
            best_acc, best_name = mean_acc, name
    print(f"  Champion: {best_name.upper()} at {best_acc:.4f}")

    # ---- Phase 2: Tune champion (leak-free) ----
    print(f"\n  --- Phase 2: Tuning {best_name.upper()} ---")
    param_grids = {
        'xgb': [{'n_estimators': n, 'max_depth': d, 'learning_rate': lr,
                  'subsample': ss, 'colsample_bytree': cs, 'eval_metric': 'logloss'}
                for n in [300, 500, 800] for d in [3, 4, 5, 6]
                for lr in [0.02, 0.05, 0.1] for ss in [0.8, 1.0] for cs in [0.8, 1.0]],
        'lgbm': [{'n_estimators': n, 'max_depth': d, 'learning_rate': lr,
                   'num_leaves': nl, 'subsample': ss}
                 for n in [300, 500, 800] for d in [4, 6, 8]
                 for lr in [0.02, 0.05, 0.1] for nl in [31, 63] for ss in [0.8, 1.0]],
        'cat': [{'iterations': n, 'depth': d, 'learning_rate': lr, 'l2_leaf_reg': l2}
                for n in [300, 500, 800] for d in [4, 6, 8]
                for lr in [0.02, 0.05, 0.1] for l2 in [1, 3, 5]],
        'rf':  [{'n_estimators': n, 'max_depth': d, 'min_samples_leaf': ml}
                for n in [200, 400, 600] for d in [8, 12, 16, None] for ml in [1, 3, 5]],
        'ada': [{'n_estimators': n, 'learning_rate': lr}
                for n in [100, 200, 300] for lr in [0.3, 0.5, 1.0]],
        'gbm': [{'n_estimators': n, 'max_depth': d, 'learning_rate': lr, 'subsample': ss}
                for n in [300, 500, 800] for d in [3, 4, 5]
                for lr in [0.02, 0.05, 0.1] for ss in [0.8, 1.0]],
    }

    def make_constructor(name, params):
        constructors = {
            'xgb':  lambda: XGBClassifier(random_state=42, **params),
            'lgbm': lambda: LGBMClassifier(random_state=42, verbose=-1, **params),
            'cat':  lambda: CatBoostClassifier(verbose=False, random_seed=42, **params),
            'rf':   lambda: RandomForestClassifier(random_state=42, n_jobs=-1, **params),
            'ada':  lambda: AdaBoostClassifier(random_state=42, **params),
            'gbm':  lambda: GradientBoostingClassifier(random_state=42, **params),
        }
        return constructors[name]

    random.seed(42)
    grid = param_grids[best_name]
    sampled = random.sample(grid, min(12, len(grid)))
    best_tuned_acc, best_params = best_acc, None

    for i, params in enumerate(sampled):
        factory = make_constructor(best_name, params)
        mean_acc, std_acc = cv_score_leakfree(factory, full_df, skf)
        marker = " ***" if mean_acc > best_tuned_acc else ""
        print(f"    [{i+1}/{len(sampled)}] {mean_acc:.4f} +/- {std_acc:.4f}{marker}  {params}")
        if mean_acc > best_tuned_acc:
            best_tuned_acc, best_params = mean_acc, params

    print(f"  Best tuned: {best_params} -> {best_tuned_acc:.4f}")

    # ---- Phase 3: Ensemble top 3 (leak-free) ----
    print("\n  --- Phase 3: Soft-voting ensemble ---")
    sorted_models = sorted(all_scores.items(), key=lambda x: -x[1])
    top3_names = [name for name, _ in sorted_models[:3]]
    print(f"  Top 3: {[f'{n.upper()}({all_scores[n]:.4f})' for n in top3_names]}")

    def ensemble_factory():
        estimators = []
        if best_params:
            estimators.append((best_name, make_constructor(best_name, best_params)()))
        else:
            estimators.append((best_name, model_factories[best_name]()))
        for name in top3_names:
            if name != best_name:
                estimators.append((name, model_factories[name]()))
        return VotingClassifier(estimators=estimators, voting='soft')

    ens_acc, ens_std = cv_score_leakfree(ensemble_factory, full_df, skf)
    print(f"  Ensemble CV: {ens_acc:.4f} +/- {ens_std:.4f}")

    use_ensemble = ens_acc > best_tuned_acc
    if use_ensemble:
        print(f"  Using ENSEMBLE (beat single model by {ens_acc - best_tuned_acc:.4f})")
    else:
        print(f"  Using single {best_name.upper()} (ensemble didn't improve)")

    # ---- Phase 4: Final training on ALL data + predictions ----
    print("\n  --- Phase 4: Final training on full data + predictions ---")

    # Compute prestige on FULL training set for final model
    full_df["label_int"] = ((full_df["label"] == "True") | (full_df["label"] == True)).astype(int)
    full_with_prestige = compute_prestige_features(full_df, full_df)
    if "label_int" in full_with_prestige.columns:
        full_with_prestige = full_with_prestige.drop(columns=["label_int"])

    X_all, y_all = prepare(full_with_prestige)
    print(f"  Final feature matrix: {X_all.shape[0]} x {X_all.shape[1]}")

    if use_ensemble:
        champion = ensemble_factory()
    else:
        if best_params:
            champion = make_constructor(best_name, best_params)()
        else:
            champion = model_factories[best_name]()

    champion.fit(X_all, y_all)

    os.makedirs('/app/output', exist_ok=True)
    joblib.dump(champion, "/app/output/best_model.pkl")

    # Generate predictions for val/test
    for split in ["validation", "test"]:
        csv_path = f"/app/imdb/{split}_hidden.csv"
        feat_path = f"/app/processed/{split}_hidden_features.parquet"
        if os.path.exists(csv_path) and os.path.exists(feat_path):
            order = pd.read_csv(csv_path)[['tconst']]
            features = pd.read_parquet(feat_path)
            merged = order.merge(features, on='tconst', how='left')

            # Apply prestige computed from full training set
            merged = compute_prestige_features(full_df, merged)

            X_eval, _ = prepare(merged)
            preds = champion.predict(X_eval)
            with open(f"/app/output/{split}.txt", "w") as f:
                for p in preds:
                    f.write(f"{bool(p)}\n")
            n_true = sum(bool(p) for p in preds)
            print(f"  {split}.txt: {len(preds)} predictions ({n_true} True, {len(preds)-n_true} False)")
        else:
            if not os.path.exists(feat_path):
                print(f"  WARNING: {feat_path} not found, skipping {split}")

    print("\nPipeline complete. Local CV scores are now leak-free and trustworthy.")
    print("Submit output/validation.txt and output/test.txt to the server.")


if __name__ == "__main__":
    run_benchmark()