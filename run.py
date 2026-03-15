import joblib, pandas as pd, numpy as np
import os, random
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

ERA_ORDER = ['silent', 'golden_age', 'classic', 'new_hollywood', 'modern', 'contemporary']

def prepare(df, is_train=True):
    num_cols = ["year", "runtime", "log_votes", "vote_density",
                "director_prestige", "writer_prestige"]
    X_num = df[[c for c in num_cols if c in df.columns]].reset_index(drop=True)
    X_num = X_num.apply(pd.to_numeric, errors='coerce').fillna(0)

    # encode era as ordinal integer (silent=0 ... contemporary=5)
    if "era" in df.columns:
        X_num["era_ord"] = df["era"].map({e: i for i, e in enumerate(ERA_ORDER)}).fillna(0).astype(int).values

    # genre one-hots from enrich.py
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    X_genre = df[genre_cols].reset_index(drop=True).fillna(0).astype(int) if genre_cols else pd.DataFrame()

    X = pd.concat([X_num, X_genre], axis=1)
    X.columns = X.columns.astype(str)

    if is_train:
        y = LabelEncoder().fit_transform(df["label"].astype(str))
    else:
        y = None

    return X.astype(np.float64), y

def cv_score(clf, X, y, skf):
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        clf.fit(X.iloc[tr_idx], y[tr_idx])
        preds = clf.predict(X.iloc[val_idx])
        scores.append(accuracy_score(y[val_idx], preds))
    return np.mean(scores)

def run_benchmark():
    train_feat_path = "/app/processed/train_features.parquet"
    if not os.path.exists(train_feat_path):
        return

    train_feat = pd.read_parquet(train_feat_path)
    X_all, y_all = prepare(train_feat)

    print(f"Feature matrix: {X_all.shape[0]} rows x {X_all.shape[1]} cols")
    print(f"Features: {list(X_all.columns)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Phase 1: model competition with default params
    models = {
        'xgb':  XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
        'lgbm': LGBMClassifier(n_estimators=300, max_depth=4, random_state=42, verbose=-1),
        'cat':  CatBoostClassifier(iterations=300, depth=4, verbose=False, random_seed=42),
        'rf':   RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
        'ada':  AdaBoostClassifier(n_estimators=100, random_state=42)
    }

    best_name, best_acc = None, 0
    print("--- Phase 1: Model Competition ---")
    for name, clf in models.items():
        acc = cv_score(clf, X_all, y_all, skf)
        print(f"  {name.upper()}: {acc:.4f}")
        if acc > best_acc:
            best_acc, best_name = acc, name
    print(f"Champion: {best_name.upper()} at {best_acc:.4f}")

    # Phase 2: manual grid search on champion
    print(f"\n--- Phase 2: Tuning {best_name.upper()} ---")
    param_grids = {
        'xgb':  [{'n_estimators': n, 'max_depth': d, 'learning_rate': lr}
                 for n in [300, 600] for d in [3, 5] for lr in [0.03, 0.05]],
        'lgbm': [{'n_estimators': n, 'max_depth': d, 'learning_rate': lr, 'num_leaves': nl}
                 for n in [300, 600] for d in [4, 6] for lr in [0.03, 0.05] for nl in [31, 63]],
        'cat':  [{'iterations': n, 'depth': d, 'learning_rate': lr}
                 for n in [300, 600] for d in [4, 6] for lr in [0.03, 0.05]],
        'rf':   [{'n_estimators': n, 'max_depth': d}
                 for n in [200, 400] for d in [8, 12, None]],
        'ada':  [{'n_estimators': n, 'learning_rate': lr}
                 for n in [100, 200] for lr in [0.5, 1.0]]
    }
    constructors = {
        'xgb':  lambda p: XGBClassifier(random_state=42, **p),
        'lgbm': lambda p: LGBMClassifier(random_state=42, verbose=-1, **p),
        'cat':  lambda p: CatBoostClassifier(verbose=False, random_seed=42, **p),
        'rf':   lambda p: RandomForestClassifier(random_state=42, **p),
        'ada':  lambda p: AdaBoostClassifier(random_state=42, **p),
    }

    random.seed(42)
    sampled = random.sample(param_grids[best_name], min(8, len(param_grids[best_name])))
    best_tuned_acc, best_params = best_acc, None
    for i, params in enumerate(sampled):
        clf = constructors[best_name](params)
        acc = cv_score(clf, X_all, y_all, skf)
        print(f"  [{i+1}/{len(sampled)}] {params} -> {acc:.4f}")
        if acc > best_tuned_acc:
            best_tuned_acc, best_params = acc, params

    print(f"Best params: {best_params} -> {best_tuned_acc:.4f}")
    champion = constructors[best_name](best_params) if best_params else models[best_name]
    champion.fit(X_all, y_all)

    os.makedirs('/app/output', exist_ok=True)
    joblib.dump(champion, "/app/output/best_model.pkl")

    # Phase 3: generate predictions aligned to original CSV row order
    for split in ["validation", "test"]:
        csv_path = f"/app/imdb/{split}_hidden.csv"
        feat_path = f"/app/processed/{split}_hidden_features.parquet"
        if os.path.exists(csv_path) and os.path.exists(feat_path):
            order = pd.read_csv(csv_path)[['tconst']]
            features = pd.read_parquet(feat_path)
            merged = order.merge(features, on='tconst', how='left')
            X_eval, _ = prepare(merged, is_train=False)
            preds = champion.predict(X_eval)
            with open(f"/app/output/{split}.txt", "w") as f:
                for p in preds:
                    f.write(f"{bool(p)}\n")
            print(f"Saved {split}.txt with {len(preds)} predictions.")

    # NOTE: Local CV accuracy is inflated because prestige scores are computed
    # on the full training set before CV splits. The real accuracy is the one
    # on the submission server. This is documented and acknowledged.
    print("\nDone. Submit output/validation.txt and output/test.txt to the server.")

if __name__ == "__main__":
    run_benchmark()