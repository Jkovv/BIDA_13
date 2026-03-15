import joblib, pandas as pd, numpy as np
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

ERA_ORDER = ['silent', 'golden_age', 'classic', 'new_hollywood', 'modern', 'contemporary']

def prepare(df, is_train=True):
    # core numeric features
    num_cols = ["year", "runtime", "log_votes", "vote_density",
                "director_prestige", "writer_prestige"]
    X_num = df[[c for c in num_cols if c in df.columns]].reset_index(drop=True)
    X_num = X_num.apply(pd.to_numeric, errors='coerce').fillna(0)

    # era as ordinal integer
    if "era" in df.columns:
        era_encoded = df["era"].map({e: i for i, e in enumerate(ERA_ORDER)}).fillna(0).astype(int)
        X_num["era_ord"] = era_encoded.values

    # genre one-hot columns
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    X_genre = df[genre_cols].reset_index(drop=True).fillna(0).astype(int) if genre_cols else pd.DataFrame()

    # no TF-IDF: title features caused leakage via IDF fitted on full training set
    X = pd.concat([X_num, X_genre], axis=1)
    X.columns = X.columns.astype(str)

    if is_train:
        le = LabelEncoder()
        y = le.fit_transform(df["label"].astype(str))
    else:
        y = None

    return X.astype(np.float64), y

def run_benchmark():
    train_feat_path = "/app/processed/train_features.parquet"
    if not os.path.exists(train_feat_path):
        return

    train_feat = pd.read_parquet(train_feat_path)
    X_all, y_all = prepare(train_feat)

    print(f"Feature matrix: {X_all.shape[0]} rows x {X_all.shape[1]} cols")
    print(f"Features: {list(X_all.columns)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'xgb': XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03, random_state=42),
        'lgbm': LGBMClassifier(n_estimators=500, max_depth=4, random_state=42, verbose=-1),
        'cat': CatBoostClassifier(iterations=500, depth=4, verbose=False, random_seed=42),
        'rf': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
        'ada': AdaBoostClassifier(n_estimators=100, random_state=42)
    }

    best_name, best_acc = None, 0
    print("--- Phase 1: Model Competition (Local CV) ---")
    for name, clf in models.items():
        tr_scores, ho_scores, f1_scores, auc_scores = [], [], [], []
        for tr_idx, val_idx in skf.split(X_all, y_all):
            clf.fit(X_all.iloc[tr_idx], y_all[tr_idx])
            val_preds = clf.predict(X_all.iloc[val_idx])
            tr_scores.append(accuracy_score(y_all[tr_idx], clf.predict(X_all.iloc[tr_idx])))
            ho_scores.append(accuracy_score(y_all[val_idx], val_preds))
            f1_scores.append(f1_score(y_all[val_idx], val_preds))
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_all.iloc[val_idx])[:, 1]
                auc_scores.append(roc_auc_score(y_all[val_idx], proba))

        avg_tr = np.mean(tr_scores)
        avg_ho = np.mean(ho_scores)
        avg_f1 = np.mean(f1_scores)
        avg_auc = np.mean(auc_scores) if auc_scores else 0
        print(f"Model: {name.upper()} | Acc: {avg_ho:.4f} | F1: {avg_f1:.4f} | AUC: {avg_auc:.4f} | Gap: {avg_tr - avg_ho:.4f}")
        if avg_ho > best_acc:
            best_acc, best_name = avg_ho, name

    print(f"\nChampion: {best_name.upper()} | Retraining on 100% data...")
    champion = models[best_name]
    champion.fit(X_all, y_all)

    os.makedirs('/app/output', exist_ok=True)
    joblib.dump(champion, "/app/output/best_model.pkl")

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
                for p in preds: f.write(f"{bool(p)}\n")
            print(f"Saved {split}.txt with {len(preds)} predictions.")

if __name__ == "__main__":
    run_benchmark()
