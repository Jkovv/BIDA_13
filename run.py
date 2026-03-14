import joblib, pandas as pd, numpy as np
import os
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def prepare(df, is_train=True):
    # Deserializing Spark SparseVectors to dense float64 arrays
    def vec_to_arr(v):
        if isinstance(v, dict):
            arr = np.zeros(v['size'])
            arr[v['indices']] = v['values']
            return arr.astype(np.float64)
        return v.toArray().astype(np.float64) if hasattr(v, 'toArray') else v

    tfidf_list = df["tfidf"].apply(vec_to_arr).tolist()
    X_tfidf = pd.DataFrame(np.vstack(tfidf_list))
    
    # Use quantitative features cleaned via Robust Centers
    cols = ["year", "runtime", "votes"]
    X_num = df[[c for c in cols if c in df.columns]].reset_index(drop=True).apply(pd.to_numeric, errors='coerce').fillna(0)
    
    X = pd.concat([X_num, X_tfidf], axis=1)
    X.columns = X.columns.astype(str)
    
    y = df["label"].values if is_train else None
    return X.astype(np.float64), y

def run_benchmark():
    # Phase 1: Competition and Leakage Analysis
    train_feat_path = "/app/processed/train_features.parquet"
    if not os.path.exists(train_feat_path):
        return
        
    train_feat = pd.read_parquet(train_feat_path)
    X_all, y_all = prepare(train_feat)
    
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
        tr_scores, ho_scores = [], []
        for tr_idx, val_idx in skf.split(X_all, y_all):
            clf.fit(X_all.iloc[tr_idx], y_all[tr_idx])
            tr_scores.append(accuracy_score(y_all[tr_idx], clf.predict(X_all.iloc[tr_idx])))
            ho_scores.append(accuracy_score(y_all[val_idx], clf.predict(X_all.iloc[val_idx])))
        
        avg_tr, avg_ho = np.mean(tr_scores), np.mean(ho_scores)
        print(f"Model: {name.upper()} | Train: {avg_tr:.4f} | Holdout: {avg_ho:.4f} | Gap: {avg_tr - avg_ho:.4f}")
        if avg_ho > best_acc:
            best_acc, best_name = avg_ho, name

    # Phase 2: Final Retrain on 100% Signal
    print(f"\nChampion: {best_name.upper()} | Retraining on 100% data...")
    champion = models[best_name]
    champion.fit(X_all, y_all)
    
    os.makedirs('/app/output', exist_ok=True)
    joblib.dump(champion, "/app/output/best_model.pkl")

    # Phase 3: Output Generation with Strict Alignment
    for split in ["validation", "test"]:
        csv_path = f"/app/imdb/{split}_hidden.csv"
        feat_path = f"/app/processed/{split}_hidden_features.parquet"
        if os.path.exists(csv_path) and os.path.exists(feat_path):
            order = pd.read_csv(csv_path)[['tconst']]
            features = pd.read_parquet(feat_path)
            # Alignment: Join features back to the original CSV order
            merged = order.merge(features, on='tconst', how='left')
            X_eval, _ = prepare(merged, is_train=False)
            preds = champion.predict(X_eval)
            
            with open(f"/app/output/{split}.txt", "w") as f:
                for p in preds: f.write(f"{bool(p)}\n")
            print(f"Saved {split}.txt with {len(preds)} predictions.")

if __name__ == "__main__":
    run_benchmark()