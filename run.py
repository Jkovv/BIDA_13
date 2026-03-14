import os, joblib, numpy as np, pandas as pd, optuna
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession

DATA_DIR, PROCESSED_DIR, OUTPUT_DIR = "/app/imdb", "/app/processed", "/app/output"
SEED = 42 

def prepare(df, smoothing_m, is_train=True):
    tfidf = np.array(df["tfidf"].apply(lambda x: x.toArray()).tolist())
    X_tfidf = pd.DataFrame(tfidf, columns=[f"f{i}" for i in range(tfidf.shape[1])])
    cols = ["year", "log_votes", "runtime_filled", "is_foreign", "title_len", "vote_density"]
    for pfx in ["dir", "wrt"]:
        col = f"{pfx}_score_m{smoothing_m}"
        if col in df.columns: cols.append(col)
    X_num = df[cols].copy()
    for col in X_num.columns:
        if X_num[col].dtype in ['bool', 'object']: X_num[col] = X_num[col].fillna(0).astype(int)
        else: X_num[col] = pd.to_numeric(X_num[col], errors='coerce').fillna(0)
    X = pd.concat([X_num.reset_index(drop=True), X_tfidf], axis=1)
    y = df["label"].astype(int).values if is_train else None
    return X, y

def run_benchmark():
    spark = SparkSession.builder.appName("IMDB-Final-Push").getOrCreate()
    train_pd = spark.read.parquet(f"{PROCESSED_DIR}/train_final.parquet").toPandas()
    
    print("\n--- Phase 1: Local Gap Analysis & Tuning (Champion Selection) ---")
    best_m, best_val, best_params = 0, 0, {}
    for m in [0, 5, 15, 30]:
        X_all, y_all = prepare(train_pd, m)
        X_tr, X_ho, y_tr, y_ho = train_test_split(X_all, y_all, test_size=0.15, random_state=SEED)
        study = optuna.create_study(direction="maximize")
        def obj(t):
            model = XGBClassifier(n_estimators=t.suggest_int("n_estimators", 400, 800), max_depth=t.suggest_int("max_depth", 3, 5), random_state=SEED)
            model.fit(X_tr, y_tr)
            return accuracy_score(y_ho, model.predict(X_ho))
        study.optimize(obj, n_trials=5)
        if study.best_value > best_val:
            best_m, best_val, best_params = m, study.best_value, study.best_params

    X_fin, y_fin = prepare(train_pd, best_m)
    X_t, X_v, y_t, y_v = train_test_split(X_fin, y_fin, test_size=0.15, random_state=SEED)
    
    # Stacking Ensemble for max learning performance
    clfs = [('xgb', XGBClassifier(**best_params, random_state=SEED)), ('rf', RandomForestClassifier(n_estimators=200, random_state=SEED))]
    stack = StackingClassifier(estimators=clfs, final_estimator=XGBClassifier(random_state=SEED))
    stack.fit(X_t, y_t)
    
    print(f"Local Train Acc: {accuracy_score(y_t, stack.predict(X_t)):.4f}")
    print(f"Local Val Acc: {accuracy_score(y_v, stack.predict(X_v)):.4f} (Smoothing M={best_m})")

    print("\n--- Phase 2: Final Retraining on 100% of Train Data ---")
    stack.fit(X_fin, y_fin)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(stack, f"{OUTPUT_DIR}/best_model.pkl")

    for split in ["validation", "test"]:
        csv_path = f"{DATA_DIR}/{split}_hidden.csv"
        if os.path.exists(csv_path):
            order = pd.read_csv(csv_path)[['tconst']]
            feat_pd = spark.read.parquet(f"{PROCESSED_DIR}/{split}_final.parquet").toPandas()
            merged = order.merge(feat_pd, on='tconst', how='left')
            X_ev, _ = prepare(merged, best_m, is_train=False)
            preds = stack.predict(X_ev)
            with open(f"{OUTPUT_DIR}/{split}.txt", "w") as f:
                for p in preds: f.write(f"{bool(p)}\n")
    spark.stop()

if __name__ == "__main__":
    run_benchmark()