import argparse
import os
import numpy as np
import pandas as pd
import optuna

# Model Libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Frameworks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession

# Paths
PROCESSED_DIR = "/app/processed"
OUTPUT_DIR = "/app/output"

# Base Features
FEATURE_COLS = [
    "year", "decade", "movie_age", "numVotes_imputed", "log_numVotes", 
    "runtimeMinutes_winsorised", "votes_per_minute", "n_directors", 
    "n_writers", "crew_size", "is_multi_director", "is_foreign", 
    "is_title_corrupted", "title_length", "title_word_count",
    "director_avg_success", "director_max_experience",
    "writer_avg_success", "writer_max_experience",
]

def prepare_dataset(df, is_train=True):
    """
    Standardizes feature matrix and handles TF-IDF expansion.
    """
    if "title_tfidf" in df.columns:
        tfidf_data = np.array(df["title_tfidf"].apply(lambda x: x.toArray()).tolist())
        tfidf_df = pd.DataFrame(tfidf_data, columns=[f"tfidf_{i}" for i in range(tfidf_data.shape[1])])
        df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

    model_ready_cols = [c for c in df.columns if c in FEATURE_COLS or c.startswith("tfidf_")]
    
    for col in model_ready_cols:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            
    X = df[model_ready_cols].fillna(0).values
    y = df["label"].astype(int).values if is_train and "label" in df.columns else None
    
    return X, y

def get_tuning_objective(model_name, X, y):
    """
    Search space logic with extreme regularization for XGBoost and LGBM.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        if model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 2, 3), # Force extremely shallow trees
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 20, 60), # Huge leaves
                "reg_lambda": trial.suggest_float("reg_lambda", 100.0, 500.0), # Extreme L2 penalty
                "subsample": trial.suggest_float("subsample", 0.4, 0.6),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.5), # Feature dropout
                "random_state": 42, "eval_metric": "logloss", "n_jobs": -1
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
        
        elif model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "num_leaves": trial.suggest_int("num_leaves", 5, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 100.0, 500.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 150),
                "random_state": 42, "n_jobs": -1, "verbosity": -1
            }
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[xgb.callback.EarlyStopping(stopping_rounds=30)])

        elif model_name == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 400, 1000),
                "depth": trial.suggest_int("depth", 2, 4),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 10.0, 80.0),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "random_seed": 42, "verbose": False, "early_stopping_rounds": 40
            }
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        
        else: # Random Forest
            params = {
                "n_estimators": 300, "max_depth": 6, "min_samples_leaf": 80, 
                "random_state": 42, "n_jobs": -1
            }
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

        return accuracy_score(y_val, model.predict(X_val))

    return objective

def run_benchmark(tune=False):
    spark = SparkSession.builder.appName("IMDB-Final-Benchmark").getOrCreate()
    
    print("\n[INFO] Loading feature-engineered data...")
    train_df = spark.read.parquet(os.path.join(PROCESSED_DIR, "train_final.parquet")).toPandas()
    X, y = prepare_dataset(train_df, is_train=True)
    
    # Internal holdout check
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.15, random_state=42)

    factories = {
        "catboost": lambda p: CatBoostClassifier(**p, verbose=False),
        "xgboost": lambda p: xgb.XGBClassifier(**p, eval_metric="logloss"),
        "lightgbm": lambda p: LGBMClassifier(**p, verbosity=-1),
        "random_forest": lambda p: RandomForestClassifier(**p)
    }

    stats = []
    best_clf, best_acc = None, 0

    print(f"\n[INFO] Starting benchmarking (Tune={tune})...")
    for name in factories.keys():
        if tune:
            print(f" >> Tuning {name}...")
            study = optuna.create_study(direction="maximize")
            study.optimize(get_tuning_objective(name, X_train, y_train), n_trials=15)
            params = study.best_params
        else:
            params = {"random_state": 42, "max_depth": 3 if name != "random_forest" else 6}

        if name == "catboost" and "random_state" in params:
            params["random_seed"] = params.pop("random_state")

        clf = factories[name](params)
        clf.fit(X_train, y_train)

        tr_score = accuracy_score(y_train, clf.predict(X_train))
        ho_score = accuracy_score(y_holdout, clf.predict(X_holdout))
        print(f" >> {name.upper():15} | Train: {tr_score:.4f} | Holdout: {ho_score:.4f}")

        stats.append({"model": name, "train": tr_score, "holdout": ho_score})
        if ho_score > best_acc:
            best_acc, best_clf = ho_score, clf

    print("\n" + "="*40 + "\n FINAL RESULTS \n" + "="*40)
    print(pd.DataFrame(stats).sort_values("holdout", ascending=False).to_string(index=False))
    print("="*40)

    # Submission logic
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split, parquet_name in [("validation", "validation_final.parquet"), ("test", "test_final.parquet")]:
        path = os.path.join(PROCESSED_DIR, parquet_name)
        if os.path.exists(path):
            pdf = spark.read.parquet(path).toPandas()
            X_eval, _ = prepare_dataset(pdf, is_train=False)
            preds = best_clf.predict(X_eval)
            with open(os.path.join(OUTPUT_DIR, f"{split}.txt"), "w") as f:
                for p in preds: f.write(f"{bool(p)}\n")
    
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    run_benchmark(tune=parser.parse_args().tune)