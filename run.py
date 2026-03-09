import argparse
import os
import json
import numpy as np
import pandas as pd
import optuna

# Models
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Evaluation & Spark
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession

# Path config
PROCESSED_DIR = "/app/processed"
OUTPUT_DIR = "/app/output"

# Features derived from the cleaning and features modules
FEATURE_COLS = [
    "year", "decade", "movie_age", "numVotes_imputed", "log_numVotes", 
    "runtimeMinutes_winsorised", "votes_per_minute", "n_directors", 
    "n_writers", "crew_size", "is_multi_director", "is_foreign", 
    "is_title_corrupted", "title_length", "title_word_count",
    "director_avg_success", "director_max_experience",
    "writer_avg_success", "writer_max_experience",
]

def prepare_data(df, is_train=True):
    """
    Converts Spark-processed Parquet into a Pandas-friendly format.
    Handles TF-IDF expansion and casts booleans for the boosters.
    """
    # Expand TF-IDF vectors if they exist
    if "title_tfidf" in df.columns:
        tfidf_arrays = np.array(df["title_tfidf"].apply(lambda x: x.toArray()).tolist())
        tfidf_df = pd.DataFrame(
            tfidf_arrays, 
            columns=[f"tfidf_{i}" for i in range(tfidf_arrays.shape[1])]
        )
        df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

    # Filter columns to only what the model needs
    model_features = [c for c in df.columns if c in FEATURE_COLS or c.startswith("tfidf_")]
    
    # Cast booleans to int for compatibility
    for col in model_features:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            
    X = df[model_features].fillna(0).values
    y = df["label"].astype(int).values if is_train and "label" in df.columns else None
    
    return X, y

def get_tuning_objective(model_name, X, y):
    """
    Wraps the Optuna objective function. 
    Uses a 20% validation split to monitor overfitting during tuning.
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        if model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 2, 4),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 15, 50),
                "reg_lambda": trial.suggest_float("reg_lambda", 50.0, 200.0),
                "subsample": trial.suggest_float("subsample", 0.5, 0.7),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.6),
                "random_state": 42, "eval_metric": "logloss", "n_jobs": -1
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
        
        elif model_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 500),
                "max_depth": trial.suggest_int("max_depth", 4, 7),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 50, 150),
                "max_features": trial.suggest_float("max_features", 0.3, 0.6),
                "random_state": 42, "n_jobs": -1
            }
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
        elif model_name == "lightgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "num_leaves": trial.suggest_int("num_leaves", 7, 20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 50.0, 200.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 30, 100),
                "random_state": 42, "n_jobs": -1, "verbosity": -1
            }
            model = LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[xgb.callback.EarlyStopping(stopping_rounds=20)])

        elif model_name == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 300, 1000),
                "depth": trial.suggest_int("depth", 2, 4),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 20.0, 100.0),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "random_seed": 42, "verbose": False, "early_stopping_rounds": 30
            }
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))

        return accuracy_score(y_val, model.predict(X_val))

    return objective

def run_benchmark(tune=False):
    spark = SparkSession.builder.appName("IMDB-Final-Benchmark").getOrCreate()
    
    print("Loading finalized training data...")
    train_df = spark.read.parquet(os.path.join(PROCESSED_DIR, "train_final.parquet")).toPandas()
    X, y = prepare_data(train_df, is_train=True)
    
    # Holdout set (15%) to check final generalization before submitting
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.15, random_state=42)

    factories = {
        "catboost": lambda p: CatBoostClassifier(**p, verbose=False),
        "xgboost": lambda p: xgb.XGBClassifier(**p, eval_metric="logloss"),
        "lightgbm": lambda p: LGBMClassifier(**p, verbosity=-1),
        "random_forest": lambda p: RandomForestClassifier(**p)
    }

    model_stats = []
    best_clf, best_acc = None, 0

    print(f"Benchmarking {len(factories)} models...")
    for name in factories.keys():
        if tune:
            print(f"Optimizing {name} with Optuna...")
            study = optuna.create_study(direction="maximize")
            study.optimize(get_tuning_objective(name, X_train, y_train), n_trials=15)
            params = study.best_params
        else:
            params = {"random_state": 42, "max_depth": 4}

        # CatBoost expects random_seed, others use random_state
        if name == "catboost" and "random_state" in params:
            params["random_seed"] = params.pop("random_state")

        # Train final version of this model
        clf = factories[name](params)
        clf.fit(X_train, y_train)

        tr_score = accuracy_score(y_train, clf.predict(X_train))
        ho_score = accuracy_score(y_holdout, clf.predict(X_holdout))
        print(f"{name.upper():15} | Train Acc: {tr_score:.4f} | Holdout Acc: {ho_score:.4f}")

        model_stats.append({"model": name, "train": tr_score, "holdout": ho_score})
        
        if ho_score > best_acc:
            best_acc, best_clf = ho_score, clf

    # final results table
    print("\n FINAL BENCHMARK RANKING \n")
    ranking_df = pd.DataFrame(model_stats).sort_values("holdout", ascending=False)
    print(ranking_df.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split, filename in [("validation", "validation_final.parquet"), ("test", "test_final.parquet")]:
        path = os.path.join(PROCESSED_DIR, filename)
        if os.path.exists(path):
            eval_df = spark.read.parquet(path).toPandas()
            X_eval, _ = prepare_data(eval_df, is_train=False)
            
            preds = best_clf.predict(X_eval)
            
            output_path = os.path.join(OUTPUT_DIR, f"{split}.txt")
            with open(output_path, "w") as f:
                for p in preds:
                    f.write(f"{bool(p)}\n")
            print(f"[INFO] Created submission file: {output_path}")

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter optimization")
    args = parser.parse_args()
    
    run_benchmark(tune=args.tune)