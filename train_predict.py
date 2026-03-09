"""
train_predict.py - trains XGBoost, generates submission files, runs ablations.

Usage:
    python train_predict.py              # train and generate predictions
    python train_predict.py --ablation   # run the 3 ablation experiments
"""
import argparse
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from pyspark.sql import SparkSession
import unicodedata
import duckdb
import json


DATA_DIR = "/app/imdb"             # raw csv/json files
PROCESSED_DIR = "/app/processed"   # parquet from cleaning/features
OUTPUT_DIR = "/app/output"         # prediction txt files

FEATURE_COLS = [
    "year", "decade", "movie_age",
    "numVotes_imputed", "log_numVotes", "runtimeMinutes_winsorised", "votes_per_minute",
    "n_directors", "n_writers", "crew_size", "is_multi_director",
    "is_foreign", "is_title_corrupted",
    "title_length", "title_word_count",
    "director_avg_success", "director_max_experience",
    "writer_avg_success", "writer_max_experience",
]

SEEDS = [42, 123, 456, 789, 1024]


def train_and_predict(data_dir=PROCESSED_DIR, output_dir=OUTPUT_DIR, seed=42):
    spark = SparkSession.builder.appName("IMDB-Train").getOrCreate()
    train_pd = spark.read.parquet(os.path.join(data_dir, "train_final.parquet")).toPandas()
    print(f"Loaded {len(train_pd)} training rows")

    tfidf_cols = []
    if "title_tfidf" in train_pd.columns:
        mat = np.array(train_pd["title_tfidf"].apply(lambda x: x.toArray()).tolist())
        tfidf_cols = [f"tfidf_{i}" for i in range(mat.shape[1])]
        train_pd = pd.concat([train_pd.reset_index(drop=True),
                               pd.DataFrame(mat, columns=tfidf_cols)], axis=1)

    all_features = FEATURE_COLS + tfidf_cols

    for col in all_features:
        if col in train_pd.columns and train_pd[col].dtype == bool:
            train_pd[col] = train_pd[col].astype(int)

    X = train_pd[all_features].fillna(0).values
    y = train_pd["label"].astype(int).values

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, eval_metric="logloss", use_label_encoder=False,
    )
    model.fit(X, y)
    print(f"Train accuracy: {(model.predict(X) == y).mean():.4f}")

    importances = sorted(zip(all_features, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nTop 10 features:")
    for name, imp in importances[:10]:
        print(f"  {name}: {imp:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    for split_name, filename, out_file in [
        ("validation", "validation_final.parquet", "validation.txt"),
        ("test", "test_final.parquet", "test.txt"),
    ]:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            print(f"Skipping {split_name} - not found")
            continue

        sp = spark.read.parquet(path).toPandas()
        if "title_tfidf" in sp.columns:
            m = np.array(sp["title_tfidf"].apply(lambda x: x.toArray()).tolist())
            sp = pd.concat([sp.reset_index(drop=True), pd.DataFrame(m, columns=tfidf_cols)], axis=1)
        for col in all_features:
            if col in sp.columns and sp[col].dtype == bool:
                sp[col] = sp[col].astype(int)

        preds = model.predict(sp[all_features].fillna(0).values)
        with open(os.path.join(output_dir, out_file), "w") as f:
            for p in preds:
                f.write("True\n" if p == 1 else "False\n")
        print(f"{split_name}: {len(preds)} predictions -> {out_file}")

    spark.stop()


# ----- ablation helpers -----

def strip_diacritics(text):
    if pd.isna(text):
        return text
    return ''.join(c for c in unicodedata.normalize('NFKD', str(text)) if not unicodedata.combining(c))


def load_raw(data_dir=DATA_DIR):
    con = duckdb.connect()
    train = con.execute(f"SELECT * FROM read_csv_auto('{data_dir}/train-*.csv', header=true)").fetchdf()
    with open(f"{data_dir}/directing.json") as f:
        d = json.load(f)
    directing = pd.DataFrame({"tconst": d["movie"].values(), "director_id": d["director"].values()})
    con.close()
    return train, directing


def prep(df, directing):
    df = df.copy()
    df['year'] = pd.to_numeric(df['startYear'].replace('\\N', pd.NA), errors='coerce')
    mask = df['year'].isna()
    df.loc[mask, 'year'] = pd.to_numeric(df.loc[mask, 'endYear'].replace('\\N', pd.NA), errors='coerce')
    df['decade'] = (df['year'] // 10 * 10)
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'].replace('\\N', pd.NA), errors='coerce')
    df['runtimeMinutes'] = df['runtimeMinutes'].fillna(df['runtimeMinutes'].median())
    crew = directing.groupby('tconst').size().reset_index(name='n_directors')
    df = df.merge(crew, on='tconst', how='left')
    df['n_directors'] = df['n_directors'].fillna(0)
    df['label_int'] = df['label'].astype(int)
    return df


def evaluate(X, y):
    accs = []
    for seed in SEEDS:
        m = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                               random_state=seed, eval_metric='logloss', use_label_encoder=False)
        accs.append(cross_val_score(m, X, y, cv=5, scoring='accuracy').mean())
    return accs


def run_ablations(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    train, directing = load_raw(data_dir)
    results = []

    print("\n--- Experiment A: Diacritics Cleaning ---")
    feats = ['year', 'decade', 'runtimeMinutes', 'numVotes', 'n_directors', 'title_len']
    for name, func in [
        ("V0: dirty titles", lambda x: x),
        ("V1: strip non-ascii", lambda x: x.encode('ascii', 'ignore').decode() if pd.notna(x) else x),
        ("V2: NFKD normalization", strip_diacritics),
    ]:
        df = prep(train, directing)
        df['title_len'] = df['primaryTitle'].apply(func).str.len()
        df['numVotes'] = df['numVotes'].fillna(df.groupby('decade')['numVotes'].transform('median'))
        accs = evaluate(df[feats].fillna(0).values, df['label_int'].values)
        print(f"  {name}: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        results.append({"experiment": "A", "variant": name, "mean": np.mean(accs), "std": np.std(accs)})

    print("\n--- Experiment B: Year Swap Repair ---")
    feats = ['year', 'decade', 'runtimeMinutes', 'numVotes', 'n_directors']
    for name, coalesce, drop in [
        ("V0: raw startYear", False, False),
        ("V1: drop null years", False, True),
        ("V2: coalesce start+end", True, False),
    ]:
        df = train.copy()
        df['year'] = pd.to_numeric(df['startYear'].replace('\\N', pd.NA), errors='coerce')
        if coalesce:
            mask = df['year'].isna()
            df.loc[mask, 'year'] = pd.to_numeric(df.loc[mask, 'endYear'].replace('\\N', pd.NA), errors='coerce')
        if drop:
            df = df.dropna(subset=['year'])
        else:
            df['year'] = df['year'].fillna(df['year'].median())
        df['decade'] = (df['year'] // 10 * 10)
        df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'].replace('\\N', pd.NA), errors='coerce').fillna(100)
        df['numVotes'] = df['numVotes'].fillna(df.groupby('decade')['numVotes'].transform('median'))
        crew = directing.groupby('tconst').size().reset_index(name='n_directors')
        df = df.merge(crew, on='tconst', how='left').fillna(0)
        df['label_int'] = df['label'].astype(int)
        accs = evaluate(df[feats].fillna(0).values, df['label_int'].values)
        print(f"  {name}: {np.mean(accs):.4f} +/- {np.std(accs):.4f} ({len(df)} rows)")
        results.append({"experiment": "B", "variant": name, "mean": np.mean(accs), "std": np.std(accs)})

    print("\n--- Experiment C: numVotes Imputation ---")
    df_base = prep(train, directing)
    for name, func in [
        ("V0: drop nulls", lambda d: d.dropna(subset=['numVotes'])),
        ("V1: global median", lambda d: d.fillna({'numVotes': d['numVotes'].median()})),
        ("V2: median per decade", lambda d: d.assign(numVotes=d['numVotes'].fillna(d.groupby('decade')['numVotes'].transform('median')))),
        ("V3: fill with zero", lambda d: d.fillna({'numVotes': 0})),
    ]:
        df = func(df_base.copy())
        df['label_int'] = df['label'].astype(int)
        accs = evaluate(df[feats].fillna(0).values, df['label_int'].values)
        print(f"  {name}: {np.mean(accs):.4f} +/- {np.std(accs):.4f} ({len(df)} rows)")
        results.append({"experiment": "C", "variant": name, "mean": np.mean(accs), "std": np.std(accs)})

    out = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ablation_results.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    if args.ablation:
        run_ablations()
    else:
        train_and_predict()