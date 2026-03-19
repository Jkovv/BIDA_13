import warnings
warnings.filterwarnings("ignore")

import joblib, json, os, random
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, VotingClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna


# -- prestige (recomputed per CV fold) --

DIRECTING, WRITING, DIR_COUNT, WRI_COUNT = None, None, None, None

def _init_crew():
    global DIRECTING, WRITING, DIR_COUNT, WRI_COUNT
    if DIRECTING is not None:
        return
    with open("/app/imdb/directing.json") as f:
        d = json.load(f)
    DIRECTING = pd.DataFrame({"tconst": list(d["movie"].values()),
                               "director_id": list(d["director"].values())})
    with open("/app/imdb/writing.json") as f:
        w = json.load(f)
    WRITING = pd.DataFrame({"tconst": [x["movie"] for x in w],
                             "writer_id": [x["writer"] for x in w]})
    DIR_COUNT = DIRECTING.groupby("tconst").size().reset_index(name="n_directors")
    WRI_COUNT = WRITING.groupby("tconst").size().reset_index(name="n_writers")

def _bayes(df, id_col, label_col, k=20):
    gm = df[label_col].mean()
    s = df.groupby(id_col)[label_col].agg(["mean", "count"]).reset_index()
    s.columns = [id_col, "pm", "n"]
    s["prestige"] = (s["n"] / (s["n"] + k)) * s["pm"] + (k / (s["n"] + k)) * gm
    return s[[id_col, "prestige"]], gm

def add_prestige(train_df, target_df):
    _init_crew()
    td = train_df.merge(DIRECTING, on="tconst", how="left")
    tw = train_df.merge(WRITING, on="tconst", how="left")
    dp, dg = _bayes(td.dropna(subset=["director_id"]), "director_id", "label_int")
    wp, wg = _bayes(tw.dropna(subset=["writer_id"]), "writer_id", "label_int")

    ds = DIRECTING.merge(dp, on="director_id", how="left")
    ds["prestige"] = ds["prestige"].fillna(dg)
    dm = ds.groupby("tconst")["prestige"].mean().reset_index()
    dm.columns = ["tconst", "director_prestige"]

    ws = WRITING.merge(wp, on="writer_id", how="left")
    ws["prestige"] = ws["prestige"].fillna(wg)
    wm = ws.groupby("tconst")["prestige"].mean().reset_index()
    wm.columns = ["tconst", "writer_prestige"]

    drop_cols = ["director_prestige", "writer_prestige", "n_directors", "n_writers"]
    out = target_df.drop(columns=[c for c in drop_cols if c in target_df.columns])
    out = out.merge(dm, on="tconst", how="left")
    out = out.merge(wm, on="tconst", how="left")
    out = out.merge(DIR_COUNT, on="tconst", how="left")
    out = out.merge(WRI_COUNT, on="tconst", how="left")
    out["director_prestige"] = out["director_prestige"].fillna(dg)
    out["writer_prestige"] = out["writer_prestige"].fillna(wg)
    out["n_directors"] = out["n_directors"].fillna(1).astype(int)
    out["n_writers"] = out["n_writers"].fillna(1).astype(int)
    return out


# -- statistical feature selection --

def _bh_correction(pvals, alpha):
    n = len(pvals)
    idx = np.argsort(pvals)
    sorted_p = np.array(pvals)[idx]
    adjusted = np.ones(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        adj = min(prev, sorted_p[i] * n / (i + 1))
        adjusted[idx[i]] = adj
        prev = adj
    return adjusted < alpha, adjusted


def _partial_spearman(X, feat, controls, y_col='label_int'):
    ranked = X[[feat] + controls + [y_col]].rank()
    lr_x = LinearRegression().fit(ranked[controls], ranked[feat])
    lr_y = LinearRegression().fit(ranked[controls], ranked[y_col])
    resid_x = ranked[feat] - lr_x.predict(ranked[controls])
    resid_y = ranked[y_col] - lr_y.predict(ranked[controls])
    return spearmanr(resid_x, resid_y)


def select_features(X, y, cols, n_perms=100, mw_alpha=0.05, mi_alpha=0.1, corr_thresh=0.85):
    data = X.copy()
    data['label_int'] = y

    print("\n  1) Mann-Whitney U + Benjamini-Hochberg (alpha=0.05)")
    mw_pvals, mw_effects = [], []
    for c in cols:
        U, p = mannwhitneyu(X.loc[y==0, c].values, X.loc[y==1, c].values, alternative='two-sided')
        mw_pvals.append(p)
        mw_effects.append(1 - (2 * U) / (sum(y==0) * sum(y==1)))

    mw_reject, mw_padj = _bh_correction(mw_pvals, mw_alpha)
    print(f"     {'feature':25s} {'effect_r':>9} {'p_adj':>10} {'sig':>4}")
    for i, c in enumerate(cols):
        print(f"     {c:25s} {mw_effects[i]:9.4f} {mw_padj[i]:10.2e} {'*' if mw_reject[i] else '':>4}")

    print(f"\n  2) Permutation MI ({n_perms} shuffles) + BH (alpha=0.1)")
    real_mi = mutual_info_classif(X[cols], y, random_state=42, n_neighbors=5)
    rng = np.random.RandomState(42)
    null_mis = np.zeros((n_perms, len(cols)))
    for i in range(n_perms):
        null_mis[i] = mutual_info_classif(X[cols], rng.permutation(y), random_state=i, n_neighbors=5)

    perm_pvals = [(np.sum(null_mis[:, j] >= real_mi[j]) + 1) / (n_perms + 1) for j in range(len(cols))]
    mi_reject, mi_padj = _bh_correction(perm_pvals, mi_alpha)
    mi_dict = dict(zip(cols, real_mi))

    print(f"     {'feature':25s} {'MI':>7} {'null_95':>8} {'p_adj':>8} {'sig':>4}")
    for j, c in enumerate(cols):
        print(f"     {c:25s} {real_mi[j]:7.4f} {np.percentile(null_mis[:,j], 95):8.4f} {mi_padj[j]:8.3f} {'*' if mi_reject[j] else '':>4}")

    candidates = [c for i, c in enumerate(cols) if mw_reject[i] and mi_reject[i]]
    dropped = [c for c in cols if c not in candidates]
    if dropped:
        print(f"\n  Dropped (failed tests): {dropped}")
    print(f"  Passed: {candidates}")

    interaction_tests = [
        ('votes_x_runtime', ['log_votes', 'runtime']),
        ('vote_density', ['log_votes', 'film_age']),
        ('votes_per_minute', ['log_votes', 'runtime']),
        ('runtime_short', ['runtime']),
        ('runtime_long', ['runtime']),
        ('era_ord', ['film_age']),
        ('title_has_number', ['is_sequel']),
        ('title_length', ['title_word_count']),
        ('title_word_count', ['title_length']),
        ('bechdel_pass', ['bechdel_score']),
        ('ml_log_count', ['ml_rating_count']),
    ]

    partial_drops = set()
    relevant_tests = [(f, c) for f, c in interaction_tests
                      if f in candidates and all(x in candidates for x in c)]
    if relevant_tests:
        print(f"\n  3) Partial Spearman (interaction redundancy)")
        for feat, controls in relevant_tests:
            r, p = _partial_spearman(data, feat, controls)
            verdict = "keep" if p < 0.05 else "DROP"
            print(f"     {feat:25s} | {str(controls):30s} p={p:.3e} -> {verdict}")
            if p >= 0.05:
                partial_drops.add(feat)

    candidates = [c for c in candidates if c not in partial_drops]

    if len(candidates) > 1:
        corr = X[candidates].corr(method='spearman')
        to_drop = set()
        redundant_found = False
        for i, a in enumerate(candidates):
            for j, b in enumerate(candidates):
                if j <= i or a in to_drop or b in to_drop:
                    continue
                rho = abs(corr.loc[a, b])
                if rho > corr_thresh:
                    if not redundant_found:
                        print(f"\n  4) Spearman redundancy (|rho| > {corr_thresh})")
                        redundant_found = True
                    worse = a if mi_dict.get(a, 0) < mi_dict.get(b, 0) else b
                    print(f"     {a} ~ {b} (rho={rho:.3f}) -> drop {worse}")
                    to_drop.add(worse)
        candidates = [c for c in candidates if c not in to_drop]

    print(f"\n  SELECTED: {candidates} ({len(candidates)} features)")
    return candidates


# -- prepare --

def prepare(df, selected=None):
    num = ["runtime", "log_votes", "film_age", "vote_density", "votes_per_minute",
           "runtime_short", "runtime_long", "votes_x_runtime", "is_foreign",
           "director_prestige", "writer_prestige", "n_directors", "n_writers",
           "ml_rating_mean", "ml_rating_std", "ml_rating_count",
           "ml_rating_median", "ml_log_count", "ml_tag_count",
           "bechdel_score", "bechdel_pass"]
    genre = [c for c in df.columns if c.startswith("genre_")]
    mltag = [c for c in df.columns if c.startswith("mltag_")]
    all_cols = [c for c in num if c in df.columns] + genre + mltag

    X = df[all_cols].reset_index(drop=True)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    X = X.loc[:, ~X.columns.duplicated()]

    if selected:
        X = X[[c for c in selected if c in X.columns]]

    y = None
    if "label" in df.columns:
        y = LabelEncoder().fit_transform(df["label"].astype(str))
    return X.astype(np.float64), y


# -- leak-free CV --

def cv_leakfree(clf_fn, df, skf, selected=None, return_oof=False):
    y_all = LabelEncoder().fit_transform(df["label"].astype(str))
    accs = []
    oof_probs = np.zeros(len(df))
    oof_labels = np.zeros(len(df))
    for tr_idx, val_idx in skf.split(np.arange(len(df)), y_all):
        tr = df.iloc[tr_idx].copy().reset_index(drop=True)
        va = df.iloc[val_idx].copy().reset_index(drop=True)
        tr["label_int"] = ((tr["label"] == "True") | (tr["label"] == True)).astype(int)
        tr = add_prestige(tr, tr)
        va = add_prestige(tr, va)
        tr = tr.drop(columns=["label_int"], errors="ignore")
        X_tr, y_tr = prepare(tr, selected)
        X_va, y_va = prepare(va, selected)
        clf = clf_fn()
        clf.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_va, clf.predict(X_va)))
        if return_oof:
            probs = clf.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = probs
            oof_labels[val_idx] = y_va
    if return_oof:
        return np.mean(accs), np.std(accs), oof_probs, oof_labels
    return np.mean(accs), np.std(accs)


# -- main --

def run_benchmark():
    print("Step 6: Training")

    feat_path = "/app/processed/train_features.parquet"
    if not os.path.exists(feat_path):
        print("  missing train_features.parquet")
        return

    df = pd.read_parquet(feat_path)
    print(f"  {len(df)} movies")

    # feature selection
    df["label_int"] = ((df["label"] == "True") | (df["label"] == True)).astype(int)
    df_pres = add_prestige(df, df)
    X_tmp, _ = prepare(df_pres)
    y_tmp = df_pres["label_int"].values
    selected = select_features(X_tmp, y_tmp, list(X_tmp.columns))
    df = df.drop(columns=["label_int"])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # all models listed for reference, only ET is active
    models = {
        # 'xgb':  lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
        #                                random_state=42, eval_metric='logloss'),
        # 'lgbm': lambda: LGBMClassifier(n_estimators=300, max_depth=4, random_state=42, verbose=-1),
        # 'cat':  lambda: CatBoostClassifier(iterations=300, depth=4, verbose=False, random_seed=42),
        # 'rf':   lambda: RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        # 'gbm':  lambda: GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
        #                                             random_state=42),
        'et':   lambda: ExtraTreesClassifier(n_estimators=500, max_depth=16, random_state=42, n_jobs=-1),
    }

    print("\n  -- Phase 1: ET default --")
    default_fn = models['et']
    default_m, default_s = cv_leakfree(default_fn, df, skf, selected)
    print(f"  et   : {default_m:.4f} +/- {default_s:.4f}")

    # -- Phase 2: Optuna tuning --
    print("\n  -- Phase 2: Optuna ET tuning (80 trials) --")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500, step=100),
            'max_depth': trial.suggest_categorical('max_depth', [10, 12, 14, 16, 18, 20, 24, None]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features',
                            ['sqrt', 'log2', 0.3, 0.4, 0.5, 0.6, 0.7]),
        }
        fn = lambda: ExtraTreesClassifier(random_state=42, n_jobs=-1, **params)
        m, s = cv_leakfree(fn, df, skf, selected)
        print(f"    [{trial.number+1}] {m:.4f}  {params}")
        return m

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=80)

    best_params = study.best_params
    best_score = study.best_value
    print(f"\n  Optuna best: {best_score:.4f}  {best_params}")

    # use whichever is better — default or optuna
    if best_score > default_m:
        print(f"  Optuna wins (+{best_score - default_m:.4f})")
        champ_fn = lambda: ExtraTreesClassifier(random_state=42, n_jobs=-1, **best_params)
        final_score = best_score
    else:
        print(f"  Default wins ({default_m:.4f} >= {best_score:.4f})")
        champ_fn = default_fn
        final_score = default_m

    # -- Phase 3: Threshold tuning --
    print("\n  -- Phase 3: Threshold tuning --")
    _, _, oof_probs, oof_labels = cv_leakfree(champ_fn, df, skf, selected, return_oof=True)
    best_thresh, best_thresh_acc = 0.5, 0.0
    for t in np.arange(0.30, 0.71, 0.01):
        acc = accuracy_score(oof_labels, (oof_probs >= t).astype(int))
        if acc > best_thresh_acc:
            best_thresh_acc, best_thresh = acc, t
    print(f"  OOF accuracy at 0.5:  {accuracy_score(oof_labels, (oof_probs >= 0.5).astype(int)):.4f}")
    print(f"  Best threshold: {best_thresh:.2f} -> {best_thresh_acc:.4f}")

    # -- Final --
    print("\n  -- Final --")
    df["label_int"] = ((df["label"] == "True") | (df["label"] == True)).astype(int)
    final = add_prestige(df, df)
    final = final.drop(columns=["label_int"], errors="ignore")
    X_all, y_all = prepare(final, selected)
    print(f"  {X_all.shape[1]} features: {list(X_all.columns)}")
    print(f"  threshold: {best_thresh:.2f}")

    champ = champ_fn()
    champ.fit(X_all, y_all)
    os.makedirs('/app/output', exist_ok=True)
    joblib.dump(champ, "/app/output/best_model.pkl")

    for split in ["validation", "test"]:
        csv_path = f"/app/imdb/{split}_hidden.csv"
        fp = f"/app/processed/{split}_hidden_features.parquet"
        if not (os.path.exists(csv_path) and os.path.exists(fp)):
            continue
        order = pd.read_csv(csv_path)[['tconst']]
        feats = pd.read_parquet(fp)
        merged = order.merge(feats, on='tconst', how='left')
        merged = add_prestige(df, merged)
        X_eval, _ = prepare(merged, selected)
        probs = champ.predict_proba(X_eval)[:, 1]
        preds = (probs >= best_thresh).astype(int)
        with open(f"/app/output/{split}.txt", "w") as f:
            for p in preds:
                f.write(f"{bool(p)}\n")
        print(f"  {split}.txt: {sum(bool(p) for p in preds)}/{len(preds)} True")

    print("\nDone.")


if __name__ == "__main__":
    run_benchmark()