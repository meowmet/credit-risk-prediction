from pathlib import Path
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    log_loss,
)

import joblib

# SHAP + plotting
try:
    import shap
    import matplotlib.pyplot as plt
except ImportError:
    shap = None
    plt = None
    print("[WARN] shap or matplotlib not installed. SHAP plots will be skipped.")


# ================== GLOBAL CONFIG ==================

FE_V1_PATH = Path("credit_500k_fe_v1.csv")

# Outputs for multi-version FE + RF (Day 3–4 style)
OUTPUT_DIR_MULTI = Path("outputs_500k_day3_4")
OUTPUT_DIR_MULTI.mkdir(parents=True, exist_ok=True)

# Outputs for advanced RF + feature selection
OUTPUT_DIR_RF_ONLY = Path("outputs_500k_rf_only")
OUTPUT_DIR_RF_ONLY.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target_default_500k"
RANDOM_STATE = 42

# Training caps
MAX_TRAIN_ROWS_MULTI = 300_000      # for multi-version RF
MAX_TRAIN_ROWS_ADV = 300_000        # for advanced RF

# SHAP config (LIGHT mode to avoid freezing)
SHAP_SAMPLE_SIZE_MULTI = 300        # was 5000
RUN_SHAP_ADV = False                # disable advanced SHAP (heavy)
SHAP_SAMPLE_SIZE_ADV = 200          # kept small in case you turn it on

# Feature-selection threshold for advanced RF
FEATURE_SELECTION_THRESHOLD = 0.51


# ================== COMMON HELPERS ==================

def report_memory(df: pd.DataFrame, label: str) -> None:
    mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"[MEM] {label}: {mb:.2f} MB")


def load_fe_v1(path: Path = FE_V1_PATH) -> pd.DataFrame:
    print(f"[LOAD] Reading FE v1 from {path} ...")
    df = pd.read_csv(path)
    print(f"[INFO] FE v1 shape: {df.shape}")
    report_memory(df, "FE v1")
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} not found in FE v1.")
    return df


def clean_X_for_model(X: pd.DataFrame) -> pd.DataFrame:
    """
    Keep ONLY numeric columns and make them model-safe:
      - Drop all non-numeric columns (raw strings, categories, etc.)
      - Replace ±inf with NaN
      - Fill NaN with median (per column)
      - Clip to [-1e9, 1e9]
      - Cast numeric columns to float64
    """
    # 1) Keep only numeric columns
    X = X.select_dtypes(include=[np.number]).copy()

    # 2) Replace infinities
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3) Median impute + clip + cast
    for col in X.columns:
        col_data = X[col]
        if col_data.isna().any():
            col_data = col_data.fillna(col_data.median())
        col_data = np.clip(col_data, -1e9, 1e9)
        X[col] = col_data.astype("float64")

    return X


def re_impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace inf/-inf with NaN, then fill numeric NaNs with median and downcast.
    Used during FE construction on the full dataset.
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL)

    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Downcast floats/ints to reduce memory
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def safe_add_feature(df: pd.DataFrame, name: str, func) -> None:
    """Add a derived feature if all required columns exist."""
    try:
        df[name] = func(df)
    except Exception:
        print(f"[FE] Skipping feature {name} (missing columns or error).")


def make_consistent_splits(n_rows: int, y: np.ndarray) -> Dict[str, np.ndarray]:
    """Create consistent stratified train/val/test splits."""
    all_idx = np.arange(n_rows)
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        all_idx, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"[SPLIT] Sizes: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    return {"train": idx_train, "val": idx_val, "test": idx_test}


# ================== FE v2 / v3 (multi-version block) ==================

def build_fe_v2(df_v1: pd.DataFrame) -> pd.DataFrame:
    """
    FE v2: start from FE v1, add:
      - log transforms
      - extra ratios and spreads
    """
    print("[FE] Building FE v2 from FE v1 ...")
    df = df_v1.copy()

    # Log transforms
    safe_add_feature(df, "log_loan_amnt",
                     lambda d: np.log1p(d["loan_amnt"]))
    safe_add_feature(df, "log_annual_inc",
                     lambda d: np.log1p(d["annual_inc"]))
    safe_add_feature(df, "log_revol_bal",
                     lambda d: np.log1p(d["revol_bal"]))
    safe_add_feature(df, "log_total_rev_hi_lim",
                     lambda d: np.log1p(d["total_rev_hi_lim"]))

    # FICO mean and spread
    safe_add_feature(
        df, "fico_mean_v2",
        lambda d: 0.5 * (d["fico_range_low"] + d["fico_range_high"])
    )
    safe_add_feature(
        df, "fico_spread",
        lambda d: d["fico_range_high"] - d["fico_range_low"]
    )

    # Ratios
    safe_add_feature(
        df, "loan_to_total_rev",
        lambda d: d["loan_amnt"] / np.where(d["total_rev_hi_lim"] > 0,
                                            d["total_rev_hi_lim"], np.nan)
    )
    safe_add_feature(
        df, "revol_util_alt_v2",
        lambda d: d["revol_bal"] / np.where(d["total_rev_hi_lim"] > 0,
                                            d["total_rev_hi_lim"], np.nan)
    )
    safe_add_feature(
        df, "open_to_total_acc_ratio",
        lambda d: d["open_acc"] / np.where(d["total_acc"] > 0,
                                           d["total_acc"], np.nan)
    )
    safe_add_feature(
        df, "delinq_to_total_acc",
        lambda d: d["delinq_2yrs"] / np.where(d["total_acc"] > 0,
                                              d["total_acc"], 1.0)
    )

    # Term * rate interaction
    safe_add_feature(
        df, "term_x_int_rate",
        lambda d: d["term"] * d["int_rate"]
    )

    df = re_impute_numeric(df)
    report_memory(df, "FE v2")
    print("[FE] FE v2 built with", df.shape[1], "columns.")
    return df


def build_fe_v3(df_v2: pd.DataFrame) -> pd.DataFrame:
    """
    FE v3: start from FE v2, add coarse bins (risk bands)
      - fico_mean_v2 bins
      - dti bins
      - loan_to_income bins
    """
    print("[FE] Building FE v3 from FE v2 ...")
    df = df_v2.copy()

    # FICO bands
    if "fico_mean_v2" in df.columns:
        try:
            df["fico_band"] = pd.cut(
                df["fico_mean_v2"],
                bins=[0, 640, 680, 720, 760, 900],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True,
            ).astype("float").fillna(0)
        except Exception as e:
            print(f"[FE] Skipping fico_band: {e}")

    # DTI bands
    if "dti" in df.columns:
        try:
            df["dti_band"] = pd.cut(
                df["dti"],
                bins=[0, 10, 20, 30, 40, 1000],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True,
            ).astype("float").fillna(0)
        except Exception as e:
            print(f"[FE] Skipping dti_band: {e}")

    # loan_to_income bands
    if "loan_to_income" in df.columns:
        try:
            df["loan_to_income_band"] = pd.cut(
                df["loan_to_income"],
                bins=[0, 0.1, 0.2, 0.3, 0.5, 100],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True,
            ).astype("float").fillna(0)
        except Exception as e:
            print(f"[FE] Skipping loan_to_income_band: {e}")

    df = re_impute_numeric(df)
    report_memory(df, "FE v3")
    print("[FE] FE v3 built with", df.shape[1], "columns.")
    return df


# ================== RF training (multi-version) ==================

def train_rf_for_version(
    fe_name: str,
    df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    max_train_rows: int = MAX_TRAIN_ROWS_MULTI,
) -> Tuple[RandomForestClassifier, Dict[str, Dict[str, float]]]:
    """
    Train a RandomForest on given FE version and return:
      - trained model
      - metrics dict for val/test
    """
    print(f"\n[RF] Training RF for {fe_name} ...")

    y_all = df[TARGET_COL].values
    X_all_raw = df.drop(columns=[TARGET_COL])
    X_all = clean_X_for_model(X_all_raw)

    idx_train = splits["train"]
    idx_val = splits["val"]
    idx_test = splits["test"]

    # Optional subsampling
    if (max_train_rows is not None) and (len(idx_train) > max_train_rows):
        rng = np.random.RandomState(RANDOM_STATE)
        idx_train = rng.choice(idx_train, size=max_train_rows, replace=False)
        print(f"[RF] Subsampling train to {len(idx_train)} rows.")

    X_train = X_all.iloc[idx_train]
    y_train = y_all[idx_train]
    X_val = X_all.iloc[idx_val]
    y_val = y_all[idx_val]
    X_test = X_all.iloc[idx_test]
    y_test = y_all[idx_test]

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    print(f"[RF] Fitting RF on {X_train.shape[0]} rows, {X_train.shape[1]} features ...")
    rf.fit(X_train, y_train)

    metrics: Dict[str, Dict[str, float]] = {}

    def eval_split(name: str, X, y) -> Dict[str, float]:
        proba = rf.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y, proba)
        f1 = f1_score(y, preds)
        prec = precision_score(y, preds)
        rec = recall_score(y, preds)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        print(f"[RF][{fe_name}][{name}] AUC={auc:.4f} F1={f1:.4f} "
              f"Prec={prec:.4f} Rec={rec:.4f}")
        return {
            "auc": float(auc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
        }

    metrics["val"] = eval_split("val", X_val, y_val)
    metrics["test"] = eval_split("test", X_test, y_test)

    model_path = OUTPUT_DIR_MULTI / f"rf_500k_{fe_name}.pkl"
    joblib.dump(rf, model_path)
    print(f"[RF] Saved model for {fe_name} -> {model_path}")

    return rf, metrics


# ================== Univariate AUC (multi-version) ==================

def compute_univariate_auc(
    df: pd.DataFrame,
    target_col: str,
    out_path: Path,
) -> None:
    """
    Compute univariate AUC for each feature vs target.
    Uses symmetric AUC = max(AUC, 1-AUC).
    """
    print("[AUC1D] Computing univariate AUC for all features ...")

    y = df[target_col].values
    X_raw = df.drop(columns=[target_col])
    X = clean_X_for_model(X_raw)

    results: List[Dict[str, float]] = []

    for col in X.columns:
        x = X[col].values
        if np.unique(x).size <= 1:
            continue
        try:
            auc = roc_auc_score(y, x)
            auc_sym = max(auc, 1.0 - auc)
            results.append({
                "feature": col,
                "auc_raw": float(auc),
                "auc_sym": float(auc_sym),
            })
        except ValueError:
            continue

    res_df = pd.DataFrame(results).sort_values("auc_sym", ascending=False)
    res_df.to_csv(out_path, index=False)
    print(f"[AUC1D] Saved univariate AUC results -> {out_path}")


# ================== SHAP (multi-version) ==================

def run_shap_analysis_multi(
    fe_name: str,
    model: RandomForestClassifier,
    df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    sample_size: int = SHAP_SAMPLE_SIZE_MULTI,
) -> None:
    if shap is None or plt is None:
        print("[SHAP] shap/matplotlib not available; skipping SHAP.")
        return

    if sample_size <= 0:
        print("[SHAP] sample_size <= 0; skipping SHAP.")
        return

    print(f"[SHAP] Running SHAP for {fe_name} (multi-version block) ...")

    X_all_raw = df.drop(columns=[TARGET_COL])
    X_all = clean_X_for_model(X_all_raw)

    idx_val = splits["val"]
    X_val = X_all.iloc[idx_val]

    if len(X_val) > sample_size:
        rng = np.random.RandomState(RANDOM_STATE)
        sample_idx = rng.choice(X_val.index, size=sample_size, replace=False)
        X_sample = X_val.loc[sample_idx]
    else:
        X_sample = X_val

    print(f"[SHAP] Using {X_sample.shape[0]} rows, {X_sample.shape[1]} features.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Class 1 (default)
    shap.summary_plot(
        shap_values[1],
        X_sample,
        show=False,
        max_display=25,
    )
    shap_path_beeswarm = OUTPUT_DIR_MULTI / f"shap_summary_beeswarm_{fe_name}.png"
    plt.tight_layout()
    plt.savefig(shap_path_beeswarm, dpi=200)
    plt.close()
    print(f"[SHAP] Saved beeswarm summary -> {shap_path_beeswarm}")

    shap.summary_plot(
        shap_values[1],
        X_sample,
        show=False,
        plot_type="bar",
        max_display=25,
    )
    shap_path_bar = OUTPUT_DIR_MULTI / f"shap_summary_bar_{fe_name}.png"
    plt.tight_layout()
    plt.savefig(shap_path_bar, dpi=200)
    plt.close()
    print(f"[SHAP] Saved bar summary -> {shap_path_bar}")


def run_multi_version_block() -> None:
    """
    Block 1: FE v1 → v2 → v3, RF comparison, univariate AUC, SHAP on best.
    """
    print("\n========== BLOCK 1: Multi-version FE + RF + SHAP (500k) ==========\n")

    # Load and basic re-impute
    df_v1 = load_fe_v1()
    df_v1 = re_impute_numeric(df_v1)

    # Build FE v2 and v3
    df_v2 = build_fe_v2(df_v1)
    df_v3 = build_fe_v3(df_v2)

    # Splits
    y = df_v1[TARGET_COL].values
    splits = make_consistent_splits(len(df_v1), y)

    # Train RF for each FE version
    metrics_all: Dict[str, Dict[str, Dict[str, float]]] = {}
    models: Dict[str, RandomForestClassifier] = {}

    for fe_name, df in [("fe_v1", df_v1), ("fe_v2", df_v2), ("fe_v3", df_v3)]:
        rf, metrics = train_rf_for_version(fe_name, df, splits)
        metrics_all[fe_name] = metrics
        models[fe_name] = rf

    # Comparison table
    rows = []
    for fe_name, m in metrics_all.items():
        row = {"fe_version": fe_name}
        for split_name in ["val", "test"]:
            for k, v in m[split_name].items():
                row[f"{split_name}_{k}"] = v
        rows.append(row)

    cmp_df = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    cmp_path = OUTPUT_DIR_MULTI / "rf_500k_comparison.csv"
    cmp_df.to_csv(cmp_path, index=False)
    print(f"\n[RESULT] Saved RF comparison -> {cmp_path}")
    print(cmp_df)

    # Univariate AUC on v3
    auc_path = OUTPUT_DIR_MULTI / "univariate_auc_500k_v3.csv"
    compute_univariate_auc(df_v3, TARGET_COL, auc_path)

    # SHAP on best FE version (light mode)
    best_fe_name = cmp_df.iloc[0]["fe_version"]
    print(f"\n[RESULT] Best FE version by val AUC: {best_fe_name}")
    best_model = models[best_fe_name]
    if best_fe_name == "fe_v1":
        best_df = df_v1
    elif best_fe_name == "fe_v2":
        best_df = df_v2
    else:
        best_df = df_v3

    run_shap_analysis_multi(best_fe_name, best_model, best_df, splits)

    print("\n[BLOCK 1 DONE] Multi-version FE + RF + SHAP completed.\n")


# ================== Advanced FE (for RF-only block) ==================

def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced FE designed for RandomForest (no leakage):
      - logs: loan_amnt, annual_inc, revol_bal, total_rev_hi_lim
      - FICO: mean, spread, squared, bands
      - ratios: loan_to_income, installment_to_income, loan_to_total_rev,
                revol_util_computed, open_to_total_acc, delinq_to_total_acc
      - interactions: int_rate_x_dti, int_rate_x_loan_amnt, term_x_int_rate
      - polynomial: dti_sq
      - risk flags: high_dti_flag, delinq_flag, pub_rec_flag, risk_score
      - bands: dti_band, loan_to_income_band
    """
    print("[FE_ADV] Building advanced features ...")
    df = df.copy()

    # Log transforms
    for col in ["loan_amnt", "annual_inc", "revol_bal", "total_rev_hi_lim"]:
        if col in df.columns:
            safe_add_feature(df, f"log_{col}", lambda d, c=col: np.log1p(d[c]))

    # FICO features
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_mean"] = 0.5 * (df["fico_range_low"] + df["fico_range_high"])
        df["fico_spread"] = df["fico_range_high"] - df["fico_range_low"]
        safe_add_feature(df, "fico_mean_sq", lambda d: d["fico_mean"] ** 2)

        try:
            df["fico_band"] = pd.cut(
                df["fico_mean"],
                bins=[0, 640, 680, 720, 760, 900],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True,
            ).astype("float").fillna(2)
        except Exception as e:
            print(f"[FE_ADV] Skipping fico_band: {e}")

    # Ratios
    safe_add_feature(
        df, "loan_to_income",
        lambda d: d["loan_amnt"] / np.where(d["annual_inc"] > 0, d["annual_inc"], np.nan)
    )
    safe_add_feature(
        df, "installment_to_income",
        lambda d: d["installment"] / np.where(d["annual_inc"] > 0, d["annual_inc"] / 12, np.nan)
    )
    safe_add_feature(
        df, "loan_to_total_rev",
        lambda d: d["loan_amnt"] / np.where(d["total_rev_hi_lim"] > 0, d["total_rev_hi_lim"], np.nan)
    )
    safe_add_feature(
        df, "revol_util_computed",
        lambda d: d["revol_bal"] / np.where(d["total_rev_hi_lim"] > 0, d["total_rev_hi_lim"], np.nan)
    )
    safe_add_feature(
        df, "open_to_total_acc",
        lambda d: d["open_acc"] / np.where(d["total_acc"] > 0, d["total_acc"], np.nan)
    )
    safe_add_feature(
        df, "delinq_to_total_acc",
        lambda d: d["delinq_2yrs"] / np.where(d["total_acc"] > 0, d["total_acc"], 1.0)
    )

    # Polynomial
    if "dti" in df.columns:
        df["dti_sq"] = df["dti"] ** 2

    # Interactions
    safe_add_feature(df, "int_rate_x_dti", lambda d: d["int_rate"] * d["dti"])
    safe_add_feature(df, "int_rate_x_loan_amnt", lambda d: d["int_rate"] * d["loan_amnt"])
    safe_add_feature(df, "term_x_int_rate", lambda d: d["term"] * d["int_rate"])

    # Bands
    if "dti" in df.columns:
        try:
            df["dti_band"] = pd.cut(
                df["dti"],
                bins=[0, 10, 20, 30, 40, 1000],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True,
            ).astype("float").fillna(2)
        except Exception as e:
            print(f"[FE_ADV] Skipping dti_band: {e}")

    if "loan_to_income" in df.columns:
        try:
            df["loan_to_income_band"] = pd.cut(
                df["loan_to_income"],
                bins=[0, 0.1, 0.2, 0.3, 0.5, 100],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True,
            ).astype("float").fillna(2)
        except Exception as e:
            print(f"[FE_ADV] Skipping loan_to_income_band: {e}")

    # Risk flags
    df["high_dti_flag"] = (df.get("dti", 0) > 30).astype(int)
    df["delinq_flag"] = (df.get("delinq_2yrs", 0) > 0).astype(int)
    df["pub_rec_flag"] = (df.get("pub_rec", 0) > 0).astype(int)

    df["risk_score"] = (
        df.get("high_dti_flag", 0)
        + df.get("delinq_flag", 0)
        + df.get("pub_rec_flag", 0)
    )

    df = re_impute_numeric(df)
    report_memory(df, "Advanced FE")
    print(f"[FE_ADV] Advanced features built: {df.shape[1]} columns.")
    return df


# ================== Feature selection + RF (advanced block) ==================

def select_features_by_auc(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    threshold: float = FEATURE_SELECTION_THRESHOLD,
) -> List[str]:
    """
    Select features using univariate AUC on the train set.
    Keep features where max(AUC, 1-AUC) >= threshold.
    """
    print(f"[FEAT_SELECT] Selecting features with symmetric AUC >= {threshold} ...")
    selected = []

    for col in X_train.columns:
        x = X_train[col].values
        if np.unique(x).size <= 1:
            continue
        try:
            auc = roc_auc_score(y_train, x)
            auc_sym = max(auc, 1.0 - auc)
            if auc_sym >= threshold:
                selected.append(col)
        except ValueError:
            continue

    print(f"[FEAT_SELECT] Selected {len(selected)} / {X_train.shape[1]} features.")
    return selected


def train_rf_advanced(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """Train a tuned RandomForest classifier."""
    print("[RF_ADV] Training RandomForest classifier ...")

    class_weight = {0: 1.0, 1: 2.0}

    model = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
        oob_score=False,
    )

    model.fit(X_train, y_train)
    return model


def evaluate_at_threshold(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    thr: float,
) -> Dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= thr).astype(int)

    auc = roc_auc_score(y, proba)
    ll = log_loss(y, proba)
    f1 = f1_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

    return {
        "auc": float(auc),
        "logloss": float(ll),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def find_best_threshold_for_f1(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    thr_min: float = 0.1,
    thr_max: float = 0.9,
    thr_step: float = 0.01,
) -> Tuple[float, float]:
    proba = model.predict_proba(X_val)[:, 1]
    best_thr = 0.5
    best_f1 = -1.0

    thr_values = np.arange(thr_min, thr_max + 1e-6, thr_step)
    for thr in thr_values:
        preds = (proba >= thr).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr, best_f1


def run_shap_analysis_advanced(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    model_name: str = "rf_500k_advanced",
) -> None:
    if not RUN_SHAP_ADV:
        print("[SHAP_ADV] RUN_SHAP_ADV is False; skipping SHAP.")
        return

    if shap is None or plt is None:
        print("[SHAP_ADV] shap/matplotlib not available; skipping SHAP.")
        return

    if SHAP_SAMPLE_SIZE_ADV <= 0:
        print("[SHAP_ADV] SHAP_SAMPLE_SIZE_ADV <= 0; skipping SHAP.")
        return

    print("[SHAP_ADV] Running SHAP analysis ...")

    if len(X_val) > SHAP_SAMPLE_SIZE_ADV:
        rng = np.random.RandomState(RANDOM_STATE)
        sample_idx = rng.choice(X_val.index, size=SHAP_SAMPLE_SIZE_ADV, replace=False)
        X_sample = X_val.loc[sample_idx]
    else:
        X_sample = X_val

    print(f"[SHAP_ADV] Using {X_sample.shape[0]} rows, {X_sample.shape[1]} features.")

    explainer = shap.TreeExplainer(model)
    shap_values_list = explainer.shap_values(X_sample, check_additivity=False)

    shap_vals = shap_values_list[1]

    shap.summary_plot(
        shap_vals,
        X_sample,
        show=False,
        max_display=25,
    )
    beeswarm_path = OUTPUT_DIR_RF_ONLY / f"shap_beeswarm_{model_name}.png"
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200)
    plt.close()
    print(f"[SHAP_ADV] Saved beeswarm plot -> {beeswarm_path}")

    shap.summary_plot(
        shap_vals,
        X_sample,
        show=False,
        plot_type="bar",
        max_display=25,
    )
    bar_path = OUTPUT_DIR_RF_ONLY / f"shap_bar_{model_name}.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"[SHAP_ADV] Saved bar plot -> {bar_path}")


def run_advanced_rf_block() -> None:
    """
    Block 2: Advanced FE, feature selection, tuned RF, best-F1 threshold, SHAP.
    """
    print("\n========== BLOCK 2: Advanced RF + Feature Selection + SHAP (500k) ==========\n")

    # 1) Load FE v1 and re-impute
    df_v1 = load_fe_v1()
    df_v1 = re_impute_numeric(df_v1)

    # 2) Advanced FE
    df_fe = build_advanced_features(df_v1)

    # 3) Splits
    y_all = df_fe[TARGET_COL].values
    splits = make_consistent_splits(len(df_fe), y_all)
    idx_train = splits["train"]
    idx_val = splits["val"]
    idx_test = splits["test"]

    # Optional subsample
    if MAX_TRAIN_ROWS_ADV and len(idx_train) > MAX_TRAIN_ROWS_ADV:
        rng = np.random.RandomState(RANDOM_STATE)
        idx_train = rng.choice(idx_train, size=MAX_TRAIN_ROWS_ADV, replace=False)
        print(f"[TRAIN_ADV] Subsampled train to {len(idx_train)} rows.")

    # 4) Prepare X, y
    X_all_raw = df_fe.drop(columns=[TARGET_COL])
    X_all = clean_X_for_model(X_all_raw)

    X_train = X_all.iloc[idx_train]
    y_train = y_all[idx_train]
    X_val = X_all.iloc[idx_val]
    y_val = y_all[idx_val]
    X_test = X_all.iloc[idx_test]
    y_test = y_all[idx_test]

    # 5) Feature selection (train only)
    selected_features = select_features_by_auc(X_train, y_train)
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]
    print(f"[INFO_ADV] Final feature count for RF: {X_train.shape[1]}")

    # 6) Train RF
    rf = train_rf_advanced(X_train, y_train)
    model_path = OUTPUT_DIR_RF_ONLY / "rf_500k_advanced.pkl"
    joblib.dump(rf, model_path)
    print(f"[SAVE_ADV] RF model -> {model_path}")

    # 7) Evaluation
    # 7.1 metrics at thr 0.5
    metrics_val_05 = evaluate_at_threshold(rf, X_val, y_val, thr=0.5)
    metrics_test_05 = evaluate_at_threshold(rf, X_test, y_test, thr=0.5)

    # 7.2 best F1 threshold on validation
    best_thr, best_f1 = find_best_threshold_for_f1(rf, X_val, y_val)
    metrics_val_best = evaluate_at_threshold(rf, X_val, y_val, thr=best_thr)
    metrics_test_best = evaluate_at_threshold(rf, X_test, y_test, thr=best_thr)

    print("\n[VAL @ 0.5]", metrics_val_05)
    print("[TEST @ 0.5]", metrics_test_05)
    print(f"\n[VAL best F1] thr={best_thr:.3f}", metrics_val_best)
    print("[TEST best F1]", metrics_test_best)

    # Save metrics
    rows = []
    rows.append({"split": "val", "threshold": 0.5, **metrics_val_05})
    rows.append({"split": "test", "threshold": 0.5, **metrics_test_05})
    rows.append({"split": "val", "threshold": best_thr, **metrics_val_best})
    rows.append({"split": "test", "threshold": best_thr, **metrics_test_best})

    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR_RF_ONLY / "rf_500k_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[SAVE_ADV] Metrics -> {metrics_path}")

    # Save best threshold
    with open(OUTPUT_DIR_RF_ONLY / "rf_500k_best_threshold.txt", "w") as f:
        f.write(f"best_thr_val_f1={best_thr:.4f}\n")
        f.write(f"best_f1_val={best_f1:.4f}\n")

    # 8) SHAP analysis on validation (currently disabled by RUN_SHAP_ADV = False)
    run_shap_analysis_advanced(rf, X_val, model_name="rf_500k_advanced")

    print("\n[BLOCK 2 DONE] Advanced RF + feature selection + SHAP completed.\n")


# ================== MAIN ==================

def run_pipeline():
    # Block 1: FE v1/v2/v3 + RF comparison + SHAP
    run_multi_version_block()

    # Block 2: Advanced RF with feature selection + best-F1 threshold + SHAP
    run_advanced_rf_block()

    print("\n[ALL DONE] final_for_500k pipeline finished.")


if __name__ == "__main__":
    run_pipeline()
