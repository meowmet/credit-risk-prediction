"""
Day 3–4 – Advanced FE + RF + SHAP for 500k LendingClub dataset

Assumes you already created:
    credit_500k_fe_v1.csv
via the previous FE script.

This script:
  - Loads FE v1
  - Builds FE v2 and FE v3 (extra engineered features)
  - Uses one consistent train/val/test split
  - Trains RandomForest on v1, v2, v3
  - Compares metrics and saves rf_500k_comparison.csv
  - Computes univariate AUC for v3 and saves univariate_auc_500k_v3.csv
  - Runs SHAP on the best RF model and saves SHAP plots
"""

from pathlib import Path
from typing import Dict, Tuple, List

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
)

import joblib

try:
    import shap
    import matplotlib.pyplot as plt
except ImportError:
    shap = None
    plt = None
    print("[WARN] shap/matplotlib not installed. SHAP plots will be skipped.")


# ================== CONFIG ==================

FE_V1_PATH = Path("credit_500k_fe_v1.csv")
OUTPUT_DIR = Path("outputs_500k_day3_4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target_default_500k"
RANDOM_STATE = 42

# To keep RF training manageable, you can cap train size
MAX_TRAIN_ROWS = 300_000   # set to None to use all train rows
SHAP_SAMPLE_SIZE = 5_000   # rows for SHAP analysis


# ================== HELPERS ==================

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


def re_impute_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace inf/-inf with NaN, then fill numeric NaNs with median and downcast.
    Used inside FE-building steps (v2/v3).
    """
    df = df.copy()

    # Turn +/- inf into NaN so they get imputed
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL)

    # Fill numeric NaNs with median
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Downcast for memory
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def safe_add_feature(df: pd.DataFrame, name: str, func) -> None:
    """Add a derived feature if all required columns exist."""
    try:
        df[name] = func(df)
    except KeyError:
        print(f"[FE] Skipping feature {name} (missing columns).")


# ================== FE V2 / V3 ==================

def build_fe_v2(df_v1: pd.DataFrame) -> pd.DataFrame:
    """
    FE v2: start from FE v1, add:
      - log transforms
      - extra ratios and spreads
    """
    print("[FE] Building FE v2 from FE v1 ...")
    df = df_v1.copy()

    # Log transforms (with +1 to avoid log(0))
    safe_add_feature(df, "log_loan_amnt",
                     lambda d: np.log1p(d["loan_amnt"]))
    safe_add_feature(df, "log_annual_inc",
                     lambda d: np.log1p(d["annual_inc"]))
    safe_add_feature(df, "log_revol_bal",
                     lambda d: np.log1p(d["revol_bal"]))
    safe_add_feature(df, "log_total_rev_hi_lim",
                     lambda d: np.log1p(d["total_rev_hi_lim"]))

    # FICO mean and spread at origination
    safe_add_feature(
        df, "fico_mean_v2",
        lambda d: 0.5 * (d["fico_range_low"] + d["fico_range_high"])
    )
    safe_add_feature(
        df, "fico_spread",
        lambda d: d["fico_range_high"] - d["fico_range_low"]
    )

    # Debt/utilization style ratios
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

    # Account structure ratios
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

    # Term * rate interaction (even if term is encoded, it’s monotone)
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

    # loan_to_income from FE v1
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


# ================== SPLITS ==================

def make_consistent_splits(n_rows: int, y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create consistent train/val/test indices using only y and a global RNG,
    so we can re-use the same indices for v1, v2, v3.
    """
    all_idx = np.arange(n_rows)
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        all_idx,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    print("[SPLIT] Sizes:",
          "train", len(idx_train),
          "val", len(idx_val),
          "test", len(idx_test))

    return {
        "train": idx_train,
        "val": idx_val,
        "test": idx_test,
    }


# ================== CLEAN X FOR MODEL ==================

def clean_X_for_model(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all feature values are finite and within a reasonable numeric range.

    - Replace ±inf with NaN
    - Fill NaN with median (per column)
    - Clip to [-1e9, 1e9]
    - Cast numeric columns to float64
    """
    X = X.copy()

    # Replace infinities with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    num_cols = X.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        col_data = X[col]

        # Fill NaNs with median
        if col_data.isna().any():
            median_val = col_data.median()
            col_data = col_data.fillna(median_val)

        # Clip extreme values to avoid overflow
        col_data = np.clip(col_data, -1e9, 1e9)

        X[col] = col_data.astype("float64")

    return X


# ================== RF TRAINING / EVAL ==================

def train_rf_for_version(
    fe_name: str,
    df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    max_train_rows: int = MAX_TRAIN_ROWS,
) -> Tuple[RandomForestClassifier, Dict[str, Dict[str, float]]]:
    """
    Train a RandomForest on given FE version and return:
      - trained model
      - metrics dict for val/test
    """
    print(f"\n[RF] Training RF for {fe_name} ...")

    # Separate features and target
    y_all = df[TARGET_COL].values
    X_all_raw = df.drop(columns=[TARGET_COL])

    # Clean X to remove inf/NaN and clip extremes
    X_all = clean_X_for_model(X_all_raw)

    idx_train = splits["train"]
    idx_val = splits["val"]
    idx_test = splits["test"]

    # Optional subsampling for training
    if (max_train_rows is not None) and (len(idx_train) > max_train_rows):
        rng = np.random.RandomState(RANDOM_STATE)
        idx_train_sub = rng.choice(idx_train, size=max_train_rows, replace=False)
        print(f"[RF] Subsampling train from {len(idx_train)} to {len(idx_train_sub)} rows.")
        idx_train = idx_train_sub

    X_train = X_all.iloc[idx_train]
    y_train = y_all[idx_train]

    X_val = X_all.iloc[idx_val]
    y_val = y_all[idx_val]

    X_test = X_all.iloc[idx_test]
    y_test = y_all[idx_test]

    # RF configuration
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

    # Save model
    model_path = OUTPUT_DIR / f"rf_500k_{fe_name}.pkl"
    joblib.dump(rf, model_path)
    print(f"[RF] Saved model for {fe_name} -> {model_path}")

    return rf, metrics


# ================== UNIVARIATE AUC ==================

def compute_univariate_auc(
    df: pd.DataFrame,
    target_col: str,
    out_path: Path,
) -> None:
    """
    Compute univariate AUC for each feature vs target on the full dataset.
    We:
      - Clean features with clean_X_for_model
      - Use max(AUC, 1 - AUC) as symmetric score
    """
    print("[AUC1D] Computing univariate AUC for all features ...")

    y = df[target_col].values

    # Clean features (no target)
    X_raw = df.drop(columns=[target_col])
    X = clean_X_for_model(X_raw)

    results: List[Dict[str, float]] = []

    for col in X.columns:
        x = X[col].values

        # skip constant features
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


# ================== SHAP ==================

def run_shap_analysis(
    fe_name: str,
    model: RandomForestClassifier,
    df: pd.DataFrame,
    splits: Dict[str, np.ndarray],
    sample_size: int = SHAP_SAMPLE_SIZE,
) -> None:
    if shap is None or plt is None:
        print("[SHAP] shap/matplotlib not available; skipping SHAP.")
        return

    print(f"[SHAP] Running SHAP for {fe_name} ...")

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

    print(f"[SHAP] Using {X_sample.shape[0]} rows, {X_sample.shape[1]} features for SHAP.")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Class 1 (default) SHAP summary (beeswarm)
    shap.summary_plot(
        shap_values[1],
        X_sample,
        show=False,
        max_display=25,
    )
    shap_path_beeswarm = OUTPUT_DIR / f"shap_summary_beeswarm_{fe_name}.png"
    plt.tight_layout()
    plt.savefig(shap_path_beeswarm, dpi=200)
    plt.close()
    print(f"[SHAP] Saved beeswarm summary -> {shap_path_beeswarm}")

    # Bar plot of mean |SHAP|
    shap.summary_plot(
        shap_values[1],
        X_sample,
        show=False,
        plot_type="bar",
        max_display=25,
    )
    shap_path_bar = OUTPUT_DIR / f"shap_summary_bar_{fe_name}.png"
    plt.tight_layout()
    plt.savefig(shap_path_bar, dpi=200)
    plt.close()
    print(f"[SHAP] Saved bar summary -> {shap_path_bar}")


# ================== MAIN PIPELINE ==================

def run_pipeline():
    # ---- Load FE v1 and sanitize numeric ----
    df_v1 = load_fe_v1()
    df_v1 = re_impute_numeric(df_v1)

    # ---- Build FE v2 and v3 ----
    df_v2 = build_fe_v2(df_v1)
    df_v3 = build_fe_v3(df_v2)

    # ---- Consistent splits ----
    y = df_v1[TARGET_COL].values
    splits = make_consistent_splits(len(df_v1), y)

    # ---- Train RF for each FE version ----
    metrics_all: Dict[str, Dict[str, Dict[str, float]]] = {}
    models: Dict[str, RandomForestClassifier] = {}

    for fe_name, df in [("fe_v1", df_v1), ("fe_v2", df_v2), ("fe_v3", df_v3)]:
        rf, metrics = train_rf_for_version(fe_name, df, splits)
        metrics_all[fe_name] = metrics
        models[fe_name] = rf

    # ---- Save comparison table (using val metrics) ----
    rows = []
    for fe_name, m in metrics_all.items():
        row = {"fe_version": fe_name}
        for split_name in ["val", "test"]:
            for k, v in m[split_name].items():
                row[f"{split_name}_{k}"] = v
        rows.append(row)

    cmp_df = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    cmp_path = OUTPUT_DIR / "rf_500k_comparison.csv"
    cmp_df.to_csv(cmp_path, index=False)
    print(f"\n[RESULT] Saved RF comparison -> {cmp_path}")
    print(cmp_df)

    # ---- Univariate AUC on v3 ----
    auc_path = OUTPUT_DIR / "univariate_auc_500k_v3.csv"
    compute_univariate_auc(df_v3, TARGET_COL, auc_path)

    # ---- SHAP on best version ----
    best_fe_name = cmp_df.iloc[0]["fe_version"]
    print(f"\n[RESULT] Best FE version by val AUC: {best_fe_name}")
    best_model = models[best_fe_name]
    if best_fe_name == "fe_v1":
        best_df = df_v1
    elif best_fe_name == "fe_v2":
        best_df = df_v2
    else:
        best_df = df_v3

    run_shap_analysis(best_fe_name, best_model, best_df, splits)

    print("\n[DONE] Day 3–4 advanced FE + RF + SHAP pipeline finished.")


if __name__ == "__main__":
    run_pipeline()
