"""
Random Forest – Advanced FE on 500k LendingClub dataset (Day 3–4 final)

Assumes:
    credit_500k_fe_v1.csv
exists and already has:
    - leakage columns removed
    - target column: target_default_500k

Pipeline:
  1. Load FE v1
  2. Advanced feature engineering (logs, ratios, interactions, bands, flags)
  3. Train/val/test split (stratified)
  4. Clean features (NaN/inf handling, clipping)
  5. Univariate AUC-based feature selection (on train only)
  6. Train tuned RandomForest
  7. Evaluate on val/test (AUC, F1, precision, recall, best-F1 threshold)
  8. Optional SHAP analysis on a small validation sample

Outputs:
  - outputs_500k_rf_only/rf_500k_metrics.csv
  - outputs_500k_rf_only/rf_500k_best_threshold.txt
  - (optional) SHAP plots in outputs_500k_rf_only/
"""

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

try:
    import shap
    import matplotlib.pyplot as plt
except ImportError:
    shap = None
    plt = None
    print("[WARN] shap/matplotlib not installed. SHAP plots will be skipped.")


# ================== CONFIG ==================

FE_V1_PATH = Path("credit_500k_fe_v1.csv")
OUTPUT_DIR = Path("outputs_500k_rf_only")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target_default_500k"
RANDOM_STATE = 42

# Subsample the train set (for speed); set None to use all train rows
MAX_TRAIN_ROWS = 300_000

# Univariate AUC threshold for feature selection (symmetric AUC)
FEATURE_SELECTION_THRESHOLD = 0.51

# SHAP configuration
RUN_SHAP = True
SHAP_SAMPLE_SIZE = 1_000


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


# ================== ADVANCED FEATURE ENGINEERING ==================

def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced FE designed for RandomForest (no leakage):
      - logs: loan_amnt, annual_inc, revol_bal, total_rev_hi_lim
      - FICO features: mean, spread, squared, bands
      - ratios: loan_to_income, installment_to_income, loan_to_total_rev,
                revol_util_computed, open_to_total_acc, delinq_to_total_acc
      - interactions: int_rate_x_dti, int_rate_x_loan_amnt, term_x_int_rate
      - polynomial: dti_sq
      - risk flags: high_dti_flag, delinq_flag, pub_rec_flag, risk_score
      - bands: dti_band, loan_to_income_band
    """
    print("[FE] Building advanced features ...")
    df = df.copy()

    # ---- Log transforms ----
    for col in ["loan_amnt", "annual_inc", "revol_bal", "total_rev_hi_lim"]:
        if col in df.columns:
            safe_add_feature(df, f"log_{col}", lambda d, c=col: np.log1p(d[c]))

    # ---- FICO features ----
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_mean"] = 0.5 * (df["fico_range_low"] + df["fico_range_high"])
        df["fico_spread"] = df["fico_range_high"] - df["fico_range_low"]
        safe_add_feature(df, "fico_mean_sq", lambda d: d["fico_mean"] ** 2)

        # FICO bands
        try:
            df["fico_band"] = pd.cut(
                df["fico_mean"],
                bins=[0, 640, 680, 720, 760, 900],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True,
            ).astype("float").fillna(2)
        except Exception as e:
            print(f"[FE] Skipping fico_band: {e}")

    # ---- Ratios ----
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

    # ---- Polynomial ----
    if "dti" in df.columns:
        df["dti_sq"] = df["dti"] ** 2

    # ---- Interactions ----
    safe_add_feature(df, "int_rate_x_dti", lambda d: d["int_rate"] * d["dti"])
    safe_add_feature(df, "int_rate_x_loan_amnt", lambda d: d["int_rate"] * d["loan_amnt"])
    safe_add_feature(df, "term_x_int_rate", lambda d: d["term"] * d["int_rate"])

    # ---- Bands ----
    if "dti" in df.columns:
        try:
            df["dti_band"] = pd.cut(
                df["dti"],
                bins=[0, 10, 20, 30, 40, 1000],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True,
            ).astype("float").fillna(2)
        except Exception as e:
            print(f"[FE] Skipping dti_band: {e}")

    if "loan_to_income" in df.columns:
        try:
            df["loan_to_income_band"] = pd.cut(
                df["loan_to_income"],
                bins=[0, 0.1, 0.2, 0.3, 0.5, 100],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True,
            ).astype("float").fillna(2)
        except Exception as e:
            print(f"[FE] Skipping loan_to_income_band: {e}")

    # ---- Risk flags ----
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
    print(f"[FE] Advanced features built: {df.shape[1]} columns.")
    return df


# ================== SPLITS ==================

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


# ================== CLEAN X ==================

def clean_X_for_model(X: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all features are finite and within reasonable range.
      - Replace ±inf with NaN
      - Fill NaN with median
      - Clip to [-1e9, 1e9]
      - Cast numeric columns to float64
    """
    X = X.copy()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        col_data = X[col]
        if col_data.isna().any():
            col_data = col_data.fillna(col_data.median())
        col_data = np.clip(col_data, -1e9, 1e9)
        X[col] = col_data.astype("float64")

    return X


# ================== FEATURE SELECTION ==================

def select_features_by_auc(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    threshold: float = FEATURE_SELECTION_THRESHOLD,
) -> List[str]:
    """
    Select features using univariate AUC on the train set.
    Keep features where max(AUC, 1-AUC) >= threshold.

    This keeps both positively and negatively correlated features.
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


# ================== RF MODEL ==================

def train_rf_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> RandomForestClassifier:
    """Train a tuned RandomForest classifier."""
    print("[RF] Training RandomForest classifier ...")

    # Slightly higher weight on positives (defaults)
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


# ================== EVALUATION ==================

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


# ================== SHAP ==================

def run_shap_analysis(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    model_name: str = "rf_500k",
) -> None:
    if not RUN_SHAP:
        print("[SHAP] RUN_SHAP is False; skipping SHAP.")
        return

    if shap is None or plt is None:
        print("[SHAP] shap/matplotlib not available; skipping SHAP.")
        return

    print("[SHAP] Running SHAP analysis ...")

    if len(X_val) > SHAP_SAMPLE_SIZE:
        rng = np.random.RandomState(RANDOM_STATE)
        sample_idx = rng.choice(X_val.index, size=SHAP_SAMPLE_SIZE, replace=False)
        X_sample = X_val.loc[sample_idx]
    else:
        X_sample = X_val

    print(f"[SHAP] Using {X_sample.shape[0]} rows, {X_sample.shape[1]} features.")

    explainer = shap.TreeExplainer(model)
    shap_values_list = explainer.shap_values(X_sample, check_additivity=False)

    # RandomForest: shap_values is list [class0, class1]
    shap_vals = shap_values_list[1]

    # Beeswarm
    shap.summary_plot(
        shap_vals,
        X_sample,
        show=False,
        max_display=25,
    )
    beeswarm_path = OUTPUT_DIR / f"shap_beeswarm_{model_name}.png"
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=200)
    plt.close()
    print(f"[SHAP] Saved beeswarm plot -> {beeswarm_path}")

    # Bar plot
    shap.summary_plot(
        shap_vals,
        X_sample,
        show=False,
        plot_type="bar",
        max_display=25,
    )
    bar_path = OUTPUT_DIR / f"shap_bar_{model_name}.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"[SHAP] Saved bar plot -> {bar_path}")


# ================== MAIN ==================

def run_pipeline():
    # 1) Load FE v1
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

    # Optional subsample on train
    if MAX_TRAIN_ROWS and len(idx_train) > MAX_TRAIN_ROWS:
        rng = np.random.RandomState(RANDOM_STATE)
        idx_train = rng.choice(idx_train, size=MAX_TRAIN_ROWS, replace=False)
        print(f"[TRAIN] Subsampled train to {len(idx_train)} rows.")

    # 4) Prepare X,y
    X_all_raw = df_fe.drop(columns=[TARGET_COL])
    X_all = clean_X_for_model(X_all_raw)

    X_train = X_all.iloc[idx_train]
    y_train = y_all[idx_train]

    X_val = X_all.iloc[idx_val]
    y_val = y_all[idx_val]

    X_test = X_all.iloc[idx_test]
    y_test = y_all[idx_test]

    # 5) Feature selection (on train only)
    selected_features = select_features_by_auc(X_train, y_train)
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

    print(f"[INFO] Final feature count for RF: {X_train.shape[1]}")

    # 6) Train RF
    rf = train_rf_model(X_train, y_train)
    model_path = OUTPUT_DIR / "rf_500k_advanced.pkl"
    joblib.dump(rf, model_path)
    print(f"[SAVE] RF model -> {model_path}")

    # 7) Evaluation
    # 7.1 metrics at default threshold 0.5
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

    # Save metrics to CSV
    rows = []
    rows.append({"split": "val", "threshold": 0.5, **metrics_val_05})
    rows.append({"split": "test", "threshold": 0.5, **metrics_test_05})
    rows.append({"split": "val", "threshold": best_thr, **metrics_val_best})
    rows.append({"split": "test", "threshold": best_thr, **metrics_test_best})

    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "rf_500k_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n[SAVE] Metrics -> {metrics_path}")

    # Save best threshold
    with open(OUTPUT_DIR / "rf_500k_best_threshold.txt", "w") as f:
        f.write(f"best_thr_val_f1={best_thr:.4f}\nbest_f1_val={best_f1:.4f}\n")

    # 8) SHAP analysis on validation
    run_shap_analysis(rf, X_val, model_name="rf_500k_advanced")

    print("\n[DONE] RF-only advanced pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
