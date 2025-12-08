"""
Home Credit - Medium Dataset RF Pipeline (Day 3–4 style, RF only, NO SHAP)

- Reads all Home Credit tables:
  application_train, bureau, previous_application, POS_CASH_balance,
  credit_card_balance, installments_payments
- Builds:
  * application-level base features
  * aggregated bureau / previous / POS / CC / installments features
- Merges into a single matrix keyed by SK_ID_CURR
- Runs univariate AUC feature selection (auc_sym >= 0.52)
- Trains a RandomForestClassifier
- Evaluates at thr=0.5 and best-F1 threshold
- Saves metrics, feature importance, and RF model

Output folder: outputs_hc_day5_rf/
"""

from pathlib import Path
from typing import List, Dict, Tuple

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import joblib


# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = Path("home-credit-default-risk")
OUTPUT_DIR = Path("outputs_hc_day5_rf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

APP_TRAIN_PATH = DATA_DIR / "application_train.csv"
BUREAU_PATH = DATA_DIR / "bureau.csv"
BUREAU_BAL_PATH = DATA_DIR / "bureau_balance.csv"
PREV_APP_PATH = DATA_DIR / "previous_application.csv"
POS_PATH = DATA_DIR / "POS_CASH_balance.csv"
CC_PATH = DATA_DIR / "credit_card_balance.csv"
INSTAL_PATH = DATA_DIR / "installments_payments.csv"

TARGET_COL = "TARGET"
SK_ID_COL = "SK_ID_CURR"
RANDOM_STATE = 42

AUC_SYM_THRESHOLD = 0.52  # for feature selection


# =============================================================================
# UTILS
# =============================================================================

def report_memory(df: pd.DataFrame, label: str) -> None:
    mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"[MEM] {label}: {mb:.2f} MB")


def safe_read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Wrapper for pd.read_csv with nice error message."""
    return pd.read_csv(path, nrows=nrows)


# =============================================================================
# APPLICATION-LEVEL FEATURES
# =============================================================================

def fe_application_base() -> pd.DataFrame:
    """
    Build base features on application_train:
    - Encode categoricals
    - Simple ratios & transformations
    """
    print(f"[APP] Loading {APP_TRAIN_PATH} ...")
    app = pd.read_csv(APP_TRAIN_PATH)
    print(f"[APP] Shape: {app.shape}")
    report_memory(app, "application_train_raw")

    # Basic sanity: ensure TARGET exists
    if TARGET_COL not in app.columns:
        raise ValueError(f"{TARGET_COL} not found in application_train.csv")

    # Age & employment
    if "DAYS_BIRTH" in app.columns:
        app["AGE_YEARS"] = (-app["DAYS_BIRTH"] / 365.25).astype("float32")
    if "DAYS_EMPLOYED" in app.columns:
        app["EMPLOYED_YEARS"] = np.where(
            app["DAYS_EMPLOYED"] < 0,
            -app["DAYS_EMPLOYED"] / 365.25,
            np.nan,
        ).astype("float32")

    # External sources combinations
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        if c not in app.columns:
            app[c] = np.nan

    app["EXT_SOURCES_MEAN"] = app[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    app["EXT_SOURCES_MIN"] = app[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
    app["EXT_SOURCES_MAX"] = app[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(axis=1)

    # Main financial ratios
    def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        return num / np.where(den.abs() > 0, den, np.nan)

    if "AMT_CREDIT" in app.columns and "AMT_INCOME_TOTAL" in app.columns:
        app["CREDIT_INCOME_RATIO"] = safe_div(app["AMT_CREDIT"], app["AMT_INCOME_TOTAL"])
    if "AMT_ANNUITY" in app.columns and "AMT_INCOME_TOTAL" in app.columns:
        app["ANNUITY_INCOME_RATIO"] = safe_div(app["AMT_ANNUITY"], app["AMT_INCOME_TOTAL"])
    if "AMT_CREDIT" in app.columns and "AMT_ANNUITY" in app.columns:
        app["CREDIT_ANNUITY_RATIO"] = safe_div(app["AMT_CREDIT"], app["AMT_ANNUITY"])
    if "AMT_CREDIT" in app.columns and "AMT_GOODS_PRICE" in app.columns:
        app["CREDIT_GOODS_RATIO"] = safe_div(app["AMT_CREDIT"], app["AMT_GOODS_PRICE"])

    # Document count
    doc_cols = [c for c in app.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        app["DOC_COUNT"] = app[doc_cols].sum(axis=1)

    # Encode object columns as int codes
    obj_cols = app.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        codes, _ = pd.factorize(app[col])
        app[col] = codes.astype("int32")

    # Downcast numeric
    for col in app.select_dtypes(include=["float64"]).columns:
        app[col] = pd.to_numeric(app[col], downcast="float")
    for col in app.select_dtypes(include=["int64"]).columns:
        app[col] = pd.to_numeric(app[col], downcast="integer")

    report_memory(app, "application_base_fe")
    return app


# =============================================================================
# AGGREGATIONS
# =============================================================================

def agg_bureau() -> pd.DataFrame:
    print(f"[BUREAU] Loading {BUREAU_PATH} ...")
    df = pd.read_csv(BUREAU_PATH)
    report_memory(df, "bureau_raw")

    # Encode categoricals if any
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        codes, _ = pd.factorize(df[col])
        df[col] = codes.astype("int32")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for drop_col in ["SK_ID_CURR", "SK_ID_BUREAU"]:
        if drop_col in num_cols:
            num_cols.remove(drop_col)

    agg = df.groupby("SK_ID_CURR")[num_cols].agg(["mean", "max", "min", "sum"])
    agg.columns = [f"BUREAU_{c[0]}_{c[1].upper()}" for c in agg.columns.to_flat_index()]
    agg.reset_index(inplace=True)

    report_memory(agg, "bureau_agg")
    return agg


def agg_prev_app() -> pd.DataFrame:
    print(f"[PREV] Loading {PREV_APP_PATH} ...")
    df = pd.read_csv(PREV_APP_PATH)
    report_memory(df, "previous_raw")

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        codes, _ = pd.factorize(df[col])
        df[col] = codes.astype("int32")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for drop_col in ["SK_ID_CURR", "SK_ID_PREV"]:
        if drop_col in num_cols:
            num_cols.remove(drop_col)

    agg = df.groupby("SK_ID_CURR")[num_cols].agg(["mean", "max", "min", "sum"])
    agg.columns = [f"PREV_{c[0]}_{c[1].upper()}" for c in agg.columns.to_flat_index()]
    agg.reset_index(inplace=True)

    report_memory(agg, "previous_agg")
    return agg


def agg_pos_cash() -> pd.DataFrame:
    print(f"[POS] Loading {POS_PATH} ...")
    df = pd.read_csv(POS_PATH)
    report_memory(df, "pos_raw")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for drop_col in ["SK_ID_CURR", "SK_ID_PREV"]:
        if drop_col in num_cols:
            num_cols.remove(drop_col)

    agg = df.groupby("SK_ID_CURR")[num_cols].agg(["mean", "max", "sum"])
    agg.columns = [f"POS_{c[0]}_{c[1].upper()}" for c in agg.columns.to_flat_index()]
    agg.reset_index(inplace=True)

    report_memory(agg, "pos_agg")
    return agg


def agg_credit_card() -> pd.DataFrame:
    print(f"[CC] Loading {CC_PATH} ...")
    df = pd.read_csv(CC_PATH)
    report_memory(df, "cc_raw")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for drop_col in ["SK_ID_CURR", "SK_ID_PREV"]:
        if drop_col in num_cols:
            num_cols.remove(drop_col)

    agg = df.groupby("SK_ID_CURR")[num_cols].agg(["mean", "max", "sum"])
    agg.columns = [f"CC_{c[0]}_{c[1].upper()}" for c in agg.columns.to_flat_index()]
    agg.reset_index(inplace=True)

    report_memory(agg, "cc_agg")
    return agg


def agg_installments() -> pd.DataFrame:
    print(f"[INSTAL] Loading {INSTAL_PATH} ...")
    df = pd.read_csv(INSTAL_PATH)
    report_memory(df, "instal_raw")

    # Custom features
    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(df.columns):
        df["DAYS_PAYMENT_DIFF"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    else:
        df["DAYS_PAYMENT_DIFF"] = np.nan

    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(df.columns):
        df["PAYMENT_RATIO"] = df["AMT_PAYMENT"] / np.where(df["AMT_INSTALMENT"].abs() > 0,
                                                           df["AMT_INSTALMENT"], np.nan)
    else:
        df["PAYMENT_RATIO"] = np.nan

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for drop_col in ["SK_ID_CURR", "SK_ID_PREV"]:
        if drop_col in num_cols:
            num_cols.remove(drop_col)

    agg = df.groupby("SK_ID_CURR")[num_cols].agg(["mean", "max", "min"])
    agg.columns = [f"INST_{c[0]}_{c[1].upper()}" for c in agg.columns.to_flat_index()]
    agg.reset_index(inplace=True)

    report_memory(agg, "instal_agg")
    return agg


# =============================================================================
# BUILD FULL MATRIX
# =============================================================================

def build_full_matrix() -> pd.DataFrame:
    app = fe_application_base()
    bureau_agg = agg_bureau()
    prev_agg = agg_prev_app()
    pos_agg = agg_pos_cash()
    cc_agg = agg_credit_card()
    instal_agg = agg_installments()

    print("[MERGE] Joining aggregated tables to application FE ...")
    df = app.merge(bureau_agg, on=SK_ID_COL, how="left")
    df = df.merge(prev_agg, on=SK_ID_COL, how="left")
    df = df.merge(pos_agg, on=SK_ID_COL, how="left")
    df = df.merge(cc_agg, on=SK_ID_COL, how="left")
    df = df.merge(instal_agg, on=SK_ID_COL, how="left")

    report_memory(df, "merged_all")

    # Keep only numeric columns (TARGET & SK_ID stay as numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_num = df[numeric_cols].copy()

    # Simple imputation: fill NaNs with median per column
    for col in df_num.columns:
        if df_num[col].isna().any():
            df_num[col] = df_num[col].fillna(df_num[col].median())

    report_memory(df_num, "numeric_cleaned")
    return df_num


# =============================================================================
# UNIVARIATE AUC & FEATURE SELECTION
# =============================================================================

def compute_univariate_auc(df: pd.DataFrame) -> pd.DataFrame:
    print("[AUC1D] Computing univariate AUC per feature ...")
    y = df[TARGET_COL].values
    results: List[Dict] = []

    drop_cols = {TARGET_COL, SK_ID_COL}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    for col in feature_cols:
        x = df[col].values
        if np.unique(x).size <= 1:
            continue
        try:
            auc = roc_auc_score(y, x)
            auc_sym = max(auc, 1.0 - auc)
            results.append({"feature": col, "auc": auc, "auc_sym": auc_sym})
        except Exception:
            continue

    auc_df = pd.DataFrame(results).sort_values("auc_sym", ascending=False)
    auc_path = OUTPUT_DIR / "rf_hc_univariate_auc.csv"
    auc_df.to_csv(auc_path, index=False)
    print(f"[AUC1D] Saved univariate AUC -> {auc_path}")
    return auc_df


def select_features(auc_df: pd.DataFrame, threshold: float = AUC_SYM_THRESHOLD) -> List[str]:
    mask = auc_df["auc_sym"] >= threshold
    selected = auc_df.loc[mask, "feature"].tolist()
    print(f"[FEAT_SELECT] Selected {len(selected)} features with auc_sym >= {threshold}")
    return selected


# =============================================================================
# SPLIT, TRAIN, EVAL
# =============================================================================

def split_data(df: pd.DataFrame, feature_cols: List[str]) -> Tuple:
    y = df[TARGET_COL].values
    X = df[feature_cols].values

    idx = np.arange(df.shape[0])
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"[SPLIT] train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")

    X_train = X[idx_train]
    X_val = X[idx_val]
    X_test = X[idx_test]

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_rf(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    print(f"[TRAIN] X_train={X_train.shape}")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_leaf=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    print("[RF] Training RandomForest ...")
    rf.fit(X_train, y_train)
    return rf


def eval_at_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    thr: float
) -> Dict[str, float]:
    preds = (proba >= thr).astype(int)
    auc = roc_auc_score(y_true, proba)
    ll = log_loss(y_true, proba)
    f1 = f1_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    return {
        "threshold": float(thr),
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


def evaluate_model_full(
    rf: RandomForestClassifier,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    # Probabilities
    proba_val = rf.predict_proba(X_val)[:, 1]
    proba_test = rf.predict_proba(X_test)[:, 1]

    # 0.5 threshold
    metrics_val_05 = eval_at_threshold(y_val, proba_val, 0.5)
    metrics_test_05 = eval_at_threshold(y_test, proba_test, 0.5)
    print(f"[VAL @0.5] {metrics_val_05}")
    print(f"[TEST @0.5] {metrics_test_05}")

    # Search best F1 threshold on validation
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.1, 0.9, 81):
        m = eval_at_threshold(y_val, proba_val, thr)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = thr

    print(f"[THR] Best F1 on val at thr={best_thr:.3f}: {best_f1:.4f}")

    metrics_val_best = eval_at_threshold(y_val, proba_val, best_thr)
    metrics_test_best = eval_at_threshold(y_test, proba_test, best_thr)

    print(f"[VAL best F1] thr={best_thr:.3f} {metrics_val_best}")
    print(f"[TEST best F1] {metrics_test_best}")

    # Build metrics DataFrame
    rows = []
    rows.append({"eval_set": "val@0.5", **metrics_val_05})
    rows.append({"eval_set": "test@0.5", **metrics_test_05})
    rows.append({"eval_set": "val@bestF1", **metrics_val_best})
    rows.append({"eval_set": "test@bestF1", **metrics_test_best})

    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "rf_hc_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[SAVE] Metrics -> {metrics_path}")
    print(metrics_df)

    return metrics_df


def save_feature_importance(
    rf: RandomForestClassifier,
    feature_cols: List[str]
) -> None:
    importances = rf.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    fi_path = OUTPUT_DIR / "rf_hc_feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    print(f"[SAVE] Feature importance -> {fi_path}")
    print(fi.head(20))


# =============================================================================
# MAIN
# =============================================================================

def main():
    df_full = build_full_matrix()

    # Univariate AUC and feature selection
    auc_df = compute_univariate_auc(df_full)
    selected_features = select_features(auc_df, threshold=AUC_SYM_THRESHOLD)

    # Final matrix: keep SK_ID and TARGET + selected features
    keep_cols = [SK_ID_COL, TARGET_COL] + selected_features
    df_final = df_full[keep_cols].copy()
    print(f"[INFO] Final feature matrix shape (with target + SK_ID): {df_final.shape}")

    # Train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_final, selected_features)
    print(f"[TRAIN] X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    # Train RF
    rf = train_rf(X_train, y_train)

    # Evaluate
    metrics_df = evaluate_model_full(rf, X_train, X_val, X_test, y_train, y_val, y_test)

    # Save model and feature importance
    model_path = OUTPUT_DIR / "rf_hc_model.pkl"
    joblib.dump(rf, model_path)
    print(f"[SAVE] RF model -> {model_path}")

    save_feature_importance(rf, selected_features)


if __name__ == "__main__":
    main()
