import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import joblib


# =========================
# PATH CONFIG
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent      # C:\Users\meowm\Desktop\Kaggle\RF project
PROJECT_DIR = SCRIPT_DIR

# Raw 30k credit data
RAW_TAIWAN_CANDIDATES = [
    PROJECT_DIR / "data" / "credit_30k.csv",      # preferred (your current location)
    PROJECT_DIR / "credit_30k.csv",               # fallback
]

# Outputs (Student A + B)
TAIWAN_OUTPUT_DIR = PROJECT_DIR / "outputs_taiwan_day1_day2"
TAIWAN_STH_DIR = TAIWAN_OUTPUT_DIR / "sth"

FEATURES_PATH = TAIWAN_STH_DIR / "taiwan_final_features.csv"
MODEL_PATH = TAIWAN_OUTPUT_DIR / "rf_taiwan_v3.pkl"
SHAP_DIR = TAIWAN_STH_DIR / "shap_taiwan_deep"
DEPENDENCE_DIR = SHAP_DIR / "dependence_plots"
METRICS_PATH = TAIWAN_OUTPUT_DIR / "rf_taiwan_v3_metrics.json"

# TRUE target column name in credit_30k.csv (after skipping mapping row)
TARGET_COLUMN_OVERRIDE = "default payment next month"

# SHAP config
SHAP_SAMPLE_SIZE = 800
SHAP_BG_SIZE = 200


# =========================
# UTILS
# =========================

def find_existing_path(candidates):
    for p in candidates:
        if p is not None and Path(p).is_file():
            return Path(p)
    raise FileNotFoundError(
        "Could not find the raw Taiwan CSV file. "
        "Please check credit_30k.csv location or update RAW_TAIWAN_CANDIDATES."
    )


def detect_target_column(df: pd.DataFrame) -> str:
    # Use explicit override (we know the real name)
    if TARGET_COLUMN_OVERRIDE is not None:
        if TARGET_COLUMN_OVERRIDE not in df.columns:
            raise ValueError(
                f"TARGET_COLUMN_OVERRIDE='{TARGET_COLUMN_OVERRIDE}' "
                f"is not in DataFrame columns: {list(df.columns)}"
            )
        print(f"[2/4] Using TARGET_COLUMN_OVERRIDE: {TARGET_COLUMN_OVERRIDE}")
        return TARGET_COLUMN_OVERRIDE

    raise ValueError(
        "TARGET_COLUMN_OVERRIDE is None; set it at the top of the script."
    )


# =========================
# 1) LOAD RAW DATA
# =========================

def load_raw_taiwan() -> pd.DataFrame:
    raw_path = find_existing_path(RAW_TAIWAN_CANDIDATES)
    print(f"[1/4] Loading raw Taiwan 30k data from: {raw_path}")

    # IMPORTANT: skip first row (mapping row), like your other scripts
    df = pd.read_csv(raw_path, skiprows=[0])
    print(f"[1/4] Raw shape after skiprows=[0]: {df.shape}")
    return df


# =========================
# 2) FEATURE ENGINEERING (final features for 30k)
# =========================

def build_taiwan_features(df_raw: pd.DataFrame):
    print("[2/4] Starting feature engineering for Taiwan 30k...")

    df = df_raw.copy()

    # Drop ID / index columns if present
    for col in ["ID", "Unnamed: 0"]:
        if col in df.columns:
            print(f"[2/4] Dropping ID/index column: {col}")
            df = df.drop(columns=[col])

    # Detect target column (we know it's "default payment next month")
    target_col = detect_target_column(df)
    print(f"[2/4] Detected target column: {target_col}")

    # Identify PAY_*, BILL_AMT*, PAY_AMT* columns (UCI Taiwan style)
    pay_cols = [c for c in df.columns if c.upper().startswith("PAY_")]
    bill_cols = [c for c in df.columns if c.upper().startswith("BILL_AMT")]
    pay_amt_cols = [c for c in df.columns if c.upper().startswith("PAY_AMT")]

    # Replace special negatives in PAY_* with 0 for averages
    if pay_cols:
        pay_status = df[pay_cols].replace({-2: 0, -1: 0})
        df["AVG_PAY_STATUS"] = pay_status.mean(axis=1)
        df["MAX_PAY_STATUS"] = pay_status.max(axis=1)
        df["OVERDUE_COUNT"] = (pay_status > 0).sum(axis=1)
        df["SEVERE_OVERDUE_COUNT"] = (pay_status >= 2).sum(axis=1)

    # Bill amount aggregates
    if bill_cols:
        df["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)
        df["MAX_BILL_AMT"] = df[bill_cols].max(axis=1)
        df["MIN_BILL_AMT"] = df[bill_cols].min(axis=1)
        df["BILL_AMT_TREND"] = df[bill_cols[-1]] - df[bill_cols[0]]

    # Payment amount aggregates
    if pay_amt_cols:
        df["AVG_PAY_AMT"] = df[pay_amt_cols].mean(axis=1)
        df["MAX_PAY_AMT"] = df[pay_amt_cols].max(axis=1)

    # Limit utilization & payment ratios
    if ("LIMIT_BAL" in df.columns) and ("AVG_BILL_AMT" in df.columns):
        df["LIMIT_UTILIZATION"] = df["AVG_BILL_AMT"] / (df["LIMIT_BAL"].abs() + 1.0)

    if ("AVG_PAY_AMT" in df.columns) and ("AVG_BILL_AMT" in df.columns):
        df["PAYMENT_RATIO"] = df["AVG_PAY_AMT"] / (df["AVG_BILL_AMT"].abs() + 1.0)

    # Fill remaining NaNs
    df = df.fillna(0)

    print(f"[2/4] Feature DataFrame shape: {df.shape}")
    print("[2/4] Target distribution:")
    print(df[target_col].value_counts(dropna=False))

    # Save features (including target) under Student A+B folder
    TAIWAN_STH_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"[2/4] Saved final features to: {FEATURES_PATH}")

    return df, target_col


# =========================
# 3) TRAIN RANDOM FOREST
# =========================

def train_rf_taiwan(df_features: pd.DataFrame, target_col: str):
    print("[3/4] Training RandomForest on Taiwan 30k features...")

    if target_col not in df_features.columns:
        raise ValueError(f"Target column '{target_col}' not found in features DataFrame.")

    y = df_features[target_col]
    X = df_features.drop(columns=[target_col])

    # Check class balance before using stratify
    class_counts = y.value_counts()
    print("[3/4] Class counts:")
    print(class_counts)

    if len(class_counts) < 2:
        raise ValueError("[3/4] Target has only one class; cannot train a classifier.")

    if class_counts.min() < 2:
        print(
            "[3/4] WARNING: A class has < 2 samples; "
            "disabling stratify in train_test_split to avoid errors."
        )
        stratify_arg = None
    else:
        stratify_arg = y

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=stratify_arg,
        random_state=42,
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=50,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    print("[3/4] Fitting RandomForest...")
    rf.fit(X_train, y_train)

    y_valid_proba = rf.predict_proba(X_valid)[:, 1]
    y_valid_pred = rf.predict(X_valid)

    auc = roc_auc_score(y_valid, y_valid_proba)
    f1 = f1_score(y_valid, y_valid_pred)
    acc = accuracy_score(y_valid, y_valid_pred)

    print(f"[3/4] Validation AUC: {auc:.4f}")
    print(f"[3/4] Validation F1 : {f1:.4f}")
    print(f"[3/4] Validation ACC: {acc:.4f}")

    # Save model
    TAIWAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODEL_PATH)
    print(f"[3/4] Saved model to: {MODEL_PATH}")

    # Save metrics
    metrics = {"auc": float(auc), "f1": float(f1), "accuracy": float(acc)}
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[3/4] Saved metrics to: {METRICS_PATH}")

    return rf, X, y, metrics


# =========================
# 4) SHAP ANALYSIS
# =========================

def run_shap_taiwan(model, X: pd.DataFrame, sample_size: int = SHAP_SAMPLE_SIZE, random_state: int = 42):
    print("[4/4] Running SHAP analysis for Taiwan 30k...")
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    DEPENDENCE_DIR.mkdir(parents=True, exist_ok=True)

    n_rows = len(X)
    n_sample = min(sample_size, n_rows)
    print(f"[4/4] Sampling {n_sample} rows out of {n_rows} for SHAP...")

    X_sample = X.sample(n=n_sample, random_state=random_state)

    # Background for TreeExplainer
    bg_size = min(SHAP_BG_SIZE, n_sample)
    background = X_sample.sample(n=bg_size, random_state=random_state)

    print("[4/4] Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model, background)

    print("[4/4] Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)

    # Handle different SHAP return formats
    if isinstance(shap_values, list):
        # Multi-class: choose class 1 if available, else last class
        if len(shap_values) >= 2:
            shap_array = np.array(shap_values[1])
            selected_class = 1
        else:
            shap_array = np.array(shap_values[-1])
            selected_class = len(shap_values) - 1
        print(f"DEBUG: Detected list SHAP values with {len(shap_values)} classes. Using class {selected_class}.")
    else:
        shap_array = np.array(shap_values)
        if shap_array.ndim == 3:
            # (n_samples, n_features, n_classes)
            if shap_array.shape[2] >= 2:
                shap_array = shap_array[:, :, 1]
                print(f"DEBUG: Detected 3D SHAP values {shap_array.shape}. Selecting class 1.")
            else:
                shap_array = shap_array[:, :, 0]
                print(f"DEBUG: Detected 3D SHAP values {shap_array.shape}. Selecting class 0.")
        elif shap_array.ndim == 2:
            print(f"DEBUG: Detected 2D SHAP values {shap_array.shape}.")
        else:
            raise RuntimeError(f"Unexpected SHAP array shape: {shap_array.shape}")

    print(f"DEBUG: Final SHAP shape for plotting: {shap_array.shape}")

    # 4A) Summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(
        shap_array,
        X_sample,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    summary_path = SHAP_DIR / "shap_taiwan_v3_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot: {summary_path}")

    # 4B) Bar plot (global feature importance)
    plt.figure()
    shap.summary_plot(
        shap_array,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    bar_path = SHAP_DIR / "shap_taiwan_v3_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bar plot: {bar_path}")

    # 4C) Dependence plots for top 5 features
    mean_abs = np.abs(shap_array).mean(axis=0)
    feature_names = list(X_sample.columns)
    top_indices = np.argsort(mean_abs)[::-1][:5]
    top_features = [feature_names[i] for i in top_indices]

    print(f"Top feature indices for dependence plots: {top_indices.tolist()}")
    print(f"Top feature names: {top_features}")

    for feat in top_features:
        plt.figure()
        shap.dependence_plot(
            feat,
            shap_array,
            X_sample,
            show=False,
        )
        plt.tight_layout()
        dep_path = DEPENDENCE_DIR / f"shap_taiwan_v3_dependence_{feat}.png"
        plt.savefig(dep_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved dependence plot for: {feat} -> {dep_path}")

    print("[4/4] SHAP analysis finished.")


# =========================
# MAIN END-TO-END PIPELINE
# =========================

def main():
    print("==== TAIWAN 30K RF + SHAP PIPELINE (DAY 1–2 FINAL) ====")

    with tqdm(total=4, desc="Pipeline steps", unit="step") as pbar:
        # Step 1: load raw data
        df_raw = load_raw_taiwan()
        pbar.update(1)

        # Step 2: feature engineering
        df_features, target_col = build_taiwan_features(df_raw)
        pbar.update(1)

        # Step 3: train RF and save model + metrics
        model, X_all, y_all, metrics = train_rf_taiwan(df_features, target_col)
        pbar.update(1)

        # Step 4: SHAP analysis on full feature set (sampled rows)
        run_shap_taiwan(model, X_all)
        pbar.update(1)

    print("==== PIPELINE COMPLETE ====")
    print(f"Features CSV : {FEATURES_PATH}")
    print(f"Model path   : {MODEL_PATH}")
    print(f"Metrics JSON : {METRICS_PATH}")
    print(f"SHAP outputs : {SHAP_DIR}")


if __name__ == "__main__":
    main()
