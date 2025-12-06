from pathlib import Path
from typing import List, Dict, Any
import json

import numpy as np
import pandas as pd
import shap
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
DATA_PATH = "credit_30k.csv"  # Taiwan 30k dataset
TARGET_COL = "default payment next month"

OUTPUT_DIR = Path("outputs_taiwan_day1_day2")
SHAP_DIR = OUTPUT_DIR / "shap_taiwan_deep"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)


def save_features_as_csv(df: pd.DataFrame, filename: str, drop_target: bool = True) -> Path:
    """
    DataFrame'i güvenli şekilde CSV olarak kaydeder.
    Parquet engine gerektirmez, her ortamda çalışır.
    """
    if drop_target and TARGET_COL in df.columns:
        df_to_save = df.drop(columns=[TARGET_COL])
    else:
        df_to_save = df

    path = OUTPUT_DIR / f"{filename}.csv"
    df_to_save.to_csv(path, index=False)
    print(f"[SAVE] {filename} -> {path}")
    return path


def save_json(obj: Any, path: Path):
    path.write_text(json.dumps(obj, indent=2))


def load_taiwan_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Taiwan credit card default dataset'i okur.
    İlk satır mapping olduğu için skiprows=[0].
    """
    df = pd.read_csv(path, skiprows=[0])

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Target tipi
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      - duplicate drop
      - NA raporu + full-case drop
      - simple outlier snapshot (LIMIT_BAL, AGE)
    """
    df = df.drop_duplicates()

    # NA raporu
    na_counts = df.isna().sum()
    na_report_path = OUTPUT_DIR / "na_report.csv"
    na_counts.to_csv(na_report_path, header=["na_count"])
    print(f"[INFO] NA report saved to {na_report_path}")

    # Basit strateji: herhangi bir kolonda NA varsa satırı at
    df = df.dropna().reset_index(drop=True)

    # Outlier snapshot: LIMIT_BAL ve AGE quantile'ları
    outlier_info = df[["LIMIT_BAL", "AGE"]].quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    outlier_path = OUTPUT_DIR / "outlier_snapshot_limitbal_age.csv"
    outlier_info.to_csv(outlier_path)
    print(f"[INFO] Outlier snapshot (LIMIT_BAL, AGE) -> {outlier_path}")

    return df


# ---- FE v0: raw basic features ----

FE_V0_BASE_COLS: List[str] = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]


def build_fe_v0(df: pd.DataFrame) -> pd.DataFrame:
    """
    FE v0: sadece raw temel kolonlar.
    Day 1 sabah: baseline RF için Student A output'u.
    """
    return df[FE_V0_BASE_COLS + [TARGET_COL]].copy()


# Ortak kolon setleri
PAY_STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
]
PAY_AMT_COLS = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


# ---- FE v1: utilization & payment ratios, simple trends ----

def build_fe_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    FE v1 (Day 1 öğlen):
    - Utilization ratios
    - Payment ratios
    - Past bill differences
    - Simple trend features
    """

    out = df.copy()

    # Payment status pattern
    out["AVG_PAY_STATUS"] = out[PAY_STATUS_COLS].mean(axis=1)
    out["MAX_PAY_STATUS"] = out[PAY_STATUS_COLS].max(axis=1)
    out["OVERDUE_COUNT"] = (out[PAY_STATUS_COLS] > 0).sum(axis=1)

    # Aggregates
    out["TOTAL_BILL_AMT"] = out[BILL_COLS].sum(axis=1)
    out["TOTAL_PAY_AMT"] = out[PAY_AMT_COLS].sum(axis=1)

    # Utilization vs limit
    out["UTILIZATION_RATIO"] = out["TOTAL_BILL_AMT"] / (out["LIMIT_BAL"] + 1e-6)

    # Bill trend (level & relative change)
    out["BILL_TREND"] = out["BILL_AMT6"] - out["BILL_AMT1"]
    out["BILL_TREND_PCT"] = (
        (out["BILL_AMT6"] - out["BILL_AMT1"]) /
        (out["BILL_AMT1"].abs() + 1e-6)
    )

    # Payment vs bill
    out["PAYMENT_RATIO"] = out["TOTAL_PAY_AMT"] / (
        out["TOTAL_BILL_AMT"].abs() + 1e-6
    )
    out["RECENT_PAYMENT_RATIO"] = out["PAY_AMT6"] / (
        out["BILL_AMT6"].abs() + 1e-6
    )

    # Payment trend
    out["PAY_TREND"] = out["PAY_AMT6"] - out["PAY_AMT1"]

    # Simple past bill diffs
    for i in range(1, 6):
        col_from = f"BILL_AMT{i}"
        col_to = f"BILL_AMT{i+1}"
        out[f"BILL_DIFF_{i}_{i+1}"] = out[col_to] - out[col_from]

    return out


# ---- FE v2: rolling trends, payment stability, variance, aggregations ----

def build_fe_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    FE v2 (Day 2 sabah):
    - Rolling trends
    - Payment stability
    - Variance features
    - Aggregation: mean/max/min across 6 months
    """
    out = build_fe_v1(df)  # v1 üzerine inşa

    # Aggregations over 6 months
    out["BILL_MEAN_6M"] = out[BILL_COLS].mean(axis=1)
    out["BILL_MAX_6M"] = out[BILL_COLS].max(axis=1)
    out["BILL_MIN_6M"] = out[BILL_COLS].min(axis=1)
    out["BILL_STD_6M"] = out[BILL_COLS].std(axis=1)

    out["PAY_AMT_MEAN_6M"] = out[PAY_AMT_COLS].mean(axis=1)
    out["PAY_AMT_MAX_6M"] = out[PAY_AMT_COLS].max(axis=1)
    out["PAY_AMT_MIN_6M"] = out[PAY_AMT_COLS].min(axis=1)
    out["PAY_AMT_STD_6M"] = out[PAY_AMT_COLS].std(axis=1)

    # Payment stability: consecutive differences in PAY_AMT
    pay_amt_diffs = []
    for i in range(1, 6):
        col_from = f"PAY_AMT{i}"
        col_to = f"PAY_AMT{i+1}"
        diff_col = f"PAY_AMT_DIFF_{i}_{i+1}"
        out[diff_col] = out[col_to] - out[col_from]
        pay_amt_diffs.append(diff_col)

    out["PAY_AMT_DIFF_MEAN"] = out[pay_amt_diffs].mean(axis=1)
    out["PAY_AMT_DIFF_STD"] = out[pay_amt_diffs].std(axis=1)

    # Rolling trend: average slope of bills (approx)
    month_idx = np.arange(1, 7)
    bills = out[BILL_COLS].values
    # slope ~ cov(month_idx, bill) / var(month_idx)
    month_centered = month_idx - month_idx.mean()
    var_month = (month_centered ** 2).sum()
    bill_centered = bills - bills.mean(axis=1, keepdims=True)
    slopes = (bill_centered * month_centered).sum(axis=1) / (var_month + 1e-6)
    out["BILL_TREND_SLOPE"] = slopes

    return out


# ---- FE v3: behavioral patterns, overdue frequency, accelerating debt ----

def build_fe_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    FE v3 (Day 2 öğleden sonra):
    - Behavioral patterns
    - Frequency of overdue
    - Accelerating debt growth
    """
    out = build_fe_v2(df)  # v2 üzerine inşa

    # Overdue frequency patterns
    out["SEVERE_OVERDUE_COUNT"] = (out[PAY_STATUS_COLS] >= 2).sum(axis=1)
    out["CHRONIC_OVERDUE"] = ((out[PAY_STATUS_COLS] > 0).sum(axis=1) >= 3).astype(int)

    # Weeks / months since last non-overdue
    # (PAY_x <= 0 means no delay or advance payment)
    non_overdue_mask = (out[PAY_STATUS_COLS] <= 0)
    last_non_overdue_pos = non_overdue_mask.iloc[:, ::-1].idxmax(axis=1)
    # map PAY_6 -> 0 months ago, PAY_5 -> 1, ..., PAY_0 -> 5
    pay_pos_map = {"PAY_6": 0, "PAY_5": 1, "PAY_4": 2, "PAY_3": 3, "PAY_2": 4, "PAY_0": 5}
    out["MONTHS_SINCE_NON_OVERDUE"] = last_non_overdue_pos.map(pay_pos_map).fillna(6)

    # Accelerating debt growth: compare first vs last 3 months
    first3_bills = out[["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"]].mean(axis=1)
    last3_bills = out[["BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].mean(axis=1)
    out["DEBT_ACCELERATION"] = last3_bills - first3_bills
    out["DEBT_ACCELERATION_PCT"] = (
        (last3_bills - first3_bills) / (first3_bills.abs() + 1e-6)
    )

    # Behavioral ratio: fraction of months where payment < bill
    pay_vs_bill_frac = []
    for i in range(1, 7):
        pay_col = f"PAY_AMT{i}"
        bill_col = f"BILL_AMT{i}"
        pay_vs_bill_frac.append(out[pay_col] < out[bill_col])
    pay_vs_bill_frac = np.column_stack(pay_vs_bill_frac)
    out["MONTHS_PAYMENT_BELOW_BILL"] = pay_vs_bill_frac.sum(axis=1)

    # High utilization flag
    out["HIGH_UTILIZATION_FLAG"] = (out["UTILIZATION_RATIO"] > 0.8).astype(int)

    return out


def best_threshold_and_f1(y_true: np.ndarray, y_proba: np.ndarray):
    best_thr = 0.5
    best_f1 = 0.0
    for thr in np.linspace(0.1, 0.9, 81):
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def evaluate_rf(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    fe_version: str,
) -> Dict[str, Any]:
    proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)
    best_thr, best_f1 = best_threshold_and_f1(y_val.values, proba)

    y_pred = (proba >= best_thr).astype(int)
    precision, recall, f1_at_thr, _ = precision_recall_fscore_support(
        y_val, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    return {
        "fe_version": fe_version,
        "auc": float(auc),
        "best_f1": float(best_f1),
        "best_thr": float(best_thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1_at_best_thr": float(f1_at_thr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_samples": int(len(y_val)),
    }


def train_rf_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 12,
    max_features="sqrt",
    random_state: int = 42,
) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    return rf


def run_deep_shap_analysis(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    out_dir: Path,
    sample_size: int = 3000,
    n_local_examples: int = 5,
):
    """
    Deep SHAP analysis for RF model on FE v3:
    - Global summary plot
    - Bar plot
    - Dependence plots for top 5 features
    - Local explanations for a few clients
    - Interaction summary (optional)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Binary classification: shap_values can be list[2] or array
    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]
        base_value = explainer.expected_value[1]
    else:
        shap_vals_pos = shap_values
        base_value = explainer.expected_value

    import matplotlib.pyplot as plt

    # ---- Global summary plot ----
    shap.summary_plot(
        shap_vals_pos,
        X_sample,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_taiwan_v3_summary.png", dpi=150)
    plt.close()

    # ---- Global bar plot ----
    shap.summary_plot(
        shap_vals_pos,
        X_sample,
        show=False,
        plot_type="bar",
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_taiwan_v3_bar.png", dpi=150)
    plt.close()

    # ---- Top features for dependence plots ----
    mean_abs_shap = np.mean(np.abs(shap_vals_pos), axis=0)
    feature_order = np.argsort(-mean_abs_shap)
    top_idx = feature_order[:5]
    top_features = [X_sample.columns[i] for i in top_idx]

    dep_dir = out_dir / "dependence_plots"
    dep_dir.mkdir(exist_ok=True)

    for feat in top_features:
        shap.dependence_plot(
            feat,
            shap_vals_pos,
            X_sample,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(dep_dir / f"shap_dependence_{feat}.png", dpi=150)
        plt.close()

    # ---- Local explanations for a few clients ----
    local_dir = out_dir / "local_explanations"
    local_dir.mkdir(exist_ok=True)

    if len(X_sample) >= n_local_examples:
        idxs = X_sample.sample(n=n_local_examples, random_state=123).index
    else:
        idxs = X_sample.index

    for i, idx in enumerate(idxs, start=1):
        x_row = X_sample.loc[idx:idx]
        if isinstance(shap_vals_pos, np.ndarray):
            shap_row = shap_vals_pos[X_sample.index.get_loc(idx)]
        else:
            shap_row = shap_vals_pos[X_sample.index.get_loc(idx)]

        shap.force_plot(
            base_value,
            shap_row,
            x_row,
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(local_dir / f"force_plot_client_{i}.png", dpi=150)
        plt.close()

    # ---- Interaction summary (optional, small subset) ----
    inter_dir = out_dir / "interaction"
    inter_dir.mkdir(exist_ok=True)

    X_inter = X_sample
    if len(X_inter) > 500:
        X_inter = X_inter.sample(n=500, random_state=42)

    try:
        shap_inter = explainer.shap_interaction_values(X_inter)
        if isinstance(shap_inter, list):
            shap_inter_pos = shap_inter[1]
        else:
            shap_inter_pos = shap_inter

        shap.summary_plot(
            shap_inter_pos,
            X_inter,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(inter_dir / "shap_interaction_summary.png", dpi=150)
        plt.close()
    except Exception as e:
        (out_dir / "interaction_error.txt").write_text(str(e))


LARGE_DATASET_FE_TEMPLATE: Dict[str, Any] = {
    "base_numeric": FE_V0_BASE_COLS,
    "pay_status_cols": PAY_STATUS_COLS,
    "bill_amount_cols": BILL_COLS,
    "payment_amount_cols": PAY_AMT_COLS,
    "feature_versions": {
        "v0": "raw basic features (demographic + PAY_x)",
        "v1": "utilization ratios, payment ratios, basic trends, bill diffs",
        "v2": "6M aggregates, stability metrics, slopes",
        "v3": "behavior patterns, overdue frequency, debt acceleration",
    },
}


def save_fe_template_for_large_datasets(path: Path):
    save_json(LARGE_DATASET_FE_TEMPLATE, path)


def save_fe_documentation(path: Path):
    lines = []

    def add(line: str = ""):
        lines.append(line)

    add("# Taiwan FE Versions Documentation")
    add("")
    add("## FE v0")
    add("- Raw basic features:")
    for col in FE_V0_BASE_COLS:
        add(f"  - {col}")
    add("")
    add("## FE v1")
    add("- Adds:")
    add("  - AVG_PAY_STATUS, MAX_PAY_STATUS, OVERDUE_COUNT")
    add("  - TOTAL_BILL_AMT, TOTAL_PAY_AMT")
    add("  - UTILIZATION_RATIO")
    add("  - BILL_TREND, BILL_TREND_PCT")
    add("  - PAYMENT_RATIO, RECENT_PAYMENT_RATIO, PAY_TREND")
    add("  - BILL_DIFF_1_2 ... BILL_DIFF_5_6")
    add("")
    add("## FE v2")
    add("- Adds:")
    add("  - BILL_MEAN_6M, BILL_MAX_6M, BILL_MIN_6M, BILL_STD_6M")
    add("  - PAY_AMT_MEAN_6M, PAY_AMT_MAX_6M, PAY_AMT_MIN_6M, PAY_AMT_STD_6M")
    add("  - PAY_AMT_DIFF_1_2 ... PAY_AMT_DIFF_5_6")
    add("  - PAY_AMT_DIFF_MEAN, PAY_AMT_DIFF_STD")
    add("  - BILL_TREND_SLOPE (approx slope of bill over months)")
    add("")
    add("## FE v3 (final)")
    add("- Adds behavioral features:")
    add("  - SEVERE_OVERDUE_COUNT (PAY_x >= 2)")
    add("  - CHRONIC_OVERDUE (>=3 months overdue)")
    add("  - MONTHS_SINCE_NON_OVERDUE (recency of good behaviour)")
    add("  - DEBT_ACCELERATION, DEBT_ACCELERATION_PCT")
    add("  - MONTHS_PAYMENT_BELOW_BILL")
    add("  - HIGH_UTILIZATION_FLAG (UTILIZATION_RATIO > 0.8)")
    add("")
    add("Final recommended feature set = FE v3 columns (excluding target).")

    path.write_text("\n".join(lines))


def run_full_day1_day2_pipeline():
    # ---- Load + Clean ----
    df_raw = load_taiwan_data(DATA_PATH)
    df_clean = basic_cleaning(df_raw)

    y_full = df_clean[TARGET_COL]
    # Tüm FE versiyonları için AYNI split'i kullanmak için index'ten bölüyoruz
    idx_train, idx_test = train_test_split(
        df_clean.index, test_size=0.2, stratify=y_full, random_state=42
    )

    metrics_list = []
    models = {}

    # ===== FE v0 =====
    print("\n=== FE v0 + RF baseline ===")
    df_v0 = build_fe_v0(df_clean)
    save_features_as_csv(df_v0, "taiwan_fe_v0", drop_target=True)

    X_v0 = df_v0.drop(columns=[TARGET_COL])
    y_v0 = df_v0[TARGET_COL]

    X_train_v0 = X_v0.loc[idx_train]
    X_test_v0 = X_v0.loc[idx_test]
    y_train_v0 = y_v0.loc[idx_train]
    y_test_v0 = y_v0.loc[idx_test]

    rf_v0 = train_rf_model(X_train_v0, y_train_v0)
    rf_v0_metrics = evaluate_rf(rf_v0, X_test_v0, y_test_v0, fe_version="v0")
    metrics_list.append(rf_v0_metrics)
    models["v0"] = rf_v0

    joblib.dump(rf_v0, OUTPUT_DIR / "rf_taiwan_baseline.pkl")
    save_json(rf_v0_metrics, OUTPUT_DIR / "rf_taiwan_v0_results.json")

    # ===== FE v1 =====
    print("\n=== FE v1 + RF ===")
    df_v1 = build_fe_v1(df_clean)
    save_features_as_csv(df_v1, "taiwan_fe_v1", drop_target=True)

    X_v1 = df_v1.drop(columns=[TARGET_COL])
    y_v1 = df_v1[TARGET_COL]

    X_train_v1 = X_v1.loc[idx_train]
    X_test_v1 = X_v1.loc[idx_test]
    y_train_v1 = y_v1.loc[idx_train]
    y_test_v1 = y_v1.loc[idx_test]

    rf_v1 = train_rf_model(X_train_v1, y_train_v1)
    rf_v1_metrics = evaluate_rf(rf_v1, X_test_v1, y_test_v1, fe_version="v1")
    metrics_list.append(rf_v1_metrics)
    models["v1"] = rf_v1

    joblib.dump(rf_v1, OUTPUT_DIR / "rf_taiwan_v1.pkl")
    save_json(rf_v1_metrics, OUTPUT_DIR / "rf_taiwan_v1_results.json")

    # ===== FE v2 =====
    print("\n=== FE v2 + RF (tuned) ===")
    df_v2 = build_fe_v2(df_clean)
    save_features_as_csv(df_v2, "taiwan_fe_v2", drop_target=True)

    X_v2 = df_v2.drop(columns=[TARGET_COL])
    y_v2 = df_v2[TARGET_COL]

    X_train_v2 = X_v2.loc[idx_train]
    X_test_v2 = X_v2.loc[idx_test]
    y_train_v2 = y_v2.loc[idx_train]
    y_test_v2 = y_v2.loc[idx_test]

    rf_v2 = train_rf_model(
        X_train_v2,
        y_train_v2,
        n_estimators=500,
        max_depth=16,
        max_features="sqrt",
    )
    rf_v2_metrics = evaluate_rf(rf_v2, X_test_v2, y_test_v2, fe_version="v2")
    metrics_list.append(rf_v2_metrics)
    models["v2"] = rf_v2

    joblib.dump(rf_v2, OUTPUT_DIR / "rf_taiwan_v2.pkl")
    save_json(rf_v2_metrics, OUTPUT_DIR / "rf_taiwan_v2_results.json")

    # ===== FE v3 =====
    print("\n=== FE v3 + RF (final) ===")
    df_v3 = build_fe_v3(df_clean)
    save_features_as_csv(df_v3, "taiwan_fe_v3", drop_target=True)

    X_v3 = df_v3.drop(columns=[TARGET_COL])
    y_v3 = df_v3[TARGET_COL]

    X_train_v3 = X_v3.loc[idx_train]
    X_test_v3 = X_v3.loc[idx_test]
    y_train_v3 = y_v3.loc[idx_train]
    y_test_v3 = y_v3.loc[idx_test]

    rf_v3 = train_rf_model(
        X_train_v3,
        y_train_v3,
        n_estimators=600,
        max_depth=18,
        max_features="sqrt",
    )
    rf_v3_metrics = evaluate_rf(rf_v3, X_test_v3, y_test_v3, fe_version="v3")
    metrics_list.append(rf_v3_metrics)
    models["v3"] = rf_v3

    joblib.dump(rf_v3, OUTPUT_DIR / "rf_taiwan_v3.pkl")
    save_json(rf_v3_metrics, OUTPUT_DIR / "rf_taiwan_v3_results.json")

    # ---- Comparison table ----
    comparison = pd.DataFrame(metrics_list)
    comparison_path = OUTPUT_DIR / "taiwan_rf_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    # ---- Final Taiwan feature set = FE v3 ----
    final_fe_path = save_features_as_csv(
        df_v3, "taiwan_final_features", drop_target=True
    )

    # ---- FE docs + template (Student A) ----
    save_fe_documentation(OUTPUT_DIR / "taiwan_fe_documentation.md")
    save_fe_template_for_large_datasets(
        OUTPUT_DIR / "fe_template_large_datasets.json"
    )

    # ---- SHAP deep analysis on final model (Student B) ----
    run_deep_shap_analysis(
        model=rf_v3,
        X=X_v3,
        out_dir=SHAP_DIR,
        sample_size=3000,
        n_local_examples=5,
    )


    print("\n================ RF RESULTS (TEST SET, PER FE VERSION) ================")
    for m in metrics_list:
        print(
            f"FE {m['fe_version']}: "
            f"AUC={m['auc']:.4f}, "
            f"best_F1={m['best_f1']:.4f}, "
            f"thr={m['best_thr']:.3f}, "
            f"precision={m['precision']:.4f}, "
            f"recall={m['recall']:.4f}"
        )

    print("\nFull metrics table saved to:", comparison_path)
    print("Final features saved to:   ", final_fe_path)
    print("SHAP outputs in:           ", SHAP_DIR)
    print("Outputs directory:         ", OUTPUT_DIR)


if __name__ == "__main__":
    run_full_day1_day2_pipeline()
