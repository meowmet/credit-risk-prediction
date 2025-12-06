"""
Day 1 + Day 2 Taiwan 30k RF + FE + SHAP pipeline

ÇIKTILAR (hepsi outputs_taiwan_day1_day2/ altında):

- Day 1:
  - taiwan_fe_v0.csv
  - rf_taiwan_baseline.pkl
  - rf_taiwan_v0_results.json
  - taiwan_fe_v1.csv
  - rf_taiwan_v1.pkl
  - rf_taiwan_v1_results.json

- Day 2:
  - taiwan_fe_v2.csv
  - rf_taiwan_v2.pkl
  - rf_taiwan_v2_results.json
  - taiwan_fe_v3.csv
  - rf_taiwan_v3.pkl
  - rf_taiwan_v3_results.json
  - taiwan_rf_comparison.csv
  - taiwan_final_features.csv
  - shap_taiwan_deep/ (summary, bar, dependence, local, interaction)
  - taiwan_fe_documentation.md
  - fe_template_large_datasets.json
  - rf_shap_pipeline_template.py
"""

import json
from pathlib import Path
from typing import List, Dict, Any

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
    #İlk satır mapping olduğu için skiprows=[0].

    df = pd.read_csv(path, skiprows=[0])

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


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
    """
    return df[FE_V0_BASE_COLS + [TARGET_COL]].copy()



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



def build_fe_v1(df: pd.DataFrame) -> pd.DataFrame:
    """
    FE v1 (Day 1 öğlen):
    - Utilization ratios
    - Payment ratios
    - Past bill differences
    - Simple trend features
    """

    out = df.copy()

    out["AVG_PAY_STATUS"] = out[PAY_STATUS_COLS].mean(axis=1)
    out["MAX_PAY_STATUS"] = out[PAY_STATUS_COLS].max(axis=1)
    out["OVERDUE_COUNT"] = (out[PAY_STATUS_COLS] > 0).sum(axis=1)

    out["TOTAL_BILL_AMT"] = out[BILL_COLS].sum(axis=1)
    out["TOTAL_PAY_AMT"] = out[PAY_AMT_COLS].sum(axis=1)

    out["UTILIZATION_RATIO"] = out["TOTAL_BILL_AMT"] / (out["LIMIT_BAL"] + 1e-6)

    out["BILL_TREND"] = out["BILL_AMT6"] - out["BILL_AMT1"]
    out["BILL_TREND_PCT"] = (
        (out["BILL_AMT6"] - out["BILL_AMT1"]) /
        (out["BILL_AMT1"].abs() + 1e-6)
    )

    out["PAYMENT_RATIO"] = out["TOTAL_PAY_AMT"] / (
        out["TOTAL_BILL_AMT"].abs() + 1e-6
    )
    out["RECENT_PAYMENT_RATIO"] = out["PAY_AMT6"] / (
        out["BILL_AMT6"].abs() + 1e-6
    )

    out["PAY_TREND"] = out["PAY_AMT6"] - out["PAY_AMT1"]

    for i in range(1, 6):
        col_from = f"BILL_AMT{i}"
        col_to = f"BILL_AMT{i+1}"
        out[f"BILL_DIFF_{i}_{i+1}"] = out[col_to] - out[col_from]

    return out



def build_fe_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
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
    precision, recall, f1_, _ = precision_recall_fscore_support(
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
        "f1_at_best_thr": float(f1_),
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

    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]
        base_value = explainer.expected_value[1]
    else:
        shap_vals_pos = shap_values
        base_value = explainer.expected_value

    import matplotlib.pyplot as plt

    shap.summary_plot(
        shap_vals_pos,
        X_sample,
        show=False,
        plot_type="dot",
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_taiwan_v3_summary.png", dpi=150)
    plt.close()

    shap.summary_plot(
        shap_vals_pos,
        X_sample,
        show=False,
        plot_type="bar",
    )
    plt.tight_layout()
    plt.savefig(out_dir / "shap_taiwan_v3_bar.png", dpi=150)
    plt.close()

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


# ================== 5. FE & PIPELINE TEMPLATES  ==================

LARGE_DATASET_FE_TEMPLATE = {
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


RF_SHAP_PIPELINE_TEMPLATE = """\
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

TARGET_COL = "default payment next month"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: FE v3 mantığını buraya kopyala / uyarl
    return df

def run_rf_shap_pipeline(df: pd.DataFrame, out_dir: Path, dataset_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_features(df)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    # SHAP global summary
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_vals_pos = shap_values[1]
    else:
        shap_vals_pos = shap_values

    import matplotlib.pyplot as plt
    shap.summary_plot(shap_vals_pos, X_test, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_{dataset_name}_summary.png", dpi=150)
    plt.close()
"""


def save_fe_template_for_large_datasets(path: Path):
    save_json(LARGE_DATASET_FE_TEMPLATE, path)


def save_rf_shap_pipeline_template(path: Path):
    path.write_text(RF_SHAP_PIPELINE_TEMPLATE)


#   documentation for comparison

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



def run_day1_day2_pipeline():
    df_raw = load_taiwan_data(DATA_PATH)

    df_v0 = build_fe_v0(df_raw)
    save_features_as_csv(df_v0, "taiwan_fe_v0", drop_target=True)

    X_v0 = df_v0.drop(columns=[TARGET_COL])
    y = df_v0[TARGET_COL]

    X_train_v0, X_test_v0, y_train_v0, y_test_v0 = train_test_split(
        X_v0, y, test_size=0.2, stratify=y, random_state=42
    )

    rf_v0 = train_rf_model(X_train_v0, y_train_v0)
    rf_v0_metrics = evaluate_rf(rf_v0, X_test_v0, y_test_v0, fe_version="v0")

    rf_v0_path = OUTPUT_DIR / "rf_taiwan_baseline.pkl"
    joblib.dump(rf_v0, rf_v0_path)
    save_json(rf_v0_metrics, OUTPUT_DIR / "rf_taiwan_v0_results.json")

    df_v1 = build_fe_v1(df_raw)
    save_features_as_csv(df_v1, "taiwan_fe_v1", drop_target=True)

    X_v1 = df_v1.drop(columns=[TARGET_COL])
    y_v1 = df_v1[TARGET_COL]

    X_train_v1, X_test_v1, y_train_v1, y_test_v1 = train_test_split(
        X_v1, y_v1, test_size=0.2, stratify=y_v1, random_state=42
    )

    rf_v1 = train_rf_model(X_train_v1, y_train_v1)
    rf_v1_metrics = evaluate_rf(rf_v1, X_test_v1, y_test_v1, fe_version="v1")

    rf_v1_path = OUTPUT_DIR / "rf_taiwan_v1.pkl"
    joblib.dump(rf_v1, rf_v1_path)
    save_json(rf_v1_metrics, OUTPUT_DIR / "rf_taiwan_v1_results.json")

    df_v2 = build_fe_v2(df_raw)
    save_features_as_csv(df_v2, "taiwan_fe_v2", drop_target=True)

    X_v2 = df_v2.drop(columns=[TARGET_COL])
    y_v2 = df_v2[TARGET_COL]

    X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(
        X_v2, y_v2, test_size=0.2, stratify=y_v2, random_state=42
    )

    rf_v2 = train_rf_model(
        X_train_v2,
        y_train_v2,
        n_estimators=500,
        max_depth=16,
        max_features="sqrt",
    )
    rf_v2_metrics = evaluate_rf(rf_v2, X_test_v2, y_test_v2, fe_version="v2")

    rf_v2_path = OUTPUT_DIR / "rf_taiwan_v2.pkl"
    joblib.dump(rf_v2, rf_v2_path)
    save_json(rf_v2_metrics, OUTPUT_DIR / "rf_taiwan_v2_results.json")

    df_v3 = build_fe_v3(df_raw)
    save_features_as_csv(df_v3, "taiwan_fe_v3", drop_target=True)

    X_v3 = df_v3.drop(columns=[TARGET_COL])
    y_v3 = df_v3[TARGET_COL]

    X_train_v3, X_test_v3, y_train_v3, y_test_v3 = train_test_split(
        X_v3, y_v3, test_size=0.2, stratify=y_v3, random_state=42
    )

    rf_v3 = train_rf_model(
        X_train_v3,
        y_train_v3,
        n_estimators=600,
        max_depth=18,
        max_features="sqrt",
    )
    rf_v3_metrics = evaluate_rf(rf_v3, X_test_v3, y_test_v3, fe_version="v3")

    rf_v3_path = OUTPUT_DIR / "rf_taiwan_v3.pkl"
    joblib.dump(rf_v3, rf_v3_path)
    save_json(rf_v3_metrics, OUTPUT_DIR / "rf_taiwan_v3_results.json")

    comparison = pd.DataFrame(
        [
            rf_v0_metrics,
            rf_v1_metrics,
            rf_v2_metrics,
            rf_v3_metrics,
        ]
    )
    comparison_path = OUTPUT_DIR / "taiwan_rf_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    final_fe_path = save_features_as_csv(
        df_v3, "taiwan_final_features", drop_target=True
    )

    run_deep_shap_analysis(
        model=rf_v3,
        X=X_v3,
        out_dir=SHAP_DIR,
        sample_size=3000,
        n_local_examples=5,
    )

    save_fe_documentation(OUTPUT_DIR / "taiwan_fe_documentation.md")

    save_fe_template_for_large_datasets(
        OUTPUT_DIR / "fe_template_large_datasets.json"
    )
    save_rf_shap_pipeline_template(
        OUTPUT_DIR / "rf_shap_pipeline_template.py"
    )

    print("\nDay 1 + Day 2 pipeline completed.")
    print(f"Comparison table saved to: {comparison_path}")
    print(f"Final features saved to:   {final_fe_path}")
    print(f"SHAP outputs in:          {SHAP_DIR}")


if __name__ == "__main__":
    run_day1_day2_pipeline()
