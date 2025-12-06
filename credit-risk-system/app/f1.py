import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


DATA_PATH = "credit_30k.csv"
TARGET_COL = "default payment next month"


# ========= 1. DATA LOADING & FEATURE ENGINEERING =========

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Taiwan credit card default dataset'i okur.
    İlk satır X1.. mapping olduğu için skiprows=[0].
    """
    df = pd.read_csv(path, skiprows=[0])

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    V1'de kullandığımız gelişmiş feature set.
    """
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                 "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    pay_amt_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
                    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

    # Payment status pattern
    df["AVG_PAY_STATUS"] = df[pay_cols].mean(axis=1)
    df["MAX_PAY_STATUS"] = df[pay_cols].max(axis=1)
    df["OVERDUE_COUNT"] = (df[pay_cols] > 0).sum(axis=1)

    # Aggregates
    df["TOTAL_BILL_AMT"] = df[bill_cols].sum(axis=1)
    df["TOTAL_PAY_AMT"] = df[pay_amt_cols].sum(axis=1)

    # Utilization vs limit
    df["UTILIZATION_RATIO"] = df["TOTAL_BILL_AMT"] / (df["LIMIT_BAL"] + 1e-6)

    # Bill trend (level & relative change)
    df["BILL_TREND"] = df["BILL_AMT6"] - df["BILL_AMT1"]
    df["BILL_TREND_PCT"] = (
        (df["BILL_AMT6"] - df["BILL_AMT1"]) /
        (df["BILL_AMT1"].abs() + 1e-6)
    )

    # Payment vs bill
    df["PAYMENT_RATIO"] = df["TOTAL_PAY_AMT"] / (df["TOTAL_BILL_AMT"].abs() + 1e-6)
    df["RECENT_PAYMENT_RATIO"] = df["PAY_AMT6"] / (df["BILL_AMT6"].abs() + 1e-6)

    # Payment trend
    df["PAY_TREND"] = df["PAY_AMT6"] - df["PAY_AMT1"]

    return df


def get_feature_cols() -> list[str]:
    """
    Stacking için kullanacağımız feature set.
    """
    raw_features = [
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

    fe_features = [
        "AVG_PAY_STATUS",
        "MAX_PAY_STATUS",
        "OVERDUE_COUNT",
        "UTILIZATION_RATIO",
        "BILL_TREND",
        "BILL_TREND_PCT",
        "PAYMENT_RATIO",
        "RECENT_PAYMENT_RATIO",
        "PAY_TREND",
    ]

    return raw_features + fe_features


# ========= 2. MODELS =========

def build_catboost(class_weights=None) -> CatBoostClassifier:
    """
    CatBoost base model (biraz daha konservatif learning_rate).
    """
    model = CatBoostClassifier(
        depth=6,
        learning_rate=0.05,
        iterations=1000,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_state=42,
        class_weights=class_weights,
        border_count=255,
        bagging_temperature=0.5,
        verbose=False
    )
    return model


def build_lgbm(class_weight=None) -> LGBMClassifier:
    """
    LightGBM base model (gradient boosted trees).
    """
    model = LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        objective="binary",
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    return model


def build_meta_model() -> LogisticRegression:
    """
    Stacking meta model (üstte, base modellerin çıktısını kullanacak).
    """
    meta = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1
    )
    return meta


# ========= 3. STACKING TRAINING =========

def train_stacking_ensemble(X: pd.DataFrame, y: pd.Series,
                            n_splits: int = 5, test_size: float = 0.2,
                            random_state: int = 42):
    """
    - X, y'den önce train/test ayırır.
    - Train kısmında Stratified KFold ile:
        * CatBoost ve LightGBM için OOF probaları üretir.
        * Bu OOF'larla meta modeli (LogReg) eğitir.
    - Base modelleri tüm train'de yeniden eğitir.
    - Test setinde:
        * CatBoost AUC/F1
        * LGBM AUC/F1
        * Stacked ensemble AUC/F1
    """

    # ---- Train / Test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train positives: {y_train.sum()} ({y_train.mean():.4f})")

    # Class weights (imbalance)
    pos_rate = y_train.mean()
    neg_rate = 1 - pos_rate
    # class_weights for [0, 1]
    class_weights_cb = [1.0, neg_rate / pos_rate]
    class_weight_lgbm = {0: 1.0, 1: neg_rate / pos_rate}
    print(f"Using class weight ratio ~ {neg_rate / pos_rate:.3f}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # OOF predictions: (n_train, n_models)
    n_train = X_train.shape[0]
    oof_cat = np.zeros(n_train)
    oof_lgb = np.zeros(n_train)

    # ---- 3.1 Generate OOF preds ----
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        cb_model = build_catboost(class_weights=class_weights_cb)
        cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof_cat[val_idx] = cb_model.predict_proba(X_val)[:, 1]

        lgb_model = build_lgbm(class_weight=class_weight_lgbm)
        lgb_model.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:, 1]

        auc_cb = roc_auc_score(y_val, oof_cat[val_idx])
        auc_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
        print(f"Fold {fold} - CB AUC: {auc_cb:.4f}, LGBM AUC: {auc_lgb:.4f}")

    # Stack OOF preds into meta-features
    X_meta_train = np.vstack([oof_cat, oof_lgb]).T

    # ---- 3.2 Train meta model ----
    meta_model = build_meta_model()
    meta_model.fit(X_meta_train, y_train)

    # ---- 3.3 Retrain base models on FULL train ----
    cb_full = build_catboost(class_weights=class_weights_cb)
    cb_full.fit(X_train, y_train)

    lgb_full = build_lgbm(class_weight=class_weight_lgbm)
    lgb_full.fit(X_train, y_train)

    # ---- 3.4 Evaluate on TEST ----
    proba_cb_test = cb_full.predict_proba(X_test)[:, 1]
    proba_lgb_test = lgb_full.predict_proba(X_test)[:, 1]

    # Stacked probas
    X_meta_test = np.vstack([proba_cb_test, proba_lgb_test]).T
    proba_stack_test = meta_model.predict_proba(X_meta_test)[:, 1]

    # Helper: best F1 threshold search
    def best_thr_and_f1(y_true, y_proba):
        best_thr = 0.5
        best_f1 = 0.0
        for thr in np.linspace(0.1, 0.9, 81):
            y_pred = (y_proba >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        return best_thr, best_f1

    # AUC
    auc_cb = roc_auc_score(y_test, proba_cb_test)
    auc_lgb = roc_auc_score(y_test, proba_lgb_test)
    auc_stack = roc_auc_score(y_test, proba_stack_test)

    thr_cb, f1_cb = best_thr_and_f1(y_test, proba_cb_test)
    thr_lgb, f1_lgb = best_thr_and_f1(y_test, proba_lgb_test)
    thr_stack, f1_stack = best_thr_and_f1(y_test, proba_stack_test)

    print("\n===== TEST PERFORMANCE =====")
    print(f"CatBoost   - AUC: {auc_cb:.4f}, best_thr: {thr_cb:.3f}, best_F1: {f1_cb:.4f}")
    print(f"LightGBM   - AUC: {auc_lgb:.4f}, best_thr: {thr_lgb:.3f}, best_F1: {f1_lgb:.4f}")
    print(f"STACKED    - AUC: {auc_stack:.4f}, best_thr: {thr_stack:.3f}, best_F1: {f1_stack:.4f}")

    return {
        "cb_model": cb_full,
        "lgb_model": lgb_full,
        "meta_model": meta_model,
        "test_auc": {
            "cb": auc_cb,
            "lgb": auc_lgb,
            "stack": auc_stack
        },
        "test_f1": {
            "cb": f1_cb,
            "lgb": f1_lgb,
            "stack": f1_stack
        },
        "thresholds": {
            "cb": thr_cb,
            "lgb": thr_lgb,
            "stack": thr_stack
        }
    }


# ========= 4. MAIN =========

def main():
    df = load_data()
    df = add_advanced_features(df)

    feature_cols = get_feature_cols()
    needed_cols = feature_cols + [TARGET_COL]
    df = df[needed_cols].copy()
    df = df.dropna(subset=needed_cols)

    X = df[feature_cols]
    y = df[TARGET_COL]

    print(f"Full data shape: {X.shape}, positives: {y.sum()}, pos_rate: {y.mean():.4f}")

    results = train_stacking_ensemble(X, y, n_splits=5, test_size=0.2, random_state=42)

    print("\nFinal test AUCs:", results["test_auc"])
    print("Final test F1s :", results["test_f1"])
    print("Final thresholds:", results["thresholds"])


if __name__ == "__main__":
    main()
