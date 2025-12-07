import pandas as pd
import numpy as np
from pathlib import Path

RAW_500K_PATH = Path("credit_500k.csv")


def report_memory(df: pd.DataFrame, label: str) -> None:
    mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"[MEM] {label}: {mb:.2f} MB")


def main():
    # ---------- Step 0: load ----------
    print("[STEP 0] Loading dataset...")
    df = pd.read_csv(RAW_500K_PATH, low_memory=False)
    print("[INFO] Shape:", df.shape)
    report_memory(df, "raw 500k")

    # ---------- Step 1: columns and dtypes ----------
    print("\n[STEP 1] Dtypes overview")
    print(df.dtypes.value_counts())

    print("\n[STEP 1] First 40 column names:")
    print(list(df.columns[:40]))

    print("\n[STEP 1] Sample rows (head):")
    print(df.head(5))

    # ---------- Step 2: missing values summary ----------
    print("\n[STEP 2] Missing-value summary (top 30 columns by NA fraction):")
    na_frac = df.isna().mean().sort_values(ascending=False)
    print(na_frac.head(30))

    # ---------- Step 3: inspect loan_status ----------
    if "loan_status" in df.columns:
        print("\n[STEP 3] loan_status value counts (full):")
        print(df["loan_status"].value_counts(dropna=False))

        print("\n[STEP 3] loan_status value counts (normalized):")
        print(df["loan_status"].value_counts(normalize=True, dropna=False))
    else:
        print("\n[STEP 3] Column 'loan_status' NOT found in data.")

    # ---------- Step 4: inspect key categorical columns ----------
    # Only show those that exist
    cat_candidates = [
        "term",
        "grade",
        "sub_grade",
        "home_ownership",
        "verification_status",
        "purpose",
        "addr_state",
        "application_type",
        "hardship_flag",
        "debt_settlement_flag",
    ]
    print("\n[STEP 4] Categorical columns value counts (top 10 for each):")
    for col in cat_candidates:
        if col in df.columns:
            print(f"\n[CAT] {col} (top 10):")
            print(df[col].value_counts(dropna=False).head(10))

    # ---------- Step 5: numeric summary on a sample ----------
    # Use a sample to avoid heavy computation on all 1.3M rows
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\n[STEP 5] Number of numeric columns:", len(num_cols))

    if len(num_cols) > 0:
        # Take up to 100k rows for numeric describe
        sample_size = min(100_000, len(df))
        df_sample = df.sample(sample_size, random_state=42)
        print(f"[STEP 5] Numeric describe on sample of {sample_size} rows:")
        print(df_sample[num_cols].describe().T.head(20))  # show first 20 nums
    else:
        print("[STEP 5] No numeric columns detected.")

    # ---------- Step 6 (optional): define a binary target and inspect ----------
    # Run this part once you are happy with the above and want to see
    # how many 'good' vs 'bad' loans you have. This does NOT save anything.
    if "loan_status" in df.columns:
        print("\n[STEP 6] Creating binary target from loan_status (analysis only).")
        status = df["loan_status"].astype(str).str.strip()

        good_status = {"Fully Paid"}
        bad_status = {"Charged Off", "Default"}

        mask_keep = status.isin(good_status | bad_status)
        print(f"[STEP 6] Rows with loan_status in {good_status | bad_status}: "
              f"{mask_keep.sum()} / {len(df)}")

        df_bin = df.loc[mask_keep].copy()
        df_bin["target_default_500k"] = status.loc[mask_keep].isin(bad_status).astype(int)

        print("[STEP 6] target_default_500k distribution:")
        print(df_bin["target_default_500k"].value_counts(normalize=True).rename("share"))

        # Optional: correlation of numeric features with target on a sample
        num_cols_bin = df_bin.select_dtypes(include=[np.number]).columns.tolist()
        if "target_default_500k" in num_cols_bin:
            num_cols_bin.remove("target_default_500k")

        if num_cols_bin:
            sample_size2 = min(100_000, len(df_bin))
            df_bin_sample = df_bin.sample(sample_size2, random_state=42)
            corr = df_bin_sample[num_cols_bin].corrwith(
                df_bin_sample["target_default_500k"]
            ).abs().sort_values(ascending=False)
            print("\n[STEP 6] Top 30 numeric features by absolute correlation with target:")
            print(corr.head(30))
        else:
            print("[STEP 6] No numeric cols for correlation with target.")
    else:
        print("[STEP 6] Skipped (loan_status not present).")

    print("\n[DONE] EDA script finished. No models trained, nothing saved.")


if __name__ == "__main__":
    main()
