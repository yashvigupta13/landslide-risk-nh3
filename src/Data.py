# %% Rainfall imports and folder structure

import os
import argparse
import pandas as pd
import numpy as np
import jenkspy

from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


def compute_jenks(df_proc, df_rain, rain_col="Rainfall_30yr_MAX_CONT", n_classes=6):
    """Compute Jenks natural breaks on rainfall values and attach class labels.

    Returns the updated `df_proc` with a `Rainfall_Jenks` column.
    """
    rainfall = df_rain[rain_col].values
    breaks = jenkspy.jenks_breaks(rainfall, n_classes=n_classes)

    df_proc["Rainfall_Jenks"] = pd.cut(
        rainfall,
        bins=breaks,
        labels=list(range(1, n_classes + 1)),
        include_lowest=True
    ).astype(int)

    return df_proc


def undersample_and_split(df_in, out_prefix="", oversample_out_dir="Train-test-split-oversampling"):
    """Apply Tomek Links, random undersampling, and produce train/test splits.

    Saves CSVs to disk and returns (X_train, X_test, Y_train, Y_test).
    """
    df = df_in.copy()

    # Drop accidental index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Ensure Landslide label exists
    if "Landslide" not in df.columns:
        raise ValueError("Input dataframe must contain a 'Landslide' column")

    print("Initial distribution:\n", df["Landslide"].value_counts())

    # Tomek Links to remove borderline samples
    X_full = df.drop(columns=["Landslide"])
    y_full = df["Landslide"]

    tl = TomekLinks(sampling_strategy="auto", n_jobs=-1)
    X_tomek, y_tomek = tl.fit_resample(X_full, y_full)

    df_tomek = pd.concat([X_tomek, y_tomek.rename("Landslide")], axis=1)

    # Random undersample majority class to match minority
    minority_label = df_tomek["Landslide"].value_counts().idxmin()
    n_min = int(df_tomek["Landslide"].value_counts().min())

    df_min = df_tomek[df_tomek["Landslide"] == minority_label]
    df_maj = df_tomek[df_tomek["Landslide"] != minority_label].sample(
        n=n_min, random_state=1, replace=False
    )

    df_balanced = pd.concat([df_min, df_maj]).sample(frac=1, random_state=1).reset_index(drop=True)

    # Split features / labels; preserve Latitude/Longitude for outputs
    X_final = df_balanced.drop(columns=[c for c in ["Latitude", "Longitude", "Landslide"] if c in df_balanced.columns])
    Y_final = df_balanced["Landslide"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_final, Y_final, test_size=0.30, random_state=1, stratify=Y_final
    )

    # Attach coordinates back to Y outputs for mapping convenience
    coords = df_balanced.loc[Y_train.index, [c for c in ["Latitude", "Longitude"] if c in df_balanced.columns]].reset_index(drop=True)
    Y_train_out = pd.concat([Y_train.reset_index(drop=True), coords], axis=1)

    coords_test = df_balanced.loc[Y_test.index, [c for c in ["Latitude", "Longitude"] if c in df_balanced.columns]].reset_index(drop=True)
    Y_test_out = pd.concat([Y_test.reset_index(drop=True), coords_test], axis=1)

    # Save baseline train/test files
    X_train.to_csv(f"{out_prefix}X_train.csv", index=False)
    X_test.to_csv(f"{out_prefix}X_test.csv", index=False)
    Y_train_out.to_csv(f"{out_prefix}Y_train.csv", index=False)
    Y_test_out.to_csv(f"{out_prefix}Y_test.csv", index=False)

    print("Saved X/Y train/test CSVs.")

    # --- Oversampling (SMOTE) on training set only ---
    os.makedirs(oversample_out_dir, exist_ok=True)

    smote = SMOTE(sampling_strategy="auto", random_state=1, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, Y_train)

    pd.DataFrame(X_train_sm, columns=X_train.columns).to_csv(f"{oversample_out_dir}/X_train_smote.csv", index=False)
    X_test.to_csv(f"{oversample_out_dir}/X_test_smote.csv", index=False)
    pd.DataFrame(y_train_sm, columns=["Landslide"]).to_csv(f"{oversample_out_dir}/Y_train_smote.csv", index=False)
    pd.DataFrame(Y_test).to_csv(f"{oversample_out_dir}/Y_test_smote.csv", index=False)

    print("Saved SMOTE oversampled train/test CSVs in:", oversample_out_dir)

    return X_train, X_test, Y_train, Y_test


def main():
    parser = argparse.ArgumentParser(description="Simplified data prep from processed CSV")
    parser.add_argument("--processed", default="src/processed1_truncated_data_with_rainfall_jenks.csv", help="Processed CSV (default: src/processed1_truncated_data_with_rainfall_jenks.csv)")
    parser.add_argument("--rain", required=False, help="Optional rainfall CSV to compute Jenks classes")
    parser.add_argument("--rain-col", default="Rainfall_30yr_MAX_CONT", help="Rainfall column name in the rainfall CSV")
    parser.add_argument("--out-prefix", default="", help="Prefix for saved X/Y files (default: current dir)")
    args = parser.parse_args()

    proc_path = args.processed
    if not os.path.exists(proc_path):
        raise FileNotFoundError(f"Processed CSV not found: {proc_path}")

    df_proc = pd.read_csv(proc_path)

    # If user supplied a rainfall CSV and the processed df lacks Jenks, compute it
    if args.rain and "Rainfall_Jenks" not in df_proc.columns:
        if not os.path.exists(args.rain):
            raise FileNotFoundError(f"Rainfall CSV not found: {args.rain}")
        df_rain = pd.read_csv(args.rain)
        print("Computing Jenks classes from rainfall CSV...")
        df_proc = compute_jenks(df_proc, df_rain, rain_col=args.rain_col)
        # save an annotated copy next to original
        annotated = os.path.splitext(proc_path)[0] + "_with_rain_jenks.csv"
        df_proc.to_csv(annotated, index=False)
        print("Annotated processed CSV written to:", annotated)

    # Run undersampling / splitting / SMOTE and save outputs
    undersample_and_split(df_proc, out_prefix=args.out_prefix)


if __name__ == "__main__":
    main()

# Note: the previous top-level procedural pipeline (train/test split and
# oversampling) has been removed from top-level execution. Use the
# `undersample_and_split()` function and call this module via the CLI.

# Example:
# python src/Data.py --processed src/processed1_truncated_data_with_rainfall_jenks.csv --rain-col Rainfall_30yr_MAX_CONT