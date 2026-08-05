# -*- coding: utf-8 -*-
"""
Created on Mon Mar 9 14:59:17 2026
@author: yashvigupta
"""
# %% Average rainfall imports and folder structure
import os
import re
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import subprocess
import jenkspy

from rasterio.transform import from_origin
from rasterio.warp import reproject, calculate_default_transform
from rasterio.enums import Resampling
from collections import defaultdict
from rasterio.transform import rowcol
from shapely.geometry import box, Polygon
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split

# %% Tomek Links + Random Undersampling
# Undersampling pipeline: first remove borderline samples with Tomek Links,
# Then randomly undersample the majority class to match the minority.
print("\n===== TOMEK LINKS + UNDERSAMPLING =====\n")

CSV_IN = "processed1_truncated_data_with_rainfall_jenks.csv"

df = pd.read_csv(CSV_IN)

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print("Dataset loaded")
print("Total rows:", len(df))
print("\nOriginal Landslide distribution:")
print(df["Landslide"].value_counts())

print("\nApplying Tomek Links undersampling...")

X_full = df.drop(columns=["Landslide"])
y_full = df["Landslide"]

tl = TomekLinks(sampling_strategy="auto", n_jobs=-1)
X_tomek, y_tomek = tl.fit_resample(X_full, y_full)

df_tomek = pd.concat([X_tomek, y_tomek.rename("Landslide")], axis=1)

print("\nAfter Tomek Links:")
print(df_tomek["Landslide"].value_counts())
print("Rows removed by Tomek:", len(df) - len(df_tomek))

print("\nApplying random undersampling to balance classes...")

df_ls  = df_tomek[df_tomek["Landslide"] == 2]
df_nls = df_tomek[df_tomek["Landslide"] == 1].sample(
    n=len(df_ls), random_state=1, replace=False
)

df_balanced = pd.concat([df_ls, df_nls]).sample(
    frac=1, random_state=1
).reset_index(drop=True)

print("\nFinal balanced class distribution:")
print(df_balanced["Landslide"].value_counts())

X_final = df_balanced.drop(columns=["Latitude", "Longitude", "Landslide"])
Y_final = df_balanced["Landslide"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X_final, Y_final,
    test_size=0.30, random_state=1, stratify=Y_final
)

Y_train_out = pd.concat([
    Y_train.reset_index(drop=True),
    df_balanced.loc[Y_train.index, ["Latitude", "Longitude"]].reset_index(drop=True)
], axis=1)

Y_test_out = pd.concat([
    Y_test.reset_index(drop=True),
    df_balanced.loc[Y_test.index, ["Latitude", "Longitude"]].reset_index(drop=True)
], axis=1)

print("\nNaN checks:")
print("X_train:", X_train.isna().values.any())
print("Y_train:", Y_train_out.isna().values.any())
print("X_test :", X_test.isna().values.any())
print("Y_test :", Y_test_out.isna().values.any())

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
Y_train_out.to_csv("Y_train.csv", index=False)
Y_test_out.to_csv("Y_test.csv", index=False)

print("\n✓ Files written: X_train.csv  X_test.csv  Y_train.csv  Y_test.csv")

# %% OVERSAMPLING (SMOTE)
# (unchanged — operates on features only)

CSV_IN  = "processed1_truncated_data_with_rainfall_jenks.csv"
OUT_DIR = "Train-test-split-oversampling"

os.makedirs(OUT_DIR, exist_ok=True)

print("\n" + "="*80)
print("OVERSAMPLING: LOADING DATA")
print("="*80)

df = pd.read_csv(CSV_IN)

if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print("Total rows:", len(df))
print("\nOriginal Landslide distribution:")
print(df["Landslide"].value_counts())

if "Rainfall_Jenks" in df.columns:
    print("\nRainfall Jenks distribution:")
    print(df["Rainfall_Jenks"].value_counts().sort_index())

print("\nNaN check (entire dataset):", df.isna().sum().sum())

coords = df[["Latitude", "Longitude"]].copy()
X = df.drop(columns=["Latitude", "Longitude", "Landslide"])
y = df["Landslide"]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\n" + "="*80)
print("TRAIN–TEST SPLIT (ORIGINAL IMBALANCED DATA)")
print("="*80)

X_train_orig, X_test, y_train_orig, y_test = train_test_split(
    X, y, test_size=0.30, random_state=1, stratify=y
)

print("\nTrain distribution (BEFORE SMOTE):")
print(y_train_orig.value_counts())
print("\nTest distribution (UNCHANGED):")
print(y_test.value_counts())

print("\n" + "="*80)
print("APPLYING SMOTE (TRAIN ONLY)")
print("="*80)

smote = SMOTE(sampling_strategy="auto", random_state=1, k_neighbors=5)
X_train, y_train = smote.fit_resample(X_train_orig, y_train_orig)

print("\nTrain distribution (AFTER SMOTE):")
print(pd.Series(y_train).value_counts())
print("Synthetic samples added:", len(X_train) - len(X_train_orig))

print("\nNaN checks:")
print("X_train:", X_train.isna().values.any())
print("X_test :", X_test.isna().values.any())
print("y_train:", pd.Series(y_train).isna().any())
print("y_test :", y_test.isna().any())

assert list(X_train.columns) == list(X_test.columns), "Feature mismatch!"
print("✓ Feature alignment confirmed")

Y_train = pd.DataFrame(y_train, columns=["Landslide"])
Y_test  = pd.DataFrame(y_test).reset_index(drop=True)

X_train.to_csv(f"{OUT_DIR}/X_train_smote.csv", index=False)
X_test.to_csv(f"{OUT_DIR}/X_test_smote.csv",  index=False)
Y_train.to_csv(f"{OUT_DIR}/Y_train_smote.csv", index=False)
Y_test.to_csv(f"{OUT_DIR}/Y_test_smote.csv",  index=False)

print("\n✓ Files written:")
print("  X_train_smote.csv  X_test_smote.csv  Y_train_smote.csv  Y_test_smote.csv")

print("\n" + "="*80)
print("OVERSAMPLING PIPELINE COMPLETE")
print("="*80)
