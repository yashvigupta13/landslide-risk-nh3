#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Test — Computes results for Logistic Regression (Undersampling), standalone

Requires (in the working directory): processed1_truncated_data_with_rainfall_jenks.csv
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

RANDOM_STATE = 42
INPUT_CSV = "processed1_truncated_data_with_rainfall_jenks.csv"

# %% LOAD DATA

df = pd.read_csv(INPUT_CSV)

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print(f"Total rows: {len(df)}")
print("Original Landslide distribution:")
print(df["Landslide"].value_counts())

# %% TOMEK LINKS + RANDOM UNDERSAMPLING (creates the train/test split, in-memory only)

X_full = df.drop(columns=["Landslide"])
y_full = df["Landslide"]

tl = TomekLinks(sampling_strategy="auto", n_jobs=-1)
X_tomek, y_tomek = tl.fit_resample(X_full, y_full)
df_tomek = pd.concat([X_tomek, y_tomek.rename("Landslide")], axis=1)

print(f"\nRows removed by Tomek Links: {len(df) - len(df_tomek)}")

df_ls  = df_tomek[df_tomek["Landslide"] == 2]
df_nls = df_tomek[df_tomek["Landslide"] == 1].sample(
    n=len(df_ls), random_state=1, replace=False
)
df_balanced = pd.concat([df_ls, df_nls]).sample(
    frac=1, random_state=1
).reset_index(drop=True)

print("\nFinal balanced class distribution:")
print(df_balanced["Landslide"].value_counts())

# %% TRAIN/TEST SPLIT

X_final = df_balanced.drop(columns=["Latitude", "Longitude", "Landslide"])
Y_final = df_balanced["Landslide"]

X_train, X_test, y_train, y_test = train_test_split(
    X_final, Y_final,
    test_size=0.30, random_state=1, stratify=Y_final
)
y_train = y_train.values
y_test  = y_test.values

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# %% LOGISTIC REGRESSION
# GridSearch best: C=10, penalty='l2', solver='sag'

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Using verified best parameters: C=10, penalty='l2', solver='sag'")

lr_model = LogisticRegression(
    penalty='l2',
    solver='sag',
    C=10,
    random_state=RANDOM_STATE,
    max_iter=1000
)

start_time = time.time()
lr_model.fit(X_train_scaled, y_train)
elapsed_time = time.time() - start_time

# %% EVALUATE

y_train_pred = lr_model.predict(X_train_scaled)
y_test_pred  = lr_model.predict(X_test_scaled)
y_train_prob = lr_model.predict_proba(X_train_scaled)[:, 1]
y_test_prob  = lr_model.predict_proba(X_test_scaled)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train == 2, y_train_prob)
test_auc  = roc_auc_score(y_test == 2, y_test_prob)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Train AUROC:    {train_auc*100:.2f}%")
print(f"Test AUROC:     {test_auc*100:.2f}%")
print(f"Training time:  {elapsed_time:.2f} seconds")

# %% ROC CURVE

fpr, tpr, _ = roc_curve(y_test == 2, y_test_prob)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression (Undersampling, Quick Test)")
plt.legend()
plt.tight_layout()
plt.savefig("quick_test_logistic_regression_roc.png", dpi=150)
plt.show()

# %% FEATURE COEFFICIENTS (importance proxy)

coefficients = np.abs(lr_model.coef_[0])
coef_table = pd.Series(coefficients, index=X_train.columns).sort_values(ascending=False)

print("\nAbsolute coefficient magnitude (feature importance proxy):")
print(coef_table)

print("\n" + "=" * 70)
print("QUICK TEST COMPLETE")
print("=" * 70)
