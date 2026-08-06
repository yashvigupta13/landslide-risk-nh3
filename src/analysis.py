"""
analysis.py — SHAP feature importance + Jenks-classified susceptibility maps
              + G-Score validation for the top-3 gradient boosting models
              (XGBoost, CatBoost, LightGBM).

@author: yashvigupta
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import shap
import jenkspy

# ============================================================
# CONFIGURATION
# ============================================================

# Choose which resampling strategy's data/params to use.
# Run this script once with each setting to produce both sets of outputs.
SAMPLING_MODE = "undersampling"        # "undersampling"  or  "oversampling"

RANDOM_STATE = 42
assert SAMPLING_MODE in ("undersampling", "oversampling"), \
    "SAMPLING_MODE must be 'undersampling' or 'oversampling'"

MODE_LABEL = SAMPLING_MODE.capitalize()   # "Undersampling" / "Oversampling"
OUT_DIR = "feature"
os.makedirs(OUT_DIR, exist_ok=True)

# --- Data paths + best hyperparameters per mode -----------------------------
# Best params copied verbatim from the GridSearchCV results already produced
# by Undersampling-algos.py / Oversampling-algos.py (results*/best_params_*.json)
# so this script reuses the same tuned models instead of re-searching.

if SAMPLING_MODE == "undersampling":
    X_TRAIN_FILE = "X_train.csv"
    X_TEST_FILE  = "X_test.csv"
    Y_TRAIN_FILE = "Y_train.csv"
    Y_TEST_FILE  = "Y_test.csv"

    # From results/best_params_XGBoost.json / best_params_LightGBM.json
    XGB_PARAMS = dict(
        n_estimators=500, learning_rate=0.04, max_depth=6, gamma=0,
        subsample=0.8, colsample_bytree=0.9, min_child_weight=1,
    )
    LGBM_PARAMS = dict(max_depth=15, feature_fraction=0.6, num_leaves=80)

else:  # oversampling
    X_TRAIN_FILE = "Train-test-split-oversampling/X_train_smote.csv"
    X_TEST_FILE  = "Train-test-split-oversampling/X_test_smote.csv"
    Y_TRAIN_FILE = "Train-test-split-oversampling/Y_train_smote.csv"
    Y_TEST_FILE  = "Train-test-split-oversampling/Y_test_smote.csv"

    # From results_oversampling/best_params_XGBoost.json / best_params_LightGBM.json
    XGB_PARAMS = dict(
        n_estimators=600, learning_rate=0.05, max_depth=7, gamma=1,
        subsample=0.7, colsample_bytree=1.0, min_child_weight=1,
    )
    LGBM_PARAMS = dict(max_depth=14, feature_fraction=0.7, num_leaves=90)

# CatBoost needs no external tuning in either mode (internal optimization)

CAT_PARAMS = dict(iterations=500, depth=6)

print("=" * 80)
print(f"SAMPLING MODE: {MODE_LABEL}")
print("=" * 80)
print(f"XGBoost  params: {XGB_PARAMS}")
print(f"LightGBM params: {LGBM_PARAMS}")
print(f"CatBoost params: {CAT_PARAMS}")

# ============================================================
# LOAD DATA 
# ============================================================
print("\nLoading datasets...")

X_train = pd.read_csv(X_TRAIN_FILE)
X_test  = pd.read_csv(X_TEST_FILE)
y_train_raw = pd.read_csv(Y_TRAIN_FILE)["Landslide"].values
y_test_raw  = pd.read_csv(Y_TEST_FILE)["Landslide"].values

train_features = X_train.columns.tolist()
print("Features:", train_features)
print("Loaded successfully")

y_train = np.where(y_train_raw == 2, 1, 0)
y_test  = np.where(y_test_raw == 2, 1, 0)

# %%
# ============================================================
# XGBOOST
# ============================================================
print("\n" + "=" * 70)
print(f"XGBOOST ({MODE_LABEL}) — using verified best parameters, no GridSearch")
print("=" * 70)

best_xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    **XGB_PARAMS,
)
best_xgb.fit(X_train, y_train)

y_train_pred = best_xgb.predict(X_train)
y_test_pred  = best_xgb.predict(X_test)
y_train_prob = best_xgb.predict_proba(X_train)[:, 1]
y_test_prob  = best_xgb.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train, y_train_prob)
test_auc  = roc_auc_score(y_test, y_test_prob)

print(f"Train Accuracy : {train_acc:.4f}")
print(f"Test Accuracy  : {test_acc:.4f}")
print(f"Train AUROC    : {train_auc*100:.2f}")
print(f"Test AUROC     : {test_auc*100:.2f}")

print("\nComputing SHAP values for XGBoost...")
explainer = shap.Explainer(
    best_xgb.predict_proba,
    X_train,
    algorithm="permutation",
)
shap_values = explainer(X_train)
shap_vals = shap_values.values[:, :, 1]

xgb_fi = pd.Series(
    np.mean(np.abs(shap_vals), axis=0),
    index=X_train.columns,
).sort_values(ascending=False)
xgb_fi = xgb_fi / xgb_fi.sum()

xgb_table = xgb_fi.reset_index()
xgb_table.columns = ["Variables", "XGBoost"]
xgb_table.to_csv(f"{OUT_DIR}/xgb_shap_feature_importance_{SAMPLING_MODE}.csv", index=False)

plt.figure(figsize=(7, 5))
sorted_fi = xgb_fi.sort_values()
plt.barh(sorted_fi.index, sorted_fi.values, color="darkgreen")
plt.xlabel("mean(|SHAP value|)")
plt.title(f"Feature contribution in XGBoost ({MODE_LABEL})")
for i, v in enumerate(sorted_fi.values):
    plt.text(v + 0.003, i, f"+{v:.2f}", va="center", fontsize=10,
              fontweight="bold", color="darkgreen")
plt.xlim(0, sorted_fi.max() * 1.15)
plt.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/xgb_shap_feature_importance_{SAMPLING_MODE}.png", dpi=600)
plt.show()

# %%
# ============================================================
# CATBOOST
# ============================================================
print("\n" + "=" * 70)
print(f"CATBOOST ({MODE_LABEL}) — internal optimization, no external tuning")
print("=" * 70)

cat_model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    verbose=False,
    random_state=RANDOM_STATE,
    **CAT_PARAMS,
)
cat_model.fit(X_train, y_train)

y_train_pred = cat_model.predict(X_train)
y_test_pred  = cat_model.predict(X_test)
y_train_prob = cat_model.predict_proba(X_train)[:, 1]
y_test_prob  = cat_model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train, y_train_prob)
test_auc  = roc_auc_score(y_test, y_test_prob)

print("\n===== CATBOOST RESULTS =====")
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Test Accuracy  : {test_acc:.4f}")
print(f"Train AUROC    : {train_auc*100:.2f}")
print(f"Test AUROC     : {test_auc*100:.2f}")

explainer = shap.TreeExplainer(cat_model)
shap_values = explainer.shap_values(X_train)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

cat_fi = pd.Series(
    np.mean(np.abs(shap_values), axis=0),
    index=X_train.columns,
).sort_values(ascending=False)
cat_fi = cat_fi / cat_fi.sum()

cat_table = cat_fi.reset_index()
cat_table.columns = ["Variables", "CatBoost"]
cat_table.to_csv(f"{OUT_DIR}/catboost_shap_feature_importance_{SAMPLING_MODE}.csv", index=False)

plt.figure(figsize=(7, 5))
sorted_fi = cat_fi.sort_values()
plt.barh(sorted_fi.index, sorted_fi.values, color="darkgreen")
plt.xlabel("mean(|SHAP value|)")
plt.title(f"Feature contribution in CatBoost ({MODE_LABEL})")
for i, v in enumerate(sorted_fi.values):
    plt.text(v + 0.003, i, f"+{v:.2f}", va="center", fontsize=10,
              fontweight="bold", color="darkgreen")
plt.xlim(0, sorted_fi.max() * 1.15)
plt.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/catboost_shap_feature_importance_{SAMPLING_MODE}.png", dpi=600)
plt.show()

# %%
# ============================================================
# LIGHTGBM
# ============================================================
print("\n" + "=" * 70)
print(f"LIGHTGBM ({MODE_LABEL}) — using verified best parameters, no GridSearch")
print("=" * 70)

best_lgbm = LGBMClassifier(
    objective="binary",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
    **LGBM_PARAMS,
)
best_lgbm.fit(X_train, y_train)

y_train_pred = best_lgbm.predict(X_train)
y_test_pred  = best_lgbm.predict(X_test)
y_train_prob = best_lgbm.predict_proba(X_train)[:, 1]
y_test_prob  = best_lgbm.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train, y_train_prob)
test_auc  = roc_auc_score(y_test, y_test_prob)

print("\n===== LIGHTGBM RESULTS =====")
print(f"Train Accuracy : {train_acc:.4f}")
print(f"Test Accuracy  : {test_acc:.4f}")
print(f"Train AUROC    : {train_auc*100:.2f}")
print(f"Test AUROC     : {test_auc*100:.2f}")

print("\nComputing SHAP values for LightGBM...")
explainer = shap.TreeExplainer(best_lgbm)
shap_values = explainer.shap_values(X_train)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

lgbm_fi = pd.Series(
    np.mean(np.abs(shap_values), axis=0),
    index=X_train.columns,
).sort_values(ascending=False)
lgbm_fi = lgbm_fi / lgbm_fi.sum()

lgbm_table = lgbm_fi.reset_index()
lgbm_table.columns = ["Variables", "LightGBM"]
lgbm_table.to_csv(f"{OUT_DIR}/lightgbm_shap_feature_importance_{SAMPLING_MODE}.csv", index=False)
print("\nLightGBM SHAP Feature Importance:")
print(lgbm_table)

plt.figure(figsize=(7, 5))
sorted_fi = lgbm_fi.sort_values()
plt.barh(sorted_fi.index, sorted_fi.values, color="darkgreen")
plt.xlabel("mean(|SHAP value|)")
plt.title(f"Feature contribution in LightGBM ({MODE_LABEL})")
for i, v in enumerate(sorted_fi.values):
    plt.text(v + 0.003, i, f"+{v:.2f}", va="center", fontsize=10,
              fontweight="bold", color="darkgreen")
plt.xlim(0, sorted_fi.max() * 1.15)
plt.grid(axis="x", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/lightgbm_shap_feature_importance_{SAMPLING_MODE}.png", dpi=600)
plt.show()

# %%
# ============================================================
# G-SCORE CALCULATION (reuses the exact same fitted models above —
# no re-training with different hyperparameters)
# ============================================================

print("\n" + "=" * 80)
print(f"LANDSLIDE SUSCEPTIBILITY - G-SCORE CALCULATION ({MODE_LABEL.upper()})")
print("=" * 80)
print("\nG-Score Formula: G = (a/A) x [1 - (b-a)/(B-A)]^(1/3)")
print("  where:")
print("    a = Landslide points in high/very high zones (classes 4 & 5)")
print("    A = Total landslide points")
print("    b = Total points in high/very high zones")
print("    B = Total points in study area")

# %% GENERATE PROBABILITY MAPS OVER FULL STUDY AREA

full_df = pd.read_csv("processed1_truncated_data_with_rainfall_jenks.csv")
coords = full_df[["Latitude", "Longitude"]].copy()
X_full = full_df[train_features].copy()

print(f"\nFull dataset shape: {X_full.shape}")
print("\n" + "=" * 80)
print("GENERATING PROBABILITY MAPS FOR TOP 3 MODELS")
print("=" * 80)


def save_probabilities(model, model_name):
    prob = model.predict_proba(X_full)[:, 1]
    df = coords.copy()
    df["LSM_Probability"] = prob
    path = f"{OUT_DIR}/LSM_{model_name}_{MODE_LABEL}_Probabilities.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}  [range {prob.min():.4f} - {prob.max():.4f}]")
    return path


lgbm_prob_path = save_probabilities(best_lgbm, "LightGBM")
cat_prob_path  = save_probabilities(cat_model, "CatBoost")
xgb_prob_path  = save_probabilities(best_xgb, "XGBoost")

# %% APPLY JENKS CLASSIFICATION (5 CLASSES)

print("\n" + "=" * 80)
print("APPLYING JENKS NATURAL BREAKS CLASSIFICATION")
print("=" * 80)


def apply_jenks(input_csv, output_csv, model_name):
    """Apply Jenks Natural Breaks to classify probabilities into 5 classes"""
    df = pd.read_csv(input_csv)
    values = df["LSM_Probability"].values

    print(f"\n{model_name}:")
    print(f"  Computing Jenks breaks for {len(values):,} points...")

    breaks = jenkspy.jenks_breaks(values, n_classes=5)
    print("  Break points:")
    for i, (low, high) in enumerate(zip(breaks[:-1], breaks[1:]), 1):
        print(f"    Class {i}: {low:.4f} -> {high:.4f}")

    df["LSM_Class"] = pd.cut(
        values, bins=breaks, labels=[1, 2, 3, 4, 5], include_lowest=True
    ).astype(int)

    class_counts = df["LSM_Class"].value_counts().sort_index()
    print("  Class distribution:")
    for cls in range(1, 6):
        count = class_counts.get(cls, 0)
        pct = (count / len(df)) * 100
        susceptibility = ["Very Low", "Low", "Moderate", "High", "Very High"][cls - 1]
        print(f"    Class {cls} ({susceptibility:>10}): {count:>8,} ({pct:>5.2f}%)")

    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    return breaks


lgbm_jenks_path = f"{OUT_DIR}/LSM_LightGBM_{MODE_LABEL}_Probabilities_Jenks.csv"
cat_jenks_path  = f"{OUT_DIR}/LSM_CatBoost_{MODE_LABEL}_Probabilities_Jenks.csv"
xgb_jenks_path  = f"{OUT_DIR}/LSM_XGBoost_{MODE_LABEL}_Probabilities_Jenks.csv"

lgbm_breaks = apply_jenks(lgbm_prob_path, lgbm_jenks_path, "LightGBM")
cat_breaks  = apply_jenks(cat_prob_path, cat_jenks_path, "CatBoost")
xgb_breaks  = apply_jenks(xgb_prob_path, xgb_jenks_path, "XGBoost")

# %% G-SCORE CALCULATION


def compute_g_score(jenks_csv, model_name):
    """
    Compute G-score for a landslide susceptibility map.

    G-score measures the model's ability to concentrate landslide occurrences
    in high/very high susceptibility zones while minimizing false positives.
    """
    df = pd.read_csv(jenks_csv)

    inv_df = pd.read_csv("processed1_truncated_data_with_rainfall_jenks.csv")[
        ["Latitude", "Longitude", "Landslide"]
    ]

    for d in (df, inv_df):
        d["Latitude"] = d["Latitude"].round(6)
        d["Longitude"] = d["Longitude"].round(6)

    merged = df.merge(inv_df, on=["Latitude", "Longitude"], how="inner")

    print(f"\n{'='*80}")
    print(f"{model_name} - G-SCORE ANALYSIS ({MODE_LABEL})")
    print(f"{'='*80}")

    danger_mask = merged["LSM_Class"].isin([4, 5])

    A = (merged["Landslide"] == 2).sum()
    B = len(merged)
    a = ((merged["Landslide"] == 2) & danger_mask).sum()
    b = danger_mask.sum()

    success_rate = a / A if A > 0 else 0
    false_positive_density = (b - a) / (B - A) if (B - A) > 0 else 0
    G = success_rate * np.power(1 - false_positive_density, 1 / 3) if false_positive_density < 1 else 0

    print("\nInput Data:")
    print(f"  Total points in study area (B): {B:,}")
    print(f"  Total landslide points (A):     {A:,}")
    print(f"  Points in danger zones (b):     {b:,} ({b/B*100:.2f}% of area)")
    print(f"  Landslides in danger zones (a): {a:,} ({a/A*100:.2f}% of landslides)")

    print("\nG-Score Components:")
    print(f"  Success Rate (a/A):               {success_rate:.4f} ({success_rate*100:.2f}%)")
    print(f"  False Positive Density (b-a/B-A): {false_positive_density:.4f}")
    print(f"  Danger Zone Coverage (b/B):       {b/B:.4f} ({b/B*100:.2f}%)")

    print(f"\n{'*'*80}")
    print(f"  FINAL G-SCORE: {G:.6f}")
    print(f"{'*'*80}")


    return {
        "model": model_name,
        "G_score": G,
        "success_rate": success_rate,
        "danger_coverage": b / B,
        "false_positive_density": false_positive_density,
        "total_landslides": A,
        "captured_landslides": a,
        "danger_zone_size": b,
        "study_area_size": B,
    }


models = [
    (lgbm_jenks_path, "LightGBM"),
    (cat_jenks_path, "CatBoost"),
    (xgb_jenks_path, "XGBoost"),
]

g_score_results = [compute_g_score(csv, name) for csv, name in models]

# %% SUMMARY COMPARISON

print("\n" + "=" * 80)
print("COMPARATIVE SUMMARY")
print("=" * 80)

summary_df = pd.DataFrame(g_score_results)
summary_df = summary_df.sort_values("G_score", ascending=False).reset_index(drop=True)

print("\nRanking by G-Score:")
print("-" * 80)
print(f"{'Rank':<6} {'Model':<15} {'G-Score':<12} {'Success Rate':<15} {'Area Coverage':<15}")
print("-" * 80)
for idx, row in summary_df.iterrows():
    print(f"{idx+1:<6} {row['model']:<15} {row['G_score']:<12.6f} "
          f"{row['success_rate']*100:>6.2f}%{'':<8} {row['danger_coverage']*100:>6.2f}%")
print("-" * 80)

summary_path = f"{OUT_DIR}/G_Score_Summary_{MODE_LABEL}.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary saved to: {summary_path}")

print("\n" + "=" * 80)
print(f"G-SCORE ANALYSIS COMPLETE ({MODE_LABEL})")
print("=" * 80)
print("\nOutput files generated (all under 'feature/'):")
for p in [
    f"xgb_shap_feature_importance_{SAMPLING_MODE}.csv/.png",
    f"catboost_shap_feature_importance_{SAMPLING_MODE}.csv/.png",
    f"lightgbm_shap_feature_importance_{SAMPLING_MODE}.csv/.png",
    lgbm_prob_path, cat_prob_path, xgb_prob_path,
    lgbm_jenks_path, cat_jenks_path, xgb_jenks_path,
    summary_path,
]:
    print(f"  - {p}")
print("=" * 80)
