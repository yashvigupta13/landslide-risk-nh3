#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oversampling algorithms - OPTIMIZED PARAMETERS
Parameters updated to match actual GridSearchCV results from auroc_summary.csv

@author: athangyawalkar
"""

# %% IMPORTS AND SETUP

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

rcParams['figure.figsize'] = (10, 8)
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10

os.makedirs('results_oversampling', exist_ok=True)
os.makedirs('figures_oversampling', exist_ok=True)

# Global configuration
USE_OPTIMIZED_PARAMS = True   # ← SET TO TRUE: using verified GridSearch results
RANDOM_STATE = 42

print("="*70)
print("LANDSLIDE SUSCEPTIBILITY MAPPING - OVERSAMPLING")
print("MODE: OPTIMIZED PARAMETERS (from GridSearchCV auroc_summary.csv)")
print("="*70)

# %% LOAD DATA

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

X_train = pd.read_csv("Train-test-split-oversampling/X_train_smote.csv")
X_test  = pd.read_csv("Train-test-split-oversampling/X_test_smote.csv")
y_train = pd.read_csv("Train-test-split-oversampling/Y_train_smote.csv")["Landslide"].values
y_test  = pd.read_csv("Train-test-split-oversampling/Y_test_smote.csv")["Landslide"].values

full_df = pd.read_csv("processed1_truncated_data_with_rainfall_jenks.csv")
train_features = X_train.columns.tolist()

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of features: {len(train_features)}")
print(f"Training features: {train_features}")

unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test,  counts_test  = np.unique(y_test,  return_counts=True)

print(f"\nTraining set class distribution:")
for cls, cnt in zip(unique_train, counts_train):
    print(f"  Class {cls}: {cnt} ({cnt/len(y_train)*100:.1f}%)")
print(f"\nTest set class distribution:")
for cls, cnt in zip(unique_test, counts_test):
    print(f"  Class {cls}: {cnt} ({cnt/len(y_test)*100:.1f}%)")

# %% INITIALIZE RESULTS STORAGE

results_dict = {
    'model': [], 'train_accuracy': [], 'test_accuracy': [],
    'train_auroc': [], 'test_auroc': [], 'training_time': [], 'best_params': []
}
roc_data = {}
feature_importance_dict = {}

# %% LOGISTIC REGRESSION
# GridSearch best: C=10, penalty='l2', solver='lbfgs'
# FIX: original code had solver='sag' — corrected to 'lbfgs'

print("\n" + "="*70)
print("LOGISTIC REGRESSION")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters from GridSearchCV results...")
    print("Best params: C=10, penalty='l2', solver='lbfgs'")

    lr_model = LogisticRegression(
        penalty='l2',
        solver='lbfgs',       # ← FIXED: was 'sag'
        C=10,
        random_state=RANDOM_STATE,
        max_iter=1000
    )

    start_time = time.time()
    lr_model.fit(X_train_scaled, y_train)
    elapsed_time = time.time() - start_time

    best_params = {'C': 10, 'penalty': 'l2', 'solver': 'lbfgs'}

else:
    print("Running FOCUSED GridSearchCV...")
    print("Undersampling best: C=10, solver='sag'")
    param_grid = {
        'penalty': ['l2'],
        'solver': ['sag', 'saga', 'lbfgs'],
        'C': [5, 10, 15, 20, 25]
    }
    print(f"Grid combinations: {3 * 5} = 15")
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    grid.fit(X_train_scaled, y_train)
    elapsed_time = time.time() - start_time
    lr_model = grid.best_estimator_
    best_params = grid.best_params_

y_train_pred = lr_model.predict(X_train_scaled)
y_test_pred  = lr_model.predict(X_test_scaled)
y_train_prob = lr_model.predict_proba(X_train_scaled)[:, 1]
y_test_prob  = lr_model.predict_proba(X_test_scaled)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train == 2, y_train_prob)
test_auc  = roc_auc_score(y_test == 2, y_test_prob)

results_dict['model'].append('Logistic Regression')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

fpr, tpr, _ = roc_curve(y_test == 2, y_test_prob)
roc_data['Logistic Regression'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

coefficients = np.abs(lr_model.coef_[0])
feature_importance_dict['Logistic Regression'] = pd.DataFrame({
    'Feature': train_features, 'Importance': coefficients
}).sort_values('Importance', ascending=False)

with open('results_oversampling/best_params_LogisticRegression.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Train AUROC:    {train_auc*100:.2f}%")
print(f"Test AUROC:     {test_auc*100:.2f}%")
print(f"Training time:  {elapsed_time:.2f} seconds")
print(f"Best params:    {best_params}")

# %% DECISION TREE
# GridSearch best: criterion='entropy', max_depth=10, min_samples_leaf=7, min_samples_split=2
# FIX: original code had criterion='gini', min_samples_leaf=8

print("\n" + "="*70)
print("DECISION TREE")
print("="*70)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters from GridSearchCV results...")
    print("Best params: criterion='entropy', max_depth=10, min_samples_leaf=7, min_samples_split=2")

    dt_model = DecisionTreeClassifier(
        criterion='entropy',      # ← FIXED: was 'gini'
        max_depth=10,
        min_samples_leaf=7,       # ← FIXED: was 8
        min_samples_split=2,
        random_state=RANDOM_STATE
    )

    start_time = time.time()
    dt_model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    best_params = {
        'criterion': 'entropy',
        'max_depth': 10,
        'min_samples_leaf': 7,
        'min_samples_split': 2
    }

else:
    print("Running FOCUSED GridSearchCV...")
    print("Undersampling best: gini, depth=10, leaf=8, split=2")
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [8, 9, 10],
        'min_samples_leaf': [6, 7, 8],
        'min_samples_split': [2, 3, 4]
    }
    print(f"Grid combinations: {2 * 3 * 3 * 3} = 54")
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    grid.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    dt_model = grid.best_estimator_
    best_params = grid.best_params_

y_train_pred = dt_model.predict(X_train)
y_test_pred  = dt_model.predict(X_test)
y_train_prob = dt_model.predict_proba(X_train)[:, 1]
y_test_prob  = dt_model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train == 2, y_train_prob)
test_auc  = roc_auc_score(y_test == 2, y_test_prob)

results_dict['model'].append('Decision Tree')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

fpr, tpr, _ = roc_curve(y_test == 2, y_test_prob)
roc_data['Decision Tree'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

feature_importance_dict['Decision Tree'] = pd.DataFrame({
    'Feature': train_features, 'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

with open('results_oversampling/best_params_DecisionTree.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Train AUROC:    {train_auc*100:.2f}%")
print(f"Test AUROC:     {test_auc*100:.2f}%")
print(f"Training time:  {elapsed_time:.2f} seconds")
print(f"Best params:    {best_params}")

# %% RANDOM FOREST
# GridSearch best: max_depth=12, min_samples_leaf=2, min_samples_split=2, n_estimators=250
# FIX: original code had max_depth=10, min_samples_leaf=1

print("\n" + "="*70)
print("RANDOM FOREST")
print("="*70)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters from GridSearchCV results...")
    print("Best params: max_depth=12, min_samples_leaf=2, min_samples_split=2, n_estimators=250")

    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=12,             # ← FIXED: was 10
        min_samples_leaf=2,       # ← FIXED: was 1
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    start_time = time.time()
    rf_model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    best_params = {
        'max_depth': 12,
        'min_samples_leaf': 2,
        'min_samples_split': 2,
        'n_estimators': 250
    }

else:
    print("Running FOCUSED GridSearchCV...")
    print("Undersampling best: n_est=250, depth=10, leaf=1, split=2")
    param_grid = {
        'n_estimators': [200, 250, 300],
        'max_depth': [8, 10, 12],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 3]
    }
    print(f"Grid combinations: {3 * 3 * 2 * 2} = 36")
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    grid.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    rf_model = grid.best_estimator_
    best_params = grid.best_params_

y_train_pred = rf_model.predict(X_train)
y_test_pred  = rf_model.predict(X_test)
y_train_prob = rf_model.predict_proba(X_train)[:, 1]
y_test_prob  = rf_model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train == 2, y_train_prob)
test_auc  = roc_auc_score(y_test == 2, y_test_prob)

results_dict['model'].append('Random Forest')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

fpr, tpr, _ = roc_curve(y_test == 2, y_test_prob)
roc_data['Random Forest'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

feature_importance_dict['Random Forest'] = pd.DataFrame({
    'Feature': train_features, 'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

with open('results_oversampling/best_params_RandomForest.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Train AUROC:    {train_auc*100:.2f}%")
print(f"Test AUROC:     {test_auc*100:.2f}%")
print(f"Training time:  {elapsed_time:.2f} seconds")
print(f"Best params:    {best_params}")

# %% ADABOOST
# GridSearch best: learning_rate=1.9, n_estimators=600
# FIX: original code had learning_rate=1.8, n_estimators=500

print("\n" + "="*70)
print("ADABOOST")
print("="*70)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters from GridSearchCV results...")
    print("Best params: learning_rate=1.9, n_estimators=600")

    base_dt = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)

    ada_model = AdaBoostClassifier(
        estimator=base_dt,
        n_estimators=600,         # ← FIXED: was 500
        learning_rate=1.9,        # ← FIXED: was 1.8
        random_state=RANDOM_STATE
    )

    start_time = time.time()
    ada_model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    best_params = {'learning_rate': 1.9, 'n_estimators': 600}

else:
    print("Running FOCUSED GridSearchCV...")
    print("Undersampling best: n_est=500, lr=1.8")
    param_grid = {
        'n_estimators': [400, 500, 600],
        'learning_rate': [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    }
    print(f"Grid combinations: {3 * 6} = 18")
    grid = GridSearchCV(
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            random_state=RANDOM_STATE
        ),
        param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    grid.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    ada_model = grid.best_estimator_
    best_params = grid.best_params_

y_train_pred = ada_model.predict(X_train)
y_test_pred  = ada_model.predict(X_test)
y_train_prob = ada_model.predict_proba(X_train)[:, 1]
y_test_prob  = ada_model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train == 2, y_train_prob)
test_auc  = roc_auc_score(y_test == 2, y_test_prob)

results_dict['model'].append('AdaBoost')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

fpr, tpr, _ = roc_curve(y_test == 2, y_test_prob)
roc_data['AdaBoost'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

feature_importance_dict['AdaBoost'] = pd.DataFrame({
    'Feature': train_features, 'Importance': ada_model.feature_importances_
}).sort_values('Importance', ascending=False)

with open('results_oversampling/best_params_AdaBoost.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Train AUROC:    {train_auc*100:.2f}%")
print(f"Test AUROC:     {test_auc*100:.2f}%")
print(f"Training time:  {elapsed_time:.2f} seconds")
print(f"Best params:    {best_params}")

# %% EXTRA TREES
# GridSearch best: criterion='gini', max_depth=12, min_samples_leaf=2, min_samples_split=2, n_estimators=500
# FIX: original code had max_depth=10, min_samples_leaf=1

print("\n" + "="*70)
print("EXTRA TREES")
print("="*70)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters from GridSearchCV results...")
    print("Best params: criterion='gini', max_depth=12, min_samples_leaf=2, min_samples_split=2, n_estimators=500")

    et_model = ExtraTreesClassifier(
        n_estimators=500,
        criterion='gini',
        max_depth=12,             # ← FIXED: was 10
        min_samples_leaf=2,       # ← FIXED: was 1
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    start_time = time.time()
    et_model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    best_params = {
        'criterion': 'gini',
        'max_depth': 12,
        'min_samples_leaf': 2,
        'min_samples_split': 2,
        'n_estimators': 500
    }

else:
    print("Running FOCUSED GridSearchCV...")
    print("Undersampling best: n_est=500, gini, depth=10, leaf=1, split=2")
    param_grid = {
        'n_estimators': [400, 500],
        'max_depth': [8, 10, 12],
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 3]
    }
    print(f"Grid combinations: {2 * 3 * 2 * 2 * 2} = 48")
    grid = GridSearchCV(
        ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    grid.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    et_model = grid.best_estimator_
    best_params = grid.best_params_

y_train_pred = et_model.predict(X_train)
y_test_pred  = et_model.predict(X_test)
y_train_prob = et_model.predict_proba(X_train)[:, 1]
y_test_prob  = et_model.predict_proba(X_test)[:, 1]

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train == 2, y_train_prob)
test_auc  = roc_auc_score(y_test == 2, y_test_prob)

results_dict['model'].append('Extra Trees')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

fpr, tpr, _ = roc_curve(y_test == 2, y_test_prob)
roc_data['Extra Trees'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

feature_importance_dict['Extra Trees'] = pd.DataFrame({
    'Feature': train_features, 'Importance': et_model.feature_importances_
}).sort_values('Importance', ascending=False)

with open('results_oversampling/best_params_ExtraTrees.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Train AUROC:    {train_auc*100:.2f}%")
print(f"Test AUROC:     {test_auc*100:.2f}%")
print(f"Training time:  {elapsed_time:.2f} seconds")
print(f"Best params:    {best_params}")

# %% XGBOOST
# GridSearch best: colsample_bytree=1.0, gamma=1, learning_rate=0.05,
#                  max_depth=7, min_child_weight=1, n_estimators=600, subsample=0.7
# FIX: original code had n_est=500, lr=0.04, depth=6, gamma=0, subsample=0.8, colsample=0.9

print("\n" + "="*70)
print("XGBOOST")
print("="*70)

y_train_bin = np.where(y_train == 2, 1, 0)
y_test_bin  = np.where(y_test == 2, 1, 0)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters from GridSearchCV results...")
    print("Best params: colsample_bytree=1.0, gamma=1, learning_rate=0.05,")
    print("             max_depth=7, min_child_weight=1, n_estimators=600, subsample=0.7")

    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=600,         # ← FIXED: was 500
        learning_rate=0.05,       # ← FIXED: was 0.04
        max_depth=7,              # ← FIXED: was 6
        gamma=1,                  # ← FIXED: was 0
        subsample=0.7,            # ← FIXED: was 0.8
        colsample_bytree=1.0,     # ← FIXED: was 0.9
        min_child_weight=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    start_time = time.time()
    xgb_model.fit(X_train, y_train_bin)
    elapsed_time = time.time() - start_time

    best_params = {
        'colsample_bytree': 1.0,
        'gamma': 1,
        'learning_rate': 0.05,
        'max_depth': 7,
        'min_child_weight': 1,
        'n_estimators': 600,
        'subsample': 0.7
    }

else:
    print("Running FOCUSED GridSearchCV...")
    print("Undersampling best: n_est=500, lr=0.04, depth=6, gamma=0, subsample=0.8, colsample=0.9, child_wt=1")
    param_grid = {
        'n_estimators': [400, 500, 600],
        'learning_rate': [0.03, 0.04, 0.05],
        'max_depth': [5, 6, 7],
        'gamma': [0, 1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 2]
    }
    print(f"Grid combinations: {3 * 3 * 3 * 2 * 3 * 3 * 2} = 972")
    grid = GridSearchCV(
        XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    grid.fit(X_train, y_train_bin)
    elapsed_time = time.time() - start_time
    xgb_model = grid.best_estimator_
    best_params = grid.best_params_

y_train_prob = xgb_model.predict_proba(X_train)[:, 1]
y_test_prob  = xgb_model.predict_proba(X_test)[:, 1]
y_train_pred = (y_train_prob >= 0.5).astype(int)
y_test_pred  = (y_test_prob >= 0.5).astype(int)

train_acc = accuracy_score(y_train_bin, y_train_pred)
test_acc  = accuracy_score(y_test_bin, y_test_pred)
train_auc = roc_auc_score(y_train_bin, y_train_prob)
test_auc  = roc_auc_score(y_test_bin, y_test_prob)

results_dict['model'].append('XGBoost')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

fpr, tpr, _ = roc_curve(y_test_bin, y_test_prob)
roc_data['XGBoost'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

feature_importance_dict['XGBoost'] = pd.DataFrame({
    'Feature': train_features, 'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

with open('results_oversampling/best_params_XGBoost.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Train AUROC:    {train_auc*100:.2f}%")
print(f"Test AUROC:     {test_auc*100:.2f}%")
print(f"Training time:  {elapsed_time:.2f} seconds")
print(f"Best params:    {best_params}")

# %% CATBOOST
# No external tuning required — internal optimization only (unchanged)

print("\n" + "="*70)
print("CATBOOST (INTERNAL OPTIMIZATION)")
print("="*70)

y_train_bin = np.where(y_train == 2, 1, 0)
y_test_bin  = np.where(y_test == 2, 1, 0)

print("External hyperparameter tuning: NOT REQUIRED")
print("CatBoost uses internal optimization and regularization")

cat_model = CatBoostClassifier(
    iterations=500,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=False,
    random_state=RANDOM_STATE
)

start_time = time.time()
cat_model.fit(X_train, y_train_bin)
elapsed_time = time.time() - start_time

best_params = "Not required (internal optimization)"

y_train_prob = cat_model.predict_proba(X_train)[:, 1]
y_test_prob  = cat_model.predict_proba(X_test)[:, 1]
y_train_pred = (y_train_prob >= 0.5).astype(int)
y_test_pred  = (y_test_prob >= 0.5).astype(int)

train_acc = accuracy_score(y_train_bin, y_train_pred)
test_acc  = accuracy_score(y_test_bin, y_test_pred)
train_auc = roc_auc_score(y_train_bin, y_train_prob)
test_auc  = roc_auc_score(y_test_bin, y_test_prob)

results_dict['model'].append('CatBoost')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

fpr, tpr, _ = roc_curve(y_test_bin, y_test_prob)
roc_data['CatBoost'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

feature_importance_dict['CatBoost'] = pd.DataFrame({
    'Feature': train_features, 'Importance': cat_model.get_feature_importance()
}).sort_values('Importance', ascending=False)

with open('results_oversampling/best_params_CatBoost.json', 'w') as f:
    json.dump({'note': best_params}, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Train AUROC:    {train_auc*100:.2f}%")
print(f"Test AUROC:     {test_auc*100:.2f}%")
print(f"Training time:  {elapsed_time:.2f} seconds")
print(f"Best params:    {best_params}")

# %% LIGHTGBM
# GridSearch best: feature_fraction=0.7, max_depth=14, num_leaves=90
# FIX: original code had max_depth=15, feature_fraction=0.6, num_leaves=80

print("\n" + "="*70)
print("LIGHTGBM")
print("="*70)

y_train_bin = np.where(y_train == 2, 1, 0)
y_test_bin  = np.where(y_test == 2, 1, 0)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters from GridSearchCV results...")
    print("Best params: max_depth=14, feature_fraction=0.7, num_leaves=90")

    lgbm_model = LGBMClassifier(
        objective='binary',
        max_depth=14,             # ← FIXED: was 15
        feature_fraction=0.7,     # ← FIXED: was 0.6
        num_leaves=90,            # ← FIXED: was 80
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )

    start_time = time.time()
    lgbm_model.fit(X_train, y_train_bin)
    elapsed_time = time.time() - start_time

    best_params = {
        'feature_fraction': 0.7,
        'max_depth': 14,
        'num_leaves': 90
    }

else:
    print("Running FOCUSED GridSearchCV...")
    print("Undersampling best: depth=15, feat_frac=0.6, leaves=80")
    param_grid = {
        'max_depth': [13, 14, 15, 16, 17],
        'feature_fraction': [0.5, 0.6, 0.7],
        'num_leaves': [70, 80, 90]
    }
    print(f"Grid combinations: {5 * 3 * 3} = 45")
    grid = GridSearchCV(
        LGBMClassifier(
            objective='binary', random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1
        ),
        param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1, verbose=1
    )
    start_time = time.time()
    grid.fit(X_train, y_train_bin)
    elapsed_time = time.time() - start_time
    lgbm_model = grid.best_estimator_
    best_params = grid.best_params_

y_train_prob = lgbm_model.predict_proba(X_train)[:, 1]
y_test_prob  = lgbm_model.predict_proba(X_test)[:, 1]
y_train_pred = (y_train_prob >= 0.5).astype(int)
y_test_pred  = (y_test_prob >= 0.5).astype(int)

train_acc = accuracy_score(y_train_bin, y_train_pred)
test_acc  = accuracy_score(y_test_bin, y_test_pred)
train_auc = roc_auc_score(y_train_bin, y_train_prob)
test_auc  = roc_auc_score(y_test_bin, y_test_prob)

results_dict['model'].append('LightGBM')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

fpr, tpr, _ = roc_curve(y_test_bin, y_test_prob)
roc_data['LightGBM'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

feature_importance_dict['LightGBM'] = pd.DataFrame({
    'Feature': train_features, 'Importance': lgbm_model.feature_importances_
}).sort_values('Importance', ascending=False)

with open('results_oversampling/best_params_LightGBM.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {test_acc:.4f}")
print(f"Train AUROC:    {train_auc*100:.2f}%")
print(f"Test AUROC:     {test_auc*100:.2f}%")
print(f"Training time:  {elapsed_time:.2f} seconds")
print(f"Best params:    {best_params}")

# %% SAVE AUROC SUMMARY

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

results_df = pd.DataFrame(results_dict)
results_df['best_params'] = results_df['best_params'].apply(lambda x: str(x))
results_df.to_csv('results_oversampling/auroc_summary.csv', index=False)
print("AUROC summary saved to: results_oversampling/auroc_summary.csv")

print("\n" + "="*70)
print("SUMMARY OF ALL MODELS")
print("="*70)
print(results_df.to_string(index=False))

# %% COMBINED AUROC PLOT

print("\n" + "="*70)
print("CREATING COMBINED AUROC PLOT")
print("="*70)

plt.figure(figsize=(10, 8))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f']

for idx, (model_name, data) in enumerate(roc_data.items()):
    plt.plot(data['fpr'], data['tpr'],
             label=f"{model_name} (AUC = {data['auc']:.2f})",
             linewidth=2, color=colors[idx])

plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('AUROC Comparison of Oversampling-Based Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures_oversampling/combined_auroc.png', dpi=300, bbox_inches='tight')
print("Combined AUROC plot saved to: figures_oversampling/combined_auroc.png")
plt.close()

# %% FEATURE IMPORTANCE PLOTS

print("\n" + "="*70)
print("CREATING FEATURE IMPORTANCE PLOTS")
print("="*70)

for model_name, importance_df in feature_importance_dict.items():

    csv_filename = f"results_oversampling/feature_importance_{model_name.replace(' ', '')}.csv"
    importance_df.to_csv(csv_filename, index=False)
    print(f"Feature importance saved: {csv_filename}")

    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)

    plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    fig_filename = f"figures_oversampling/feature_importance_{model_name.replace(' ', '')}.png"
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved: {fig_filename}")
    plt.close()

# %% FINAL SUMMARY

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print("\nAll results saved to:")
print("  - results_oversampling/auroc_summary.csv")
print("  - results_oversampling/best_params_<model>.json")
print("  - results_oversampling/feature_importance_<model>.csv")
print("  - figures_oversampling/combined_auroc.png")
print("  - figures_oversampling/feature_importance_<model>.png")
print("\n" + "="*70)