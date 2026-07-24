#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Undersampling algorithms - CORRECTED VERSION

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

# Set matplotlib parameters for better plots
rcParams['figure.figsize'] = (10, 8)
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Global configuration
USE_OPTIMIZED_PARAMS = True  # True = Fast mode, False = GridSearch
RANDOM_STATE = 42

print("="*70)
print("LANDSLIDE SUSCEPTIBILITY MAPPING - MODEL TRAINING")
print("="*70)

# %% LOAD DATA

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

# Load training and test sets
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("Y_train.csv")["Landslide"].values
y_test = pd.read_csv("Y_test.csv")["Landslide"].values

# Load full dataset for feature names
full_df = pd.read_csv("processed1_truncated_data_with_rainfall_jenks.csv")
train_features = X_train.columns.tolist()

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of features: {len(train_features)}")
print(f"Training features: {train_features}")

# Check class distribution
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

print(f"\nTraining set class distribution:")
for cls, cnt in zip(unique_train, counts_train):
    print(f"  Class {cls}: {cnt} ({cnt/len(y_train)*100:.1f}%)")
print(f"\nTest set class distribution:")
for cls, cnt in zip(unique_test, counts_test):
    print(f"  Class {cls}: {cnt} ({cnt/len(y_test)*100:.1f}%)")

# %% INITIALIZE RESULTS STORAGE

# Dictionary to store all results
results_dict = {
    'model': [],
    'train_accuracy': [],
    'test_accuracy': [],
    'train_auroc': [],
    'test_auroc': [],
    'training_time': [],
    'best_params': []
}

# Dictionary to store ROC curve data for plotting
roc_data = {}

# Dictionary to store feature importance
feature_importance_dict = {}
'''
# %% LOGISTIC REGRESSION

print("\n" + "="*70)
print("LOGISTIC REGRESSION")
print("="*70)

# Scale features for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters...")
    
    lr_model = LogisticRegression(
        penalty='l2',
        solver='liblinear',
        C=10,  # CORRECTED: was 20, GridSearch found 10
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    
    start_time = time.time()
    lr_model.fit(X_train_scaled, y_train)
    elapsed_time = time.time() - start_time
    
    best_params = {'penalty': 'l2', 'solver': 'liblinear', 'C': 10}

else:
    print("Running GridSearchCV...")
    
    param_grid = {
        'penalty': ['l2'],
        'solver': ['liblinear'],
        'C': [0.1, 1, 5, 10, 12, 20]
    }
    
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid.fit(X_train_scaled, y_train)
    elapsed_time = time.time() - start_time
    
    lr_model = grid.best_estimator_
    best_params = grid.best_params_

# Predictions
y_train_pred = lr_model.predict(X_train_scaled)
y_test_pred = lr_model.predict(X_test_scaled)
y_train_prob = lr_model.predict_proba(X_train_scaled)[:, 1]
y_test_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train == 2, y_train_prob)
test_auc = roc_auc_score(y_test == 2, y_test_prob)

# Store results
results_dict['model'].append('Logistic Regression')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

# Store ROC data
fpr, tpr, _ = roc_curve(y_test == 2, y_test_prob)
roc_data['Logistic Regression'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

# Feature importance (absolute standardized coefficients)
coefficients = np.abs(lr_model.coef_[0])
feature_importance_dict['Logistic Regression'] = pd.DataFrame({
    'Feature': train_features,
    'Importance': coefficients
}).sort_values('Importance', ascending=False)

# Save best params
with open('results/best_params_LogisticRegression.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Train AUROC: {train_auc*100:.2f}")
print(f"Test AUROC: {test_auc*100:.2f}")
print(f"Training time: {elapsed_time:.2f} seconds")
print(f"Best params: {best_params}")

# %% DECISION TREE

print("\n" + "="*70)
print("DECISION TREE")
print("="*70)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters...")
    
    dt_model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=6,
        min_samples_leaf=5,  # CORRECTED: was 7, GridSearch found 5
        min_samples_split=2,
        random_state=RANDOM_STATE
    )
    
    start_time = time.time()
    dt_model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    best_params = {
        'criterion': 'entropy',
        'max_depth': 6,
        'min_samples_leaf': 5,
        'min_samples_split': 2
    }

else:
    print("Running GridSearchCV...")
    
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 4, 5, 6],
        'min_samples_leaf': [3, 5, 7],
        'min_samples_split': [2, 3, 4]
    }
    
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    dt_model = grid.best_estimator_
    best_params = grid.best_params_

# Predictions
y_train_pred = dt_model.predict(X_train)
y_test_pred = dt_model.predict(X_test)
y_train_prob = dt_model.predict_proba(X_train)[:, 1]
y_test_prob = dt_model.predict_proba(X_test)[:, 1]

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train == 2, y_train_prob)
test_auc = roc_auc_score(y_test == 2, y_test_prob)

# Store results
results_dict['model'].append('Decision Tree')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

# Store ROC data
fpr, tpr, _ = roc_curve(y_test == 2, y_test_prob)
roc_data['Decision Tree'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

# Feature importance
feature_importance_dict['Decision Tree'] = pd.DataFrame({
    'Feature': train_features,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Save best params
with open('results/best_params_DecisionTree.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Train AUROC: {train_auc*100:.2f}")
print(f"Test AUROC: {test_auc*100:.2f}")
print(f"Training time: {elapsed_time:.2f} seconds")
print(f"Best params: {best_params}")

# %% RANDOM FOREST

print("\n" + "="*70)
print("RANDOM FOREST")
print("="*70)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters...")
    
    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_leaf=1,
        min_samples_split=3,  # CORRECTED: was 2, GridSearch found 3
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    best_params = {
        'n_estimators': 250,
        'max_depth': 6,
        'min_samples_leaf': 1,
        'min_samples_split': 3
    }

else:
    print("Running GridSearchCV...")
    
    param_grid = {
        'n_estimators': [100, 250, 500],
        'max_depth': [3, 4, 5, 6],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [2, 3, 4]
    }
    
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    rf_model = grid.best_estimator_
    best_params = grid.best_params_

# Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
y_train_prob = rf_model.predict_proba(X_train)[:, 1]
y_test_prob = rf_model.predict_proba(X_test)[:, 1]

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train == 2, y_train_prob)
test_auc = roc_auc_score(y_test == 2, y_test_prob)

# Store results
results_dict['model'].append('Random Forest')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

# Store ROC data
fpr, tpr, _ = roc_curve(y_test == 2, y_test_prob)
roc_data['Random Forest'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

# Feature importance
feature_importance_dict['Random Forest'] = pd.DataFrame({
    'Feature': train_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Save best params
with open('results/best_params_RandomForest.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Train AUROC: {train_auc*100:.2f}")
print(f"Test AUROC: {test_auc*100:.2f}")
print(f"Training time: {elapsed_time:.2f} seconds")
print(f"Best params: {best_params}")

# %% ADABOOST

print("\n" + "="*70)
print("ADABOOST")
print("="*70)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters...")
    
    base_dt = DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE)
    
    ada_model = AdaBoostClassifier(
        estimator=base_dt,
        n_estimators=500,
        learning_rate=1.5,
        random_state=RANDOM_STATE
    )
    
    start_time = time.time()
    ada_model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    best_params = {
        'n_estimators': 500,
        'learning_rate': 1.5
    }

else:
    print("Running GridSearchCV...")
    
    param_grid = {
        'n_estimators': [100, 250, 500],
        'learning_rate': [0.5, 1.0, 1.5, 2.0]
    }
    
    grid = GridSearchCV(
        AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            random_state=RANDOM_STATE
        ),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    ada_model = grid.best_estimator_
    best_params = grid.best_params_

# Predictions
y_train_pred = ada_model.predict(X_train)
y_test_pred = ada_model.predict(X_test)
y_train_prob = ada_model.predict_proba(X_train)[:, 1]
y_test_prob = ada_model.predict_proba(X_test)[:, 1]

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train == 2, y_train_prob)
test_auc = roc_auc_score(y_test == 2, y_test_prob)

# Store results
results_dict['model'].append('AdaBoost')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

# Store ROC data
fpr, tpr, _ = roc_curve(y_test == 2, y_test_prob)
roc_data['AdaBoost'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

# Feature importance
feature_importance_dict['AdaBoost'] = pd.DataFrame({
    'Feature': train_features,
    'Importance': ada_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Save best params
with open('results/best_params_AdaBoost.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Train AUROC: {train_auc*100:.2f}")
print(f"Test AUROC: {test_auc*100:.2f}")
print(f"Training time: {elapsed_time:.2f} seconds")
print(f"Best params: {best_params}")

# %% EXTRA TREES

print("\n" + "="*70)
print("EXTRA TREES")
print("="*70)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters...")
    
    et_model = ExtraTreesClassifier(
        n_estimators=500,
        criterion='gini',  # Matches GridSearch result
        max_depth=5,
        min_samples_leaf=1,
        min_samples_split=3,  # Matches GridSearch result
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    start_time = time.time()
    et_model.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    best_params = {
        'n_estimators': 500,
        'criterion': 'gini',
        'max_depth': 5,
        'min_samples_leaf': 1,
        'min_samples_split': 3
    }

else:
    print("Running GridSearchCV...")
    
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [3, 4, 5],
        'criterion': ['gini', 'entropy'],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [2, 3]
    }
    
    grid = GridSearchCV(
        ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    
    et_model = grid.best_estimator_
    best_params = grid.best_params_

# Predictions
y_train_pred = et_model.predict(X_train)
y_test_pred = et_model.predict(X_test)
y_train_prob = et_model.predict_proba(X_train)[:, 1]
y_test_prob = et_model.predict_proba(X_test)[:, 1]

# Metrics
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_auc = roc_auc_score(y_train == 2, y_train_prob)
test_auc = roc_auc_score(y_test == 2, y_test_prob)

# Store results
results_dict['model'].append('Extra Trees')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

# Store ROC data
fpr, tpr, _ = roc_curve(y_test == 2, y_test_prob)
roc_data['Extra Trees'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

# Feature importance
feature_importance_dict['Extra Trees'] = pd.DataFrame({
    'Feature': train_features,
    'Importance': et_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Save best params
with open('results/best_params_ExtraTrees.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Train AUROC: {train_auc*100:.2f}")
print(f"Test AUROC: {test_auc*100:.2f}")
print(f"Training time: {elapsed_time:.2f} seconds")
print(f"Best params: {best_params}")

# %% XGBOOST

print("\n" + "="*70)
print("XGBOOST")
print("="*70)

# Convert labels to binary {0, 1}
y_train_bin = np.where(y_train == 2, 1, 0)
y_test_bin = np.where(y_test == 2, 1, 0)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters...")
    
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=100,
        learning_rate=0.1,       # CORRECTED: was 0.3, GridSearch found 0.1
        max_depth=6,
        gamma=0,                 # Already correct
        subsample=0.8,           # CORRECTED: was 1.0, GridSearch found 0.8
        colsample_bytree=1.0,    # CORRECTED: was 0.8, GridSearch found 1.0
        min_child_weight=1,      # CORRECTED: was 2, GridSearch found 1
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    start_time = time.time()
    xgb_model.fit(X_train, y_train_bin)
    elapsed_time = time.time() - start_time
    
    best_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 1.0,
        'min_child_weight': 1
    }

else:
    print("Running GridSearchCV...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.3],
        'max_depth': [4, 5, 6],
        'gamma': [0, 1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 2]
    }
    
    grid = GridSearchCV(
        XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid.fit(X_train, y_train_bin)
    elapsed_time = time.time() - start_time
    
    xgb_model = grid.best_estimator_
    best_params = grid.best_params_

# Predictions
y_train_prob = xgb_model.predict_proba(X_train)[:, 1]
y_test_prob = xgb_model.predict_proba(X_test)[:, 1]
y_train_pred = (y_train_prob >= 0.5).astype(int)
y_test_pred = (y_test_prob >= 0.5).astype(int)

# Metrics
train_acc = accuracy_score(y_train_bin, y_train_pred)
test_acc = accuracy_score(y_test_bin, y_test_pred)
train_auc = roc_auc_score(y_train_bin, y_train_prob)
test_auc = roc_auc_score(y_test_bin, y_test_prob)

# Store results
results_dict['model'].append('XGBoost')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

# Store ROC data
fpr, tpr, _ = roc_curve(y_test_bin, y_test_prob)
roc_data['XGBoost'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

# Feature importance
feature_importance_dict['XGBoost'] = pd.DataFrame({
    'Feature': train_features,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Save best params
with open('results/best_params_XGBoost.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Train AUROC: {train_auc*100:.2f}")
print(f"Test AUROC: {test_auc*100:.2f}")
print(f"Training time: {elapsed_time:.2f} seconds")
print(f"Best params: {best_params}")

# %% CATBOOST 

print("\n" + "="*70)
print("CATBOOST (INTERNAL OPTIMIZATION)")
print("="*70)

# Convert labels to binary {0, 1}
y_train_bin = np.where(y_train == 2, 1, 0)
y_test_bin  = np.where(y_test == 2, 1, 0)

print("External hyperparameter tuning: NOT REQUIRED")
print("CatBoost uses internal optimization and regularization")

# ------------------------------
# MODEL INITIALIZATION
# ------------------------------
cat_model = CatBoostClassifier(
    iterations=500,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=False,
    random_state=RANDOM_STATE
)

# ------------------------------
# TRAINING
# ------------------------------
start_time = time.time()
cat_model.fit(X_train, y_train_bin)
elapsed_time = time.time() - start_time

best_params = "Not required (internal optimization)"

# ------------------------------
# PREDICTIONS
# ------------------------------
y_train_prob = cat_model.predict_proba(X_train)[:, 1]
y_test_prob  = cat_model.predict_proba(X_test)[:, 1]

y_train_pred = (y_train_prob >= 0.5).astype(int)
y_test_pred  = (y_test_prob >= 0.5).astype(int)

# ------------------------------
# METRICS
# ------------------------------
train_acc = accuracy_score(y_train_bin, y_train_pred)
test_acc  = accuracy_score(y_test_bin, y_test_pred)

train_auc = roc_auc_score(y_train_bin, y_train_prob)
test_auc  = roc_auc_score(y_test_bin, y_test_prob)

# ------------------------------
# STORE RESULTS
# ------------------------------
results_dict['model'].append('CatBoost')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

# ------------------------------
# ROC CURVE DATA
# ------------------------------
fpr, tpr, _ = roc_curve(y_test_bin, y_test_prob)
roc_data['CatBoost'] = {
    'fpr': fpr,
    'tpr': tpr,
    'auc': test_auc
}

# ------------------------------
# FEATURE IMPORTANCE
# ------------------------------
feature_importance_dict['CatBoost'] = pd.DataFrame({
    'Feature': train_features,
    'Importance': cat_model.get_feature_importance()
}).sort_values('Importance', ascending=False)

# ------------------------------
# SAVE PARAMS
# ------------------------------
with open('results/best_params_CatBoost.json', 'w') as f:
    json.dump({'note': best_params}, f, indent=4)

# ------------------------------
# PRINT SUMMARY
# ------------------------------
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Train AUROC   : {train_auc*100:.2f}")
print(f"Test AUROC    : {test_auc*100:.2f}")
print(f"Training time: {elapsed_time:.2f} seconds")
print("Best params  :", best_params)
'''
# %% LIGHTGBM

print("\n" + "="*70)
print("LIGHTGBM")
print("="*70)

# Convert labels to binary {0, 1}
y_train_bin = np.where(y_train == 2, 1, 0)
y_test_bin = np.where(y_test == 2, 1, 0)

if USE_OPTIMIZED_PARAMS:
    print("Using optimized parameters...")
    
    lgbm_model = LGBMClassifier(
        objective='binary',
        max_depth=15,
        feature_fraction=0.6,  # Matches GridSearch result
        num_leaves=80,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )
    
    start_time = time.time()
    lgbm_model.fit(X_train, y_train_bin)
    elapsed_time = time.time() - start_time
    
    best_params = {
        'max_depth': 15,
        'feature_fraction': 0.6,
        'num_leaves': 80
    }

else:
    print("Running GridSearchCV...")
    
    param_grid = {
        'max_depth': [8, 10, 12],
        'feature_fraction': [0.7, 0.8, 0.9],
        'num_leaves': [31, 50, 70]
    }
    
    grid = GridSearchCV(
        LGBMClassifier(
            objective='binary',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1
        ),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid.fit(X_train, y_train_bin)
    elapsed_time = time.time() - start_time
    
    lgbm_model = grid.best_estimator_
    best_params = grid.best_params_

# Predictions
y_train_prob = lgbm_model.predict_proba(X_train)[:, 1]
y_test_prob = lgbm_model.predict_proba(X_test)[:, 1]
y_train_pred = (y_train_prob >= 0.5).astype(int)
y_test_pred = (y_test_prob >= 0.5).astype(int)

# Metrics
train_acc = accuracy_score(y_train_bin, y_train_pred)
test_acc = accuracy_score(y_test_bin, y_test_pred)
train_auc = roc_auc_score(y_train_bin, y_train_prob)
test_auc = roc_auc_score(y_test_bin, y_test_prob)

# Store results
results_dict['model'].append('LightGBM')
results_dict['train_accuracy'].append(train_acc)
results_dict['test_accuracy'].append(test_acc)
results_dict['train_auroc'].append(train_auc)
results_dict['test_auroc'].append(test_auc)
results_dict['training_time'].append(elapsed_time)
results_dict['best_params'].append(best_params)

# Store ROC data
fpr, tpr, _ = roc_curve(y_test_bin, y_test_prob)
roc_data['LightGBM'] = {'fpr': fpr, 'tpr': tpr, 'auc': test_auc}

# Feature importance
feature_importance_dict['LightGBM'] = pd.DataFrame({
    'Feature': train_features,
    'Importance': lgbm_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Save best params
with open('results/best_params_LightGBM.json', 'w') as f:
    json.dump(best_params, f, indent=4)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Train AUROC: {train_auc*100:.2f}")
print(f"Test AUROC: {test_auc*100:.2f}")
print(f"Training time: {elapsed_time:.2f} seconds")
print(f"Best params: {best_params}")

# %% SAVE AUROC SUMMARY

print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Create results DataFrame
results_df = pd.DataFrame(results_dict)

# Convert best_params to string for CSV
results_df['best_params'] = results_df['best_params'].apply(lambda x: str(x))

# Save AUROC summary
results_df.to_csv('results/auroc_summary.csv', index=False)
print("AUROC summary saved to: results/auroc_summary.csv")

# Display summary
print("\n" + "="*70)
print("SUMMARY OF ALL MODELS")
print("="*70)
print(results_df.to_string(index=False))

# %% COMBINED AUROC PLOT

print("\n" + "="*70)
print("CREATING COMBINED AUROC PLOT")
print("="*70)

plt.figure(figsize=(10, 8))

# Define colors for each model
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f']

# Plot ROC curve for each model
for idx, (model_name, data) in enumerate(roc_data.items()):
    plt.plot(data['fpr'], data['tpr'], 
             label=f"{model_name} (AUC = {data['auc']:.2f})",
             linewidth=2, color=colors[idx])

# Plot diagonal reference line
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')

# Formatting
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('AUROC Comparison of Undersampling-Based Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save figure
plt.savefig('figures/combined_auroc.png', dpi=300, bbox_inches='tight')
print("Combined AUROC plot saved to: figures/combined_auroc.png")
plt.close()
'''
# %% FEATURE IMPORTANCE PLOTS

print("\n" + "="*70)
print("CREATING FEATURE IMPORTANCE PLOTS")
print("="*70)

for model_name, importance_df in feature_importance_dict.items():
    
    # Save CSV
    csv_filename = f"results/feature_importance_{model_name.replace(' ', '')}.csv"
    importance_df.to_csv(csv_filename, index=False)
    print(f"Feature importance saved: {csv_filename}")
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Get top 15 features
    top_features = importance_df.head(15)
    
    # Create horizontal bar plot
    plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    fig_filename = f"figures/feature_importance_{model_name.replace(' ', '')}.png"
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved: {fig_filename}")
    plt.close()

# %% FINAL SUMMARY

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print("\nAll results saved to:")
print("  - results/auroc_summary.csv")
print("  - results/best_params_<model>.json")
print("  - results/feature_importance_<model>.csv")
print("  - figures/combined_auroc.png")
print("  - figures/feature_importance_<model>.png")
print("\n" + "="*70)

# %% GENERATE PROBABILITY MAPS (UNDERSAMPLING)

RANDOM_STATE = 42

# Load full dataset
df = pd.read_csv("processed1_truncated_data_with_rainfall_jenks.csv")

coords = df[["Latitude", "Longitude"]].copy()

X_full = df[train_features].copy()

print("X_full shape:", X_full.shape)
print("Feature alignment OK:", list(X_full.columns) == train_features)

# =======================
# XGBOOST (CORRECTED PARAMETERS)
# =======================
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100,
    learning_rate=0.1,       # CORRECTED
    max_depth=6,
    gamma=0,
    subsample=0.8,           # CORRECTED
    colsample_bytree=1.0,    # CORRECTED
    min_child_weight=1,      # CORRECTED
    random_state=RANDOM_STATE,
    n_jobs=-1
)

y_train = pd.read_csv("Y_train.csv")["Landslide"].values
X_train = pd.read_csv("X_train.csv")
y_train_bin = np.where(y_train == 2, 1, 0)

xgb.fit(X_train, y_train_bin)
xgb_prob = xgb.predict_proba(X_full)[:, 1]

xgb_df = coords.copy()
xgb_df["LSM_Probability"] = xgb_prob
xgb_df.to_csv("LSM_XGBoost_Probabilities.csv", index=False)

# =======================
# CATBOOST
# =======================
cat = CatBoostClassifier(
    iterations=500,
    loss_function='Logloss',
    eval_metric='AUC',
    random_state=RANDOM_STATE,
    verbose=False
)

cat.fit(X_train, y_train_bin)
cat_prob = cat.predict_proba(X_full)[:, 1]

cat_df = coords.copy()
cat_df["LSM_Probability"] = cat_prob
cat_df.to_csv("LSM_CatBoost_Probabilities.csv", index=False)

# =======================
# LIGHTGBM (CONSISTENT PARAMETERS)
# =======================
lgbm = LGBMClassifier(
    objective='binary',
    max_depth=10,
    feature_fraction=0.8,  # Consistent with GridSearch
    num_leaves=70,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

lgbm.fit(X_train, y_train_bin)
lgbm_prob = lgbm.predict_proba(X_full)[:, 1]

lgbm_df = coords.copy()
lgbm_df["LSM_Probability"] = lgbm_prob
lgbm_df.to_csv("LSM_LightGBM_Probabilities.csv", index=False)

print("✓ Probability maps saved for XGBoost, CatBoost, LightGBM")

# %% Applying Jenks (5 classes)

import pandas as pd
import jenkspy

def apply_jenks(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    values = df["LSM_Probability"].values

    breaks = jenkspy.jenks_breaks(values, n_classes=5)

    df["LSM_Class"] = pd.cut(
        values,
        bins=breaks,
        labels=[1, 2, 3, 4, 5],
        include_lowest=True
    ).astype(int)

    df.to_csv(output_csv, index=False)
    print(f"✓ Jenks applied → {output_csv}")

apply_jenks("LSM_XGBoost_Probabilities.csv",
            "LSM_XGBoost_Probabilities_Jenks.csv")

apply_jenks("LSM_CatBoost_Probabilities.csv",
            "LSM_CatBoost_Probabilities_Jenks.csv")

apply_jenks("LSM_LightGBM_Probabilities.csv",
            "LSM_LightGBM_Probabilities_Jenks.csv")

# %% G-SCORE CALCULATION
import pandas as pd
import numpy as np

def compute_g_score(jenks_csv, model_name):
    df = pd.read_csv(jenks_csv)
    # Reference for actual landslide locations
    inv_df = pd.read_csv("processed1_truncated_data_with_rainfall_jenks.csv")[["Latitude", "Longitude", "Landslide"]]

    # Fix floating-point precision for merging
    for d in (df, inv_df):
        d["Latitude"] = d["Latitude"].round(6)
        d["Longitude"] = d["Longitude"].round(6)

    merged = df.merge(inv_df, on=["Latitude", "Longitude"], how="inner")

    # Define high (4) and very high (5) susceptibility zones
    danger_mask = merged["LSM_Class"].isin([4, 5])

    # A: Total landslide points (assuming 2 represents a landslide event)
    A = (merged["Landslide"] == 2).sum()   
    # B: Total points in the study area
    B = len(merged)                        

    # a: Landslide points in high/very high zones
    a = ((merged["Landslide"] == 2) & danger_mask).sum()
    # b: Total points in high/very high zones
    b = danger_mask.sum()

    # G-score calculation using the cubic root of the non-landslide density ratio
    # Formula: G = (a/A) * (1 - (b-a)/(B-A))^(1/3)
    success_rate = a / A
    false_positive_density = (b - a) / (B - A)
    G = success_rate * np.power(1 - false_positive_density, 1/3)

    print(f"\n{model_name} G-score: {G:.4f}")
    print(f"Success Rate (a/A): {success_rate:.4f}")
    print(f"Area Density (b/B): {b/B:.4f}")

# Run for the models
models = [
    ("LSM_XGBoost_Probabilities_Jenks.csv", "XGBoost"),
    ("LSM_CatBoost_Probabilities_Jenks.csv", "CatBoost"),
    ("LSM_LightGBM_Probabilities_Jenks.csv", "LightGBM")
]

for csv, name in models:
    compute_g_score(csv, name)
'''