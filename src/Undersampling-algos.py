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
