

## Overview

This repository contains the source code developed for the manuscript:

> **Modelling of static and dynamic causative factors, strategies to handle imbalance data, and model explainability for landslide susceptibility leading to economic risk assessment**

The repository implements an end-to-end geospatial machine learning workflow for landslide susceptibility, hazard, vulnerability, and risk assessment. The workflow integrates GIS-based preprocessing, rainfall analysis, class imbalance handling, multiple machine learning algorithms, explainable AI (SHAP), and geospatial risk mapping.

---

## Authors

- Vivek Saxena
- Yashvi Gupta
- Athang Yawalkar
- Dr. Snehmani

---

## Study Area

National Highway-34 (NH-34), buffer 3km on both sides, Uttarakhand, India.

## Workflow
The workflow includes:

- Rainfall preprocessing from NetCDF data
- Handling class imbalance using undersampling and oversampling
- Training multiple machine learning models
- Generation of landslide susceptibility maps
- Hazard assessment
- Vulnerability assessment
- Landslide risk mapping
- Validation and visualization of results

---

# Repository Contents

| Script | Description |
|---------|-------------|
| **avg_rainfall.py** | Processes rainfall data from IMD Pune and creates rainfall-related datasets used for modelling. |
| **Data.py** | Prepares the modelling dataset, including preprocessing, feature generation, train-test split, and data preparation for machine learning. |
| **Undersampling-algos.py** | Implements the undersampling workflow using Tomek Links and trains the machine learning classifiers. |
| **Oversampling-algos.py** | Implements the oversampling workflow using SMOTE and trains the machine learning classifiers. |
| **vulnerability.py** | Computes the vulnerability layer from land-cover and road information and produces the vulnerability raster. |
| **riskmap.py** | Integrates susceptibility, hazard, and vulnerability layers to generate the final landslide risk map. |
| **plots.py** | Generates the figures and validation plots used for analysing model performance and susceptibility class distributions. |

---

# Machine Learning Models

The repository includes implementations of:

- Logistic Regression
- Decision Tree
- Random Forest
- AdaBoost
- Extra Trees
- XGBoost
- CatBoost
- LightGBM

Both undersampling and oversampling strategies are provided.

---

# Software Requirements

Python 3.10 or later.

Main packages:

- numpy
- pandas
- matplotlib
- rasterio
- geopandas
- xarray
- scikit-learn
- imbalanced-learn
- xgboost
- lightgbm
- catboost
- shap
- jenkspy

Install dependencies using

```bash
pip install -r requirements.txt
```

---

# Typical Workflow

The scripts are intended to be executed in approximately the following order:

1. `avg_rainfall.py`
2. `Data.py`
3. `Undersampling-algos.py` or `Oversampling-algos.py`
4. `vulnerability.py`
5. `riskmap.py`
6. `plots.py`

---

# Data

The workflow requires several geospatial datasets, including:

(Data / Availability Statement)

---

# Outputs

The scripts generate:

- Trained machine learning models
- Performance statistics
- ROC curves
- Feature importance
- SHAP analysis
- Landslide susceptibility geotiff
- Hazard geotiff
- Vulnerability geotiff
- Risk geotiff
- Validation plots

---

# Citation

If you use this repository, please cite:

*Citation will be updated after publication.*

---

# License

(Not yet added)

---

# Contact

For questions regarding this repository or the manuscript, please contact the corresponding author.
