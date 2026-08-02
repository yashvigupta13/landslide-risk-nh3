

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
| **Data.py** | Rainfall pipeline (30-yr maximum): NetCDF → GeoTIFFs → resample/clip → Jenks classification → Tomek/undersampling and SMOTE/oversampling train-test splits. |
| **avg_rainfall.py** | Same pipeline using 30-yr average rainfall as an alternative factor. |
| **Undersampling-algos.py** | Trains & tunes all 8 classifiers on the undersampled data; generates susceptibility probability maps and G-scores. |
| **Oversampling-algos.py** | Trains & tunes all 8 classifiers on the oversampled data; generates susceptibility probability maps and G-scores. |
| **plots.py** | Generates the figures and validation plots used for analysing model performance and susceptibility class distributions. |
| **vulnerability.py** | Computes economic vulnerability (Degree of Loss × monetary value) per land-cover element. |
| **riskmap.py** | Integrates hazard and vulnerability layers to generate the final landslide risk map. |

---

Notes:
Scripts are exploratory (# %% cell-based), with hard-coded paths — shared for methodological transparency, not as a plug-and-run package.
Raw inputs (IMD rainfall NetCDF, causative-factor rasters, landslide inventory) are not included due to size/data-sharing restrictions.

Raw inputs (IMD rainfall NetCDF, causative-factor rasters, landslide inventory) are not included due to size/data-sharing restrictions.

These raw datasets were preprocessed (for example: IMD NetCDF → daily GeoTIFFs → yearly/30yr rasters, reprojection/resampling, clipping) before being used by the scripts.

Note on coordinates: some input CSVs use column names `Latitude` / `Longitude` while actually
containing projected easting/northing values (they were effectively flipped). The scripts
treat `Latitude` as Easting and `Longitude` as Northing when sampling/clipping to handle this.

---

## Setup

Python 3.10 or later.
QGIS 3.40 or later.

Main packages:

numpy, pandas, geopandas, rasterio, xarray, shapely, jenkspy, scikit-learn, imbalanced-learn, xgboost, catboost, lightgbm, matplotlib, plus GDAL (gdalwarp).

It's recommended to use a virtual environment before running any of the scripts:

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate.ps1
pip install -r requirements.txt
```
Alternatively:

```bash
conda env create -f environment.yml
conda install -r requirements.txt
```

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

# Typical Workflow

The scripts are intended to be executed in approximately the following order:

1. `Data.py`
2. `Undersampling-algos.py` or `Oversampling-algos.py`
3. `avg_rainfall.py`
4. `plots.py`
5. `vulnerability.py`
6. `riskmap.py`

---

# Data

The workflow requires several geospatial datasets, including:

(Data / Availability Statement)

---

# Outputs

The scripts generate:

- Trained machine learning models
- Performance statistics
- Validation plots
- Feature importance
- SHAP analysis
- Landslide susceptibility geotiff
- Hazard geotiff
- Vulnerability geotiff
- Risk geotiff

---

# Citation

If you use this repository, please cite:

*Citation will be updated after publication.*

---

# License

This repository is released under the MIT License.

---

# Contact

For questions regarding this repository or the manuscript, please contact the corresponding author.
