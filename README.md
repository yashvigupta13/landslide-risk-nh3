

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

# Workflow
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

All code and data files can be found in the 'src' folder.

| Script | Description |
|---------|-------------|
| **quick_test.py** | Minimal standalone test: undersampling + Logistic Regression only. For quick verification. |
| **Data.py** | Rainfall data pipeline (30-yr maximum): Tomek link undersampling and SMOTE oversampling train-test splits. |
| **avg_rainfall.py** | Same NetCDF→raster pipeline as Data.py but using 30-yr *average* daily rainfall instead of max. |
| **Undersampling-algos.py** | Trains & tunes all 8 classifiers on the undersampled data. |
| **Oversampling-algos.py** | Trains & tunes all 8 classifiers on the oversampled data. |
| **analysis.py** | Computes SHAP feature importance and G-scores for the top-3 models. |
| **plots.py** | Generates the validation plots used for analysing model performance and susceptibility class distributions. |
| **vulnerability.py** | Computes economic vulnerability (Degree of Loss × monetary value) per land-cover element. |
| **riskmap.py** | Integrates hazard and vulnerability layers to generate the final landslide risk map. |

---

Notes:
Scripts are exploratory (# %% cell-based), with hard-coded paths — shared for methodological transparency, not as a plug-and-run package.

Raw inputs (IMD rainfall NetCDF, causative-factor rasters, landslide inventory) are not included due to size/data-sharing restrictions.

These raw datasets were preprocessed (for example: IMD NetCDF → daily GeoTIFFs → yearly/30yr rasters, reprojection/resampling, clipping) and before being used by the scripts.

`analysis.py` writes its outputs (SHAP tables/plots, probability maps, Jenks-classified maps, G-score summaries) to a `feature/` directory, created automatically if it doesn't exist. This is also where `plots.py` looks for its input files.

---
# Quick Test

For a fast sanity check of the pipeline without running full model training:

`quick_test.py` : runs end-to-end from `processed1_truncated_data_with_rainfall_jenks.csv` alone (Tomek Links undersampling → Logistic Regression), using the verified best parameters. No pre-generated train/test files needed.

---

# Setup

Python 3.10 or later.
QGIS 3.40 or later.

Main packages:

numpy, pandas, geopandas, rasterio, xarray, shapely, jenkspy, scikit-learn, imbalanced-learn, xgboost, catboost, lightgbm, shap, matplotlib, plus GDAL (gdalwarp).

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


# Data

The complete datasets used in this study are **not included** in this repository because they are either:

- large geospatial datasets (GeoTIFF, NetCDF) that exceed Github limit,
- obtained from third-party sources,
- or subject to institutional restrictions.

This repository instead provides required truncated subset (~200,000 rows) of the full study dataset for demonstrating the workflow.

**Data File: ‘processed1_truncated_data_with_rainfall_jenks.csv’**

This is the input data file for this repository, and the point at which the code pipeline begins. It can be found in the 'src' folder. 
It is a fully processed dataset. Every causative factor has already been extracted, discretized, and merged into a single table, one row per sample point (raster pixel location).

The full dataset used to generate the results reported in the manuscript contains 1,218,552 total samples, of which 5,928 are landslide pixels and 1,212,624 are non-landslide pixels (~205:1 class imbalance). Because this is a subset, the class distribution and any metrics reproduced from this file will not exactly match the published results. 

‘Data.py’ and ‘avg_rainfall.py’ both take this CSV directly as input and carry out class-imbalance handling. ‘Undersampling-algos.py’ and ‘Oversampling-algos.py’ further carry out model training/tuning, and evaluation. 'analysis.py' generates SHAP feature importance plots, susceptibility files, and G-scores. ‘plots.py,’ ‘vulnerability.py,’ and ‘riskmap.py’ consume outputs generated further downstream.

Column reference

| Column | Description | Encoding |
|---------|-------------|-------------|
| **Unnamed: 0** | Pandas row index carried over from a prior export step | Integer, no analytical meaning — safe to drop (scripts do this automatically) |
| **Latitude** | Despite the name, this is the UTM Easting (X) coordinate, not geographic latitude | Meters, EPSG:32644 (UTM Zone 44N) |
| **Longitude** | Despite the name, this is the UTM Northing (Y) coordinate, not geographic longitude | Meters, EPSG:32644 (UTM Zone 44N) |
| **Lithology** | Rock type at the sample location (from Geological Survey of India map) | 1 = Phyllite, 2 = Sandstone with shale, 3 = Sandstone, 4 = Unconsolidated sediments, 5 = Shale with limestone, 6 = Granite, 7 = Quartzite, 8 = Quartzite alternating with shale, 9 = Dolomite, 10 = Limestone |
| **Land_Cover** | Land cover class at the sample location | 1 = Cropland/Agricultural land, 2 = Waterbody, 3 = Settlement, 4 = Barren land, 5 = High/Dense vegetation (forest), 6 = Moderate vegetation (grassland), 7 = Low vegetation (grassland) |
| **Plan_Curvature** | Plan curvature of the terrain (DEM derivative) | Ordinal class, 6-class natural breaks: 1 = most concave, 6 = most convex |
| **Profile_Curvature** | Profile curvature of the terrain (DEM derivative) | Ordinal class, 6-class natural breaks: 1 = most concave, 6 = most convex |
| **Aspect** | Slope aspect (compass direction the terrain faces) | 1 = Flat, 2 = North, 3 = Northeast, 4 = East, 5 = Southeast, 6 = South, 7 = Southwest, 8 = West, 9 = Northwest |
| **Fault** | Euclidean distance-buffer band from the nearest geological fault line | Ordinal class, 6 bands at 50 m intervals: 1 (≤50 m) → 6 (>250 m, up to 300 m band) |
| **River** | Euclidean distance-buffer band from the nearest river | Ordinal class, 6 bands at 40 m intervals: 1 (≤40 m) → 6 (>200 m, up to 240 m band) |
| **Road** | Euclidean distance-buffer band from the nearest road | Ordinal class, 6 bands at 40 m intervals: 1 (≤40 m) → 6 (>200 m, up to 240 m band) |
| **Dem** | Elevation (from Digital Elevation Model) | Ordinal class, 6-class natural breaks: 1 (≤887 m) → 6 (>3457 m) |
| **Slope** | Terrain slope angle (degrees) | Ordinal class, 6-class natural breaks: 1 (≤10.9°) → 6 (>47.3°) |
| **TWI** | Topographic Wetness Index | Ordinal class, 6-class natural breaks: 1 (≤3.38) → 6 (>13.65) |
| **STI** | Sediment Transport Index | Ordinal class, 6-class natural breaks: 1 (≤0) → 6 (>4.51) |
| **SPI** | Stream Power Index | Ordinal class, 6-class natural breaks: 1 (≤0) → 6 (>31.37) |
| **Slope_Length** | Slope length (meters) | Ordinal class, 6-class natural breaks: 1 (≤0) → 6 (>345.67 m) |
| **Landslide** | Ground-truth label — landslide occurrence at this point | 1 = No landslide, 2 = Landslide |
| **Rainfall_Jenks** | 30-year rainfall factor (maximum or average daily rainfall, depending on which pipeline generated the file. See Data.py vs avg_rainfall.py), classified via jenkspy | Ordinal class 1–6, Jenks natural breaks (thresholds computed dynamically from the rainfall raster, not fixed) |

Notes:  
•	For all "natural breaks" columns above, class 1 = lowest value range, class 6 = highest value range, of the underlying continuous variable at that location. Direction of association with landslide risk varies by factor (see manuscript SHAP analysis for feature-level interpretation).  
  
•	Input CSV uses column names `Latitude` / `Longitude` while actually containing projected Easting/Northing values purely for compatibility with earlier processing steps; treat them as projected UTM 44N coordinates, not WGS84 lat/lon. The scripts treat `Latitude` as Easting and `Longitude` as Northing when sampling/clipping to handle this.  
  
•	The `Rainfall_Jenks` column in `processed1_truncated_data_with_rainfall_jenks.csv` is generated by the **MAX**-rainfall pipeline (`Data.py`), which is the method used for the published results. Running `avg_rainfall.py` reprocesses the raw rainfall archive using **average** daily rainfall instead and writes a separate file, `processed1_truncated_data_with_rainfall_jenks_AVG.csv`, with its own `Rainfall_Jenks` column. To run the downstream pipeline (`Undersampling-algos.py` / `Oversampling-algos.py`) on the average-rainfall variant, rename or copy this file to `processed1_truncated_data_with_rainfall_jenks.csv` first.  

---

# Typical Workflow

The scripts are intended to be executed in approximately the following order:

1. `Data.py`
2. `avg_rainfall.py` (alternatively to Data.py)
3. `Undersampling-algos.py` or `Oversampling-algos.py`
4. `analysis.py`
5. `plots.py`
6. `vulnerability.py`
7. `riskmap.py`

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
