
# %% Rainfall imports and folder structure

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

BASE_DIR = os.getcwd()

DIR_RAW_NC   = "0_raw_netcdf"
DIR_DAILY    = "1_daily_geotiff"
DIR_YEARLY   = "2_yearly_max"
DIR_30YR     = "3_30yr_max"
DIR_CONT     = "4_continuous_30m"

for d in [DIR_DAILY, DIR_YEARLY, DIR_30YR, DIR_CONT]:
    os.makedirs(d, exist_ok=True)
# %% NetCDF -> DAILY GEOTIFFS

print("\n===== NETCDF -> DAILY GEOTIFFS =====\n")

nc_files = sorted(f for f in os.listdir(DIR_RAW_NC) if f.endswith(".nc"))
if not nc_files:
    raise RuntimeError("No NetCDF files found")

for nc_file in nc_files:
    ds = xr.open_dataset(os.path.join(DIR_RAW_NC, nc_file), decode_times=False)

    for v in ["RAINFALL", "rf", "rainfall"]:
        if v in ds.data_vars:
            rain_var = v
            break
    else:
        raise RuntimeError(f"No rainfall variable in {nc_file}")

    rain = ds[rain_var]

    lat = "LATITUDE" if "LATITUDE" in ds.coords else "lat"
    lon = "LONGITUDE" if "LONGITUDE" in ds.coords else "lon"
    time = "TIME" if "TIME" in ds.coords else "time"

    lats = ds[lat].values
    lons = ds[lon].values
    times = rain[time].values

    xres = abs(lons[1] - lons[0])
    yres = abs(lats[1] - lats[0])

    transform = from_origin(lons.min(), lats.max(), xres, yres)

    for i, t in enumerate(times):
        date = str(t)[:10]
        out = f"{nc_file.replace('.nc','')}_{date}.tif"
        out_path = os.path.join(DIR_DAILY, out)

        data = rain.isel({time: i}).values.astype("float32")
        
        if lats[0] < lats[-1]:
            data = np.flipud(data)
    
        if np.all(np.isnan(data)):
            continue

        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=-999,
            compress="lzw"
        ) as dst:
            dst.write(np.nan_to_num(data, nan=-999), 1)

    ds.close()

print("STEP 1 COMPLETED")

# %% YEARLY MAX

print("\n===== YEARLY MAX =====\n")

files_by_year = defaultdict(list)

for f in os.listdir(DIR_DAILY):
    m = re.search(r"ind(\d{4})", f)
    if m:
        files_by_year[m.group(1)].append(os.path.join(DIR_DAILY, f))

for year, files in sorted(files_by_year.items()):
    max_data, meta, nodata = None, None, -999

    for f in files:
        with rasterio.open(f) as src:
            data = src.read(1).astype("float32")
            nodata = src.nodata
            data[data == nodata] = np.nan

            if max_data is None:
                max_data = data
                meta = src.meta.copy()
            else:
                max_data = np.fmax(max_data, data)

    max_data = np.where(np.isnan(max_data), nodata, max_data)
    meta.update(compress="lzw")

    out = os.path.join(DIR_YEARLY, f"rainfall_MAX_{year}.tif")
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(max_data, 1)

print("STEP 2 COMPLETED")

# %% 30-YEAR MAX

print("\n===== 30-YEAR MAX =====\n")

files = sorted(os.path.join(DIR_YEARLY, f) for f in os.listdir(DIR_YEARLY))
with rasterio.open(files[0]) as ref:
    meta = ref.meta.copy()
    nodata = ref.nodata

stack = []
for f in files:
    with rasterio.open(f) as ds:
        d = ds.read(1).astype("float32")
        d[d == nodata] = np.nan
        stack.append(d)

stack = np.stack(stack)
max30 = np.nanmax(stack, axis=0)
max30 = np.where(np.isnan(max30), nodata, max30)

out = os.path.join(DIR_30YR, "rainfall_30yr_MAX.tif")
with rasterio.open(out, "w", **meta) as dst:
    dst.write(max30, 1)

print("STEP 3 COMPLETED")

# %% Reproject + Resample Rainfall Raster to 30m

import subprocess

subprocess.run([
    'gdalwarp',
    '-r', 'bilinear',
    '-tr', '30', '30',
    '-t_srs', 'EPSG:32644',
    '-dstnodata', '-999',
    '-multi',
    '-wm', '2048',
    '3_30yr_max/rainfall_30yr_MAX.tif',
    '4_continuous_30m/rainfall_30yr_MAX_30m.tif'
], check=True)
# %% Clip Rainfall Raster to CSV AOI (Buffered Box Clip)

print("\n===== CLIP TO REGION (WITH SAFETY BUFFER) =====\n")

# -------------------------------
# CONFIG
# -------------------------------
CSV_POINTS = "row_truncated_data.csv"
INPUT_RASTER = "4_continuous_30m/rainfall_30yr_MAX_30m.tif"
OUTPUT_RASTER = "4_continuous_30m/rainfall_30yr_MAX_30m_clipped.tif"

PIXEL_SIZE = 30.0   # meters
BUFFER = PIXEL_SIZE # 1-pixel buffer to avoid edge loss
NODATA = -999

# -------------------------------
# READ CSV (NOTE: column names are misleading)
# Latitude  -> Easting (meters)
# Longitude -> Northing (meters)
# -------------------------------
df = pd.read_csv(CSV_POINTS)

eastings  = df["Latitude"].values
northings = df["Longitude"].values

# Bounding box + safety buffer
minx = eastings.min()  - BUFFER
maxx = eastings.max()  + BUFFER
miny = northings.min() - BUFFER
maxy = northings.max() + BUFFER

print("AOI extent (with buffer):")
print(f"  Easting (X):  {minx:.2f}  →  {maxx:.2f}")
print(f"  Northing (Y): {miny:.2f}  →  {maxy:.2f}")
print(f"  Buffer applied: {BUFFER} m")

# -------------------------------
# GDAL CLIP
# -------------------------------
cmd = [
    "gdalwarp",
    "-te", str(minx), str(miny), str(maxx), str(maxy),
    "-dstnodata", str(NODATA),
    "-multi",
    "-wm", "2048",
    INPUT_RASTER,
    OUTPUT_RASTER
]

print("\nRunning gdalwarp clip...")
subprocess.run(cmd, check=True)

print("\n✓ STEP 5 COMPLETED")
print("✓ Clipped raster saved to:")
print(f"  {OUTPUT_RASTER}")

# %% EXTRACT TO CSV (WITH COORDINATE FIX)

CSV_IN = "row_truncated_data.csv"
RASTER = "4_continuous_30m/rainfall_30yr_MAX_30m.tif"
CSV_OUT = "row_truncated_with_rainfall.csv"

print("Loading CSV...")
df = pd.read_csv(CSV_IN)
print("Rows:", len(df))

with rasterio.open(RASTER) as ds:
    rainfall = []
    missing = 0

    for _, row in df.iterrows():
        x = row["Latitude"]     # EASTING
        y = row["Longitude"]    # NORTHING

        val = list(ds.sample([(x, y)]))[0][0]

        if val == ds.nodata or np.isnan(val):
            rainfall.append(np.nan)
            missing += 1
        else:
            rainfall.append(val)

df["Rainfall_30yr_MAX_CONT"] = rainfall

print("\n===== SUMMARY =====")
print("Valid rainfall:", df["Rainfall_30yr_MAX_CONT"].notna().sum())
print("Missing rainfall:", missing)
print("Distinct values:", df["Rainfall_30yr_MAX_CONT"].nunique())
print(df["Rainfall_30yr_MAX_CONT"].describe())

df.to_csv(CSV_OUT, index=False)
print("\n CSV written:", CSV_OUT)

# %% JENKS NATURAL BREAKS (6 CLASSES) ON RAINFALL

# -------------------------------
# FILES
# -------------------------------
CSV_PROCESSED = "processed1_truncated_data.csv"
CSV_RAIN      = "row_truncated_with_rainfall.csv"
CSV_OUT       = "processed1_truncated_data_with_rainfall_jenks.csv"

RAIN_COL  = "Rainfall_30yr_MAX_CONT"
N_CLASSES = 6

# -------------------------------
# LOAD DATA
# -------------------------------
df_proc = pd.read_csv(CSV_PROCESSED)
df_rain = pd.read_csv(CSV_RAIN)

print("\n===== JENKS CLASSIFICATION (RAINFALL) =====")
print("Processed rows:", len(df_proc))
print("Rainfall rows :", len(df_rain))

# -------------------------------
# SAFETY CHECK: ROW + COORDINATE ALIGNMENT
# -------------------------------
assert len(df_proc) == len(df_rain), "Row count mismatch!"

coord_match = (
    (df_proc["Latitude"].values  == df_rain["Latitude"].values) &
    (df_proc["Longitude"].values == df_rain["Longitude"].values)
)

assert coord_match.all(), "Latitude / Longitude mismatch between files!"

print("✓ Coordinate alignment confirmed")

# -------------------------------
# JENKS NATURAL BREAKS
# -------------------------------
rainfall = df_rain[RAIN_COL].values

breaks = jenkspy.jenks_breaks(rainfall, n_classes=N_CLASSES)

print("\nJenks break values:")
for i in range(len(breaks) - 1):
    print(f"  Class {i+1}: {breaks[i]:.3f} → {breaks[i+1]:.3f}")

# Assign Jenks classes (1–6)
df_proc["Rainfall_Jenks"] = pd.cut(
    rainfall,
    bins=breaks,
    labels=list(range(1, N_CLASSES + 1)),
    include_lowest=True
).astype(int)

# -------------------------------
# QUICK SANITY OUTPUTS
# -------------------------------
print("\nRainfall Jenks class distribution:")
print(df_proc["Rainfall_Jenks"].value_counts().sort_index())

print("\nRainfall Jenks vs Landslide:")
print(pd.crosstab(df_proc["Rainfall_Jenks"], df_proc["Landslide"]))

# -------------------------------
# SAVE
# -------------------------------
df_proc.to_csv(CSV_OUT, index=False)

print("\n✓ Jenks rainfall classification completed")
print("✓ Output written to:", CSV_OUT)

# %%
# Tomek Links + Random Undersampling

print("\n===== TOMEK LINKS + UNDERSAMPLING =====\n")

# INPUT
CSV_IN = "processed1_truncated_data_with_rainfall_jenks.csv"

df = pd.read_csv(CSV_IN)

# Remove unwanted index column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print("Dataset loaded")
print("Total rows:", len(df))

print("\nOriginal Landslide distribution:")
print(df["Landslide"].value_counts())

# ---------------------------------------------
# APPLY TOMEK LINKS (ON FULL FEATURE SET)
# ---------------------------------------------
print("\nApplying Tomek Links undersampling...")

X_full = df.drop(columns=["Landslide"])
y_full = df["Landslide"]

tl = TomekLinks(sampling_strategy="auto", n_jobs=-1)

X_tomek, y_tomek = tl.fit_resample(X_full, y_full)

df_tomek = pd.concat(
    [X_tomek, y_tomek.rename("Landslide")],
    axis=1
)

print("\nAfter Tomek Links:")
print(df_tomek["Landslide"].value_counts())
print("Rows removed by Tomek:", len(df) - len(df_tomek))

# ---------------------------------------------
# RANDOM UNDERSAMPLING (CLASS BALANCING)
# ---------------------------------------------
print("\nApplying random undersampling to balance classes...")

df_ls  = df_tomek[df_tomek["Landslide"] == 2]
df_nls = df_tomek[df_tomek["Landslide"] == 1].sample(
    n=len(df_ls),
    random_state=1,
    replace=False
)

df_balanced = pd.concat([df_ls, df_nls]).sample(
    frac=1,
    random_state=1
).reset_index(drop=True)

print("\nFinal balanced class distribution:")
print(df_balanced["Landslide"].value_counts())

# ---------------------------------------------
# TRAIN / TEST SPLIT (STRATIFIED)
# ---------------------------------------------
X_final = df_balanced.drop(columns=["Latitude", "Longitude", "Landslide"])
Y_final = df_balanced["Landslide"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X_final,
    Y_final,
    test_size=0.30,
    random_state=1,
    stratify=Y_final
)

# ---------------------------------------------
# SAVE LAT/LON WITH LABELS (FOR MAPPING)
# ---------------------------------------------
Y_train_out = pd.concat(
    [
        Y_train.reset_index(drop=True),
        df_balanced.loc[Y_train.index, ["Latitude", "Longitude"]].reset_index(drop=True)
    ],
    axis=1
)

Y_test_out = pd.concat(
    [
        Y_test.reset_index(drop=True),
        df_balanced.loc[Y_test.index, ["Latitude", "Longitude"]].reset_index(drop=True)
    ],
    axis=1
)

# ---------------------------------------------
# FINAL SANITY CHECKS
# ---------------------------------------------
print("\nNaN checks:")
print("X_train:", X_train.isna().values.any())
print("Y_train:", Y_train_out.isna().values.any())
print("X_test :", X_test.isna().values.any())
print("Y_test :", Y_test_out.isna().values.any())

# ---------------------------------------------
# SAVE FILES
# ---------------------------------------------
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
Y_train_out.to_csv("Y_train.csv", index=False)
Y_test_out.to_csv("Y_test.csv", index=False)

print("\n✓ Files written successfully:")
print("  X_train.csv")
print("  X_test.csv")
print("  Y_train.csv")
print("  Y_test.csv")

# %% OVERSAMPLING 

CSV_IN = "processed1_truncated_data_with_rainfall_jenks.csv"
OUT_DIR = "Train-test-split-oversampling"

os.makedirs(OUT_DIR, exist_ok=True)

print("\n" + "="*80)
print("OVERSAMPLING: LOADING DATA")
print("="*80)

df = pd.read_csv(CSV_IN)

# Remove unwanted index column if present
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print("Total rows:", len(df))
print("\nOriginal Landslide distribution:")
print(df["Landslide"].value_counts())

if "Rainfall_Jenks" in df.columns:
    print("\nRainfall Jenks distribution:")
    print(df["Rainfall_Jenks"].value_counts().sort_index())

print("\nNaN check (entire dataset):", df.isna().sum().sum())

# %% FEATURE / LABEL SPLIT (OVERSAMPLING)

print("\nPreparing features and labels...")

coords = df[["Latitude", "Longitude"]].copy()

X = df.drop(columns=["Latitude", "Longitude", "Landslide"])
y = df["Landslide"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# %% STRATIFIED TRAIN–TEST SPLIT (BEFORE SMOTE)

print("\n" + "="*80)
print("TRAIN–TEST SPLIT (ORIGINAL IMBALANCED DATA)")
print("="*80)

X_train_orig, X_test, y_train_orig, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=1,
    stratify=y
)

print("\nTrain distribution (BEFORE SMOTE):")
print(y_train_orig.value_counts())

print("\nTest distribution (UNCHANGED):")
print(y_test.value_counts())

# %% APPLY SMOTE (TRAINING SET ONLY)

print("\n" + "="*80)
print("APPLYING SMOTE (TRAIN ONLY)")
print("="*80)

smote = SMOTE(
    sampling_strategy="auto",
    random_state=1,
    k_neighbors=5
)

X_train, y_train = smote.fit_resample(X_train_orig, y_train_orig)

print("\nTrain distribution (AFTER SMOTE):")
print(pd.Series(y_train).value_counts())

print("Synthetic samples added:", len(X_train) - len(X_train_orig))

print("\n" + "="*80)
print("FINAL SANITY CHECKS")
print("="*80)

print("NaN checks:")
print("X_train:", X_train.isna().values.any())
print("X_test :", X_test.isna().values.any())
print("y_train:", pd.Series(y_train).isna().any())
print("y_test :", y_test.isna().any())

assert list(X_train.columns) == list(X_test.columns), "Feature mismatch!"
print("✓ Feature alignment confirmed")

print("\n" + "="*80)
print("SAVING OVERSAMPLING FILES")
print("="*80)

Y_train = pd.DataFrame(y_train, columns=["Landslide"])
Y_test  = pd.DataFrame(y_test).reset_index(drop=True)

X_train.to_csv(f"{OUT_DIR}/X_train_smote.csv", index=False)
X_test.to_csv(f"{OUT_DIR}/X_test_smote.csv", index=False)
Y_train.to_csv(f"{OUT_DIR}/Y_train_smote.csv", index=False)
Y_test.to_csv(f"{OUT_DIR}/Y_test_smote.csv", index=False)

print("✓ Files written:")
print("  X_train_smote.csv")
print("  X_test_smote.csv")
print("  Y_train_smote.csv")
print("  Y_test_smote.csv")

print("\n" + "="*80)
print("OVERSAMPLING PIPELINE COMPLETE")
print("="*80)