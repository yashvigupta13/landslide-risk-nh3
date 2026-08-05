import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

# --------------------------------------------------
# VULNERABILITY VALUES (FROM YOUR FINAL TABLE)
# --------------------------------------------------


vulnerability_lookup = {
    "Cropland": 0.000412811292750107,
    "Barren land": 0.0,
    "Dense Forest": 0.00110815319380855,
    "Grassland": 0.00352582858346913,
    "Settlement": 0.0,
    "Waterbody": 0.00223856675984424,
    "Road": 1.0
}

# --------------------------------------------------
# Land-cover code → vulnerability mapping
# --------------------------------------------------

lc_to_vul = {
    1: vulnerability_lookup["Cropland"],
    2: vulnerability_lookup["Waterbody"],
    3: vulnerability_lookup["Settlement"],
    4: vulnerability_lookup["Barren land"],
    5: vulnerability_lookup["Dense Forest"],
    6: vulnerability_lookup["Grassland"],
    7: vulnerability_lookup["Grassland"]
}

# --------------------------------------------------
# STEP 1: Read Land_Cover raster
# --------------------------------------------------

with rasterio.open("Land_Cover.tif") as lc_src:
    landcover = lc_src.read(1)
    meta = lc_src.meta.copy()

print("STEP 1: Land_Cover loaded")
print("Shape:", landcover.shape)
print("Unique land-cover values:", np.unique(landcover))
print("-" * 50)

# --------------------------------------------------
# STEP 2: Initialize Vulnerability raster
# --------------------------------------------------

vulnerability = np.zeros_like(landcover, dtype=np.float32)

for lc_class, vul_value in lc_to_vul.items():
    count = np.sum(landcover == lc_class)
    vulnerability[landcover == lc_class] = vul_value
    print(f"Assigned vulnerability {vul_value} to LC {lc_class} ({count} pixels)")

print("-" * 50)
print("STEP 2 sanity check")
print("Min vulnerability:", vulnerability.min())
print("Max vulnerability:", vulnerability.max())
print("-" * 50)

# --------------------------------------------------
# STEP 3 (CORRECTED): Apply Road overlay using distance values
# --------------------------------------------------

with rasterio.open("Road.tif") as road_src:
    road = road_src.read(1)

print("STEP 3: Road raster loaded")
print("Unique road values:", np.unique(road))

road_mask = np.isin(road, [80, 120, 160, 200, 240])

print("Road-influenced pixels:", np.sum(road_mask))
print("-" * 50)

# Override vulnerability where road influence exists
vulnerability[road_mask] = vulnerability_lookup["Road"]

print("STEP 3 sanity check")
print("Max vulnerability after road overlay:", vulnerability.max())
print("-" * 50)


# --------------------------------------------------
# STEP 4: Save Vulnerability.tif
# --------------------------------------------------

meta.update(dtype=rasterio.float32, count=1)

with rasterio.open("Vulnerability.tif", "w", **meta) as dst:
    dst.write(vulnerability, 1)

print("STEP 4: Vulnerability.tif written")
print("-" * 50)

# --------------------------------------------------
# STEP 5: Read Hazard raster
# --------------------------------------------------

with rasterio.open("Hazard_undersampling.tif") as hz_src:
    hazard = hz_src.read(1)
    hz_meta = hz_src.meta.copy()
    hz_transform = hz_src.transform
    hz_crs = hz_src.crs
    hz_nodata = hz_src.nodata  # -999.0

print("STEP 5: Hazard raster loaded")
print("Shape:", hazard.shape)
print("Min hazard:", hazard.min())
print("Max hazard:", hazard.max())
print("-" * 50)

# --------------------------------------------------
# STEP 6: Align Vulnerability grid to Hazard grid, then sanity check
# Land_Cover origin is offset ~15 m from the Hazard/Susceptibility/Rainfall
# grid, causing a 1-row x 1-col size difference.  Reproject-match fixes it.
# --------------------------------------------------

if hazard.shape != vulnerability.shape:
    print("STEP 6: Grid offset detected -- reprojecting Vulnerability to match Hazard grid...")
    with rasterio.open("Vulnerability.tif") as vul_src:
        vulnerability_matched = np.zeros(hazard.shape, dtype=np.float32)
        reproject(
            source=rasterio.band(vul_src, 1),
            destination=vulnerability_matched,
            src_transform=vul_src.transform,
            src_crs=vul_src.crs,
            dst_transform=hz_transform,
            dst_crs=hz_crs,
            resampling=Resampling.nearest
        )
    vulnerability = vulnerability_matched

    # Overwrite Vulnerability.tif with the aligned version
    aligned_meta = hz_meta.copy()
    aligned_meta.update(dtype=rasterio.float32, count=1, nodata=0.0)
    with rasterio.open("Vulnerability.tif", "w", **aligned_meta) as dst:
        dst.write(vulnerability, 1)
    print("STEP 6: Vulnerability.tif re-saved on Hazard grid")

assert hazard.shape == vulnerability.shape, "ERROR: Raster dimensions do not match"

print("STEP 6: Raster alignment OK")
print("-" * 50)

# --------------------------------------------------
# STEP 7: Compute Risk raster
# Risk = Hazard x Vulnerability
# Hazard nodata pixels (-999) are masked out before multiplication
# and restored in the output so they are not treated as valid data.
# --------------------------------------------------

hazard_nodata_mask = (hazard == hz_nodata)

# Temporarily zero out nodata cells so they don't corrupt the product
hazard_safe = np.where(hazard_nodata_mask, 0.0, hazard)
risk = hazard_safe * vulnerability

# Restore nodata in output wherever Hazard was nodata
risk[hazard_nodata_mask] = hz_nodata

print("STEP 7 sanity check")
valid_risk = risk[~hazard_nodata_mask]
print("Min risk (valid):", np.nanmin(valid_risk))
print("Max risk (valid):", np.nanmax(valid_risk))
print("-" * 50)

# --------------------------------------------------
# STEP 8: Save Risk.tif
# --------------------------------------------------

hz_meta.update(dtype=rasterio.float32, count=1)

with rasterio.open("Risk.tif", "w", **hz_meta) as dst:
    dst.write(risk, 1)

print("STEP 8: Risk.tif written successfully")
print("PROCESS COMPLETE")
