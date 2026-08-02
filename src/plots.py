import pandas as pd
import numpy as np

# Class ordering used to ensure consistent plotting order
CLASS_ORDER = [5, 4, 3, 2, 1]  # Very High → Very Low

def compute_ratios_from_csv(
    susceptibility_csv,
    processed_csv,
    class_col="LSM_Class",
    landslide_col="Landslide"
):
    # Load CSV inputs: susceptibility map classes and processed sample table
    sus_df = pd.read_csv(susceptibility_csv)
    proc_df = pd.read_csv(processed_csv)

    # Safety check
    assert len(sus_df) == len(proc_df), "Row mismatch between CSVs"

    # ---------- FIGURE 42: landslide pixel ratio ----------
    # Identify pixels that experienced landslides and get their susceptibility classes
    landslide_mask = proc_df[landslide_col] == 2
    ls_classes = sus_df.loc[landslide_mask, class_col]

    # Compute percentage distribution of landslide pixels across classes
    ls_counts = ls_classes.value_counts().reindex(CLASS_ORDER, fill_value=0)
    ls_ratio = (ls_counts / ls_counts.sum()) * 100

    # ---------- FIGURE 43: study area ratio ----------
    # Compute percentage area covered by each susceptibility class (study area baseline)
    area_counts = sus_df[class_col].value_counts().reindex(CLASS_ORDER, fill_value=0)
    area_ratio = (area_counts / area_counts.sum()) * 100

    return ls_ratio.values, area_ratio.values

processed_csv = "processed1_truncated_data.csv"

# Compute proportional landslide and area ratios for three model outputs
xgb_ls, xgb_area = compute_ratios_from_csv(
    "feature/LSM_XGBoost_Oversampling_Probabilities_Jenks.csv",
    processed_csv
)

cat_ls, cat_area = compute_ratios_from_csv(
    "feature/LSM_CatBoost_Oversampling_Probabilities_Jenks.csv",
    processed_csv
)

lgb_ls, lgb_area = compute_ratios_from_csv(
    "feature/LSM_LightGBM_Oversampling_Probabilities_Jenks.csv",
    processed_csv
)

import matplotlib.pyplot as plt
import numpy as np

# Labels and positions for the bar charts
classes = ["Very High", "High", "Medium", "Low", "Very Low"]
x = np.arange(len(classes))
width = 0.25

# Plot proportional landslide pixel percentages for each model
plt.figure(figsize=(9, 5))

plt.bar(x - width, xgb_ls, width, label="XGBoost Ratio (%)", color="red")
plt.bar(x,         cat_ls, width, label="CatBoost Ratio (%)", color="green")
plt.bar(x + width, lgb_ls, width, label="LightGBM Ratio (%)", color="royalblue")

plt.xticks(x, classes)
plt.xlabel("Landslide Susceptibility Classes (Oversampling)")
plt.ylabel("Proportional Landslide Pixel (%)")
plt.legend()
plt.tight_layout()
plt.savefig("Proportional_Landslide_Pixels_os.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot proportional study-area percentages for each model (baseline area coverage)
plt.figure(figsize=(9, 5))

plt.bar(x - width, xgb_area, width, label="XGBoost Ratio (%)", color="red")
plt.bar(x,         cat_area, width, label="CatBoost Ratio (%)", color="green")
plt.bar(x + width, lgb_area, width, label="LightGBM Ratio (%)", color="royalblue")

plt.xticks(x, classes)
plt.xlabel("Landslide Susceptibility Classes (Oversampling)")
plt.ylabel("Proportional Study Area (%)")
plt.legend()
plt.tight_layout()
plt.savefig("Proportional_Study_Area_os.png", dpi=300, bbox_inches="tight")
plt.show()

df_summary = pd.DataFrame({
    "Class": classes,
    "XGBoost_Landslide_%": xgb_ls,
    "CatBoost_Landslide_%": cat_ls,
    "LightGBM_Landslide_%": lgb_ls,
    "XGBoost_Area_%": xgb_area,
    "CatBoost_Area_%": cat_area,
    "LightGBM_Area_%": lgb_area
})

print(df_summary)
