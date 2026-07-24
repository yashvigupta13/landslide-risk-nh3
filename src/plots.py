import pandas as pd
import numpy as np

CLASS_ORDER = [5, 4, 3, 2, 1]  # Very High → Very Low

def compute_ratios_from_csv(
    susceptibility_csv,
    processed_csv,
    class_col="LSM_Class",
    landslide_col="Landslide"
):
    # Load data
    sus_df = pd.read_csv(susceptibility_csv)
    proc_df = pd.read_csv(processed_csv)

    # Safety check
    assert len(sus_df) == len(proc_df), "Row mismatch between CSVs"

    # ---------- FIGURE 42: landslide pixel ratio ----------
    landslide_mask = proc_df[landslide_col] == 2
    ls_classes = sus_df.loc[landslide_mask, class_col]

    ls_counts = ls_classes.value_counts().reindex(CLASS_ORDER, fill_value=0)
    ls_ratio = (ls_counts / ls_counts.sum()) * 100

    # ---------- FIGURE 43: study area ratio ----------
    area_counts = sus_df[class_col].value_counts().reindex(CLASS_ORDER, fill_value=0)
    area_ratio = (area_counts / area_counts.sum()) * 100

    return ls_ratio.values, area_ratio.values

processed_csv = "processed1_truncated_data.csv"

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

classes = ["Very High", "High", "Medium", "Low", "Very Low"]
x = np.arange(len(classes))
width = 0.25

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
