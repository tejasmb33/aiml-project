import pandas as pd

CSV_FILE = "evaluation_data/image_quality_results.csv"
OUTPUT_FILE = "results/tables/condition_summary.csv"

df = pd.read_csv(CSV_FILE)

summary = df.groupby("Condition")[["Brightness", "Contrast", "Sharpness"]].mean().round(2)

summary["Image_Count"] = df.groupby("Condition").size()

summary.to_csv(OUTPUT_FILE)

print("\nCondition-wise Summary:\n")
print(summary)

print(f"\nSummary saved to {OUTPUT_FILE}")