import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "evaluation_data/image_quality_results.csv"

df = pd.read_csv(CSV_FILE)

metrics = ["Brightness", "Contrast", "Sharpness"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, metric in enumerate(metrics):
    summary = df.groupby("Condition")[metric].mean()

    axes[i].bar(summary.index, summary.values)
    axes[i].set_title(f"Average {metric} by Condition")
    axes[i].set_xlabel("Condition")
    axes[i].set_ylabel(metric)

plt.tight_layout()

plt.savefig("results/graphs/combined_comparison.png", dpi=300)

plt.show()

print("Combined graph saved successfully!")