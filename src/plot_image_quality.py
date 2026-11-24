import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_FILE = "evaluation_data/image_quality_results.csv"
OUTPUT_DIR = "results/graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

metrics = ["Brightness", "Contrast", "Sharpness"]

for metric in metrics:
    plt.figure()
    df.groupby("Condition")[metric].mean().plot(kind="bar")

    plt.title(f"Average {metric} by Condition")
    plt.xlabel("Condition")
    plt.ylabel(metric)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"{metric.lower()}_comparison.png")
    plt.savefig(output_path)
    plt.close()

    print(f"Saved: {output_path}")

print("All graphs generated successfully!")