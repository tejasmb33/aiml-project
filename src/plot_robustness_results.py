import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_FILE = "results/tables/robustness_results.csv"
OUTPUT_FILE = "results/graphs/robustness_comparison.png"


def plot_results():

    df = pd.read_csv(INPUT_FILE)

    summary = df.groupby("condition")["faces_detected"].mean()

    os.makedirs("results/graphs", exist_ok=True)

    plt.figure(figsize=(9, 5))

    summary.plot(kind="bar")

    plt.title("Face Detection Output Across Image Conditions")
    plt.xlabel("Image Condition")
    plt.ylabel("Average Faces Detected")
    plt.xticks(rotation=25, ha="right")

    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=300)
    plt.show()

    print(f"Graph saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    plot_results()