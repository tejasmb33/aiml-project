
import pandas as pd
import matplotlib.pyplot as plt
import os

input_file = "results/tables/face_detection_evaluation.csv"
output_dir = "results/graphs"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file)

plt.figure(figsize=(8, 5))

plt.bar(
    df["condition"],
    df["detection_rate_percent"]
)

plt.ylabel("Detection Output Rate (%)")
plt.xlabel("Image Condition")
plt.title("Face Detector Output Rate by Condition")
plt.ylim(0, 110)

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "face_detection_output_rate.png")
)

plt.show()

print("Graph saved successfully.")
