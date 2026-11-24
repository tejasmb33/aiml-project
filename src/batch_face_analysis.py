import os
import csv

from analysis_pipeline import analyze_image

DATASET_DIR = "evaluation_data"
OUTPUT_FILE = "results/tables/face_analysis_results.csv"


def run_batch_analysis():
    os.makedirs("results/tables", exist_ok=True)
    results_list = []

    for condition in os.listdir(DATASET_DIR):
        condition_path = os.path.join(DATASET_DIR, condition)

        if not os.path.isdir(condition_path):
            continue

        for root, dirs, files in os.walk(condition_path):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(root, file)

                    try:
                        annotated, normalized, results = analyze_image(image_path)

                        row = {
                            "condition": condition,
                            "image": file,
                            "face_count": results["face_count"],
                            "brightness": results["brightness"],
                            "contrast": results["contrast"],
                            "sharpness": results["sharpness"],
                            "width": results["image_width"],
                            "height": results["image_height"]
                        }

                        results_list.append(row)
                        print(f"Processed: {file}")

                    except Exception as error:
                        print(f"Error processing {file}: {error}")

    if results_list:
        with open(OUTPUT_FILE, "w", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=results_list[0].keys()
            )
            writer.writeheader()
            writer.writerows(results_list)

        print(f"\nCompleted: {len(results_list)} images processed.")
        print(f"CSV saved to: {OUTPUT_FILE}")
    else:
        print("No images processed.")


if __name__ == "__main__":
    run_batch_analysis()
