
import cv2
import time
import os
import csv

from face_detection import detect_faces

DATASET_DIR = "evaluation_data"
OUTPUT_FILE = "results/tables/performance_results.csv"


def run_performance_test():

    os.makedirs("results/tables", exist_ok=True)

    results = []

    for root, dirs, files in os.walk(DATASET_DIR):

        for file in files:

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Skipped: {file}")
                    continue

                start_time = time.perf_counter()

                annotated, faces = detect_faces(image)

                end_time = time.perf_counter()

                processing_time = end_time - start_time

                fps_estimate = (
                    1 / processing_time
                    if processing_time > 0
                    else 0
                )

                row = {
                    "image": file,
                    "condition": os.path.basename(root),
                    "processing_time_ms": round(
                        processing_time * 1000, 2
                    ),
                    "estimated_fps": round(fps_estimate, 2),
                    "faces_detected": len(faces)
                }

                results.append(row)

                print(
                    f"{file} | "
                    f"{row['processing_time_ms']} ms | "
                    f"{row['estimated_fps']} FPS"
                )

    if results:

        with open(OUTPUT_FILE, "w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=results[0].keys()
            )

            writer.writeheader()
            writer.writerows(results)

        print("\nPerformance evaluation completed.")
        print(f"Results saved to: {OUTPUT_FILE}")

    else:

        print("No valid images found.")


if __name__ == "__main__":
    run_performance_test()
