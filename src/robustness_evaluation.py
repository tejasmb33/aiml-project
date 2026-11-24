import cv2
import os
import csv

from face_detection import detect_faces

DATASET_DIR = "evaluation_data"
OUTPUT_FILE = "results/tables/robustness_results.csv"


def run_robustness_test():

    os.makedirs("results/tables", exist_ok=True)

    results = []

    transformations = {
        "Original": lambda img: img,
        "Low Brightness": lambda img: cv2.convertScaleAbs(
            img, alpha=0.5, beta=-30
        ),
        "High Brightness": lambda img: cv2.convertScaleAbs(
            img, alpha=1.3, beta=40
        ),
        "Blur": lambda img: cv2.GaussianBlur(
            img, (5, 5), 0
        )
    }

    for root, dirs, files in os.walk(DATASET_DIR):

        for file in files:

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)

                if image is None:
                    continue

                for condition, transform in transformations.items():

                    test_image = transform(image)

                    annotated, faces = detect_faces(test_image)

                    results.append({
                        "image": file,
                        "condition": condition,
                        "faces_detected": len(faces)
                    })

                    print(
                        f"{file} | {condition} | "
                        f"Faces: {len(faces)}"
                    )

    if results:

        with open(OUTPUT_FILE, "w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=results[0].keys()
            )

            writer.writeheader()
            writer.writerows(results)

        print("\nRobustness testing completed.")
        print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_robustness_test()