import cv2
import os
import csv

ROOT = "evaluation_data"

def analyze_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()
    contrast = gray.std()
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    return brightness, contrast, sharpness


def main():
    results = []

    for condition in os.listdir(ROOT):
        condition_path = os.path.join(ROOT, condition)

        if not os.path.isdir(condition_path):
            continue

        for root, _, files in os.walk(condition_path):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(root, file)

                    measurements = analyze_image(image_path)

                    if measurements:
                        results.append([
                            condition,
                            file,
                            round(measurements[0], 2),
                            round(measurements[1], 2),
                            round(measurements[2], 2)
                        ])

    output_file = "evaluation_data/image_quality_results.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Condition",
            "Image",
            "Brightness",
            "Contrast",
            "Sharpness"
        ])
        writer.writerows(results)

    print(f"Processed {len(results)} images")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()