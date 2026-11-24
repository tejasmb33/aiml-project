
import cv2
import numpy as np

from face_detection import detect_faces
from preprocessing import preprocess_image


def analyze_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError("Unable to load image.")

    # Preprocessing
    resized, gray, normalized = preprocess_image(image)

    # Face detection
    annotated, faces = detect_faces(resized)

    # Image quality metrics
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    results = {
        "face_count": len(faces),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "image_width": resized.shape[1],
        "image_height": resized.shape[0]
    }

    return annotated, normalized, results


if __name__ == "__main__":

    path = input("Enter image path: ").strip()

    annotated, normalized, results = analyze_image(path)

    print("\n--- Analysis Results ---")

    for key, value in results.items():
        print(f"{key}: {value}")

    cv2.imshow("Face Detection", annotated)
    cv2.imshow("Normalized Image", normalized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
