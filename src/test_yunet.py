import cv2
import numpy as np
import os

MODEL_PATH = "models/face_detection_yunet.onnx"

def test_yunet():

    print("Checking model file...")

    if not os.path.exists(MODEL_PATH):
        print("FAILED: Model file missing.")
        return

    print("Model file exists.")

    try:

        detector = cv2.FaceDetectorYN.create(
            MODEL_PATH,
            "",
            (320, 320),
            0.9,
            0.3,
            5000
        )

        print("YuNet model loaded successfully.")

        # Synthetic image used only to test execution
        image = np.zeros((320, 320, 3), dtype=np.uint8)

        _, faces = detector.detect(image)

        print("Detector executed successfully.")
        print(
            "Detection output:",
            "None" if faces is None else faces.shape
        )

        print("\nYU NET EXECUTION TEST PASSED.")

    except Exception as e:

        print("FAILED:", e)


if __name__ == "__main__":
    test_yunet()