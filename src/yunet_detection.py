import cv2
import os

MODEL_PATH = "models/face_detection_yunet.onnx"


def detect_faces_yunet(image):

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"YuNet model not found: {MODEL_PATH}"
        )

    height, width = image.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (width, height),
        0.9,   # Confidence threshold
        0.3,   # NMS threshold
        5000   # Top K
    )

    _, faces = detector.detect(image)

    annotated = image.copy()
    face_locations = []

    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4].astype(int)
            confidence = float(face[-1])

            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Face: {confidence:.2f}",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            face_locations.append(
                (x, y, w, h, confidence)
            )

    return annotated, face_locations


def analyze_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print("Unable to load image.")
        return

    annotated, faces = detect_faces_yunet(image)

    print(f"Faces detected using YuNet: {len(faces)}")

    cv2.imshow("YuNet Face Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    image_path = input("Enter image path: ")
    analyze_image(image_path)