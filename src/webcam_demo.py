import cv2
import time
import csv
from pathlib import Path
from datetime import datetime

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Error: Cannot access camera")
        return

    log_file = Path("captured_frames") / "measurement_log.csv"
    log_file.parent.mkdir(exist_ok=True)

    if not log_file.exists():
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "brightness", "frame_time_ms"])

    print("Camera started.")
    print("Press Q to quit.")

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            print("Unable to capture frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        processing_time = (time.time() - start_time) * 1000

        cv2.putText(frame, f"Brightness: {brightness:.1f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"Frame time: {processing_time:.1f} ms",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.imshow("Image Quality Demo", frame)

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                round(float(brightness), 2),
                round(processing_time, 2)
            ])

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Measurement log saved to: {log_file}")

if __name__ == "__main__":
    main()