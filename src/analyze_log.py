import csv
from pathlib import Path

def main():
    log_file = Path("captured_frames/measurement_log.csv")

    if not log_file.exists():
        print("Measurement log not found.")
        return

    brightness_values = []
    frame_times = []

    with open(log_file, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            brightness_values.append(float(row["brightness"]))
            frame_times.append(float(row["frame_time_ms"]))

    if not brightness_values:
        print("No measurements found.")
        return

    print("\n===== Measurement Summary =====")
    print(f"Total measurements: {len(brightness_values)}")
    print(f"Average brightness: {sum(brightness_values) / len(brightness_values):.2f}")
    print(f"Minimum brightness: {min(brightness_values):.2f}")
    print(f"Maximum brightness: {max(brightness_values):.2f}")
    print(f"Average frame time: {sum(frame_times) / len(frame_times):.2f} ms")
    print("==============================")

if __name__ == "__main__":
    main()