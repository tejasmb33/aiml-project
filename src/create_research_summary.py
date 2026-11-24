import csv
import statistics
from pathlib import Path

log_file = Path("captured_frames/measurement_log.csv")

if not log_file.exists():
    print("Measurement log not found!")
    exit()

brightness = []
frame_times = []

with open(log_file, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        try:
            brightness.append(float(row["brightness"]))
            frame_times.append(float(row["frame_time_ms"]))
        except (ValueError, KeyError):
            continue

if not brightness:
    print("No valid measurements found.")
    exit()

print("\n===== RESEARCH MEASUREMENT SUMMARY =====")
print(f"Total observations: {len(brightness)}")
print(f"Average brightness: {statistics.mean(brightness):.2f}")
print(f"Minimum brightness: {min(brightness):.2f}")
print(f"Maximum brightness: {max(brightness):.2f}")

if frame_times:
    print(f"Average processing time: {statistics.mean(frame_times):.2f} ms")

print("\nSummary generated successfully.")