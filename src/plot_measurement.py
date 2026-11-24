import csv
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    log_file = Path("captured_frames/measurement_log.csv")

    if not log_file.exists():
        print("Measurement log not found.")
        return

    brightness = []
    frame_times = []

    with open(log_file, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            brightness.append(float(row["brightness"]))
            frame_times.append(float(row["frame_time_ms"]))

    if not brightness:
        print("No data available.")
        return

    # Brightness graph
    plt.figure()
    plt.plot(brightness)
    plt.title("Brightness Measurements")
    plt.xlabel("Frame Number")
    plt.ylabel("Brightness")
    plt.tight_layout()
    plt.savefig("captured_frames/brightness_plot.png")
    plt.show()

    # Frame time graph
    plt.figure()
    plt.plot(frame_times)
    plt.title("Frame Processing Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Time (ms)")
    plt.tight_layout()
    plt.savefig("captured_frames/frame_time_plot.png")
    plt.show()

    print("Graphs saved successfully.")

if __name__ == "__main__":
    main()