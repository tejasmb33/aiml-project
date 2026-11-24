import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "captured_frames" / "measurement_log.csv"

if not LOG_FILE.exists():
    print("Measurement log not found.")
    exit()

df = pd.read_csv(LOG_FILE)

if df.empty:
    print("No measurements available.")
    exit()

print("\n===== RESEARCH DASHBOARD =====")
print(f"Observations: {len(df)}")
print(f"Average brightness: {df['brightness'].mean():.2f}")
print(f"Average frame time: {df['frame_time_ms'].mean():.2f} ms")

plt.figure()
plt.plot(df["brightness"])
plt.title("Brightness Measurements")
plt.xlabel("Observation")
plt.ylabel("Brightness")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df["frame_time_ms"])
plt.title("Processing Time Measurements")
plt.xlabel("Observation")
plt.ylabel("Processing Time (ms)")
plt.tight_layout()
plt.show()