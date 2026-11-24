import sys
import cv2
import numpy as np
import pandas as pd

print("=" * 40)
print("ENVIRONMENT CHECK")
print("=" * 40)

print("Python:", sys.version.split()[0])
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)

print("\nCore libraries loaded successfully.")
print("Environment check passed.")