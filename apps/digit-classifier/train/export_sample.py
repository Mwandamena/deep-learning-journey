import pandas as pd
import numpy as np
from PIL import Image

CSV_PATH = "data/train.csv" 
OUTPUT_PATH = "sample_digit.png"

df = pd.read_csv(CSV_PATH)
row = df.sample(1).iloc[0]

label = int(row["label"])
pixels = row.drop("label").values.astype(np.uint8).reshape(28, 28)

img = Image.fromarray(pixels, mode="L")
img.save(OUTPUT_PATH)

print(f"True label: {label}")
print(f"Saved to: {OUTPUT_PATH}")
print(f"Image size: {img.size}, mode: {img.mode}")