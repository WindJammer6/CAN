# Using the "wmconvert" virtual environment since it requires different versions of numpy and pandas library

import numpy as np
import pandas as pd
import cv2

PKL_PATH = "WM811K/LSWMD.pkl"
OUT_NPZ  = "WM811K/wm811k-64.npz"

# WM811K defect class mapping
STR_TO_ID = {
    "Center": 0,
    "Donut": 1,
    "Edge-Loc": 2,
    "Edge-Ring": 3,
    "Loc": 4,
    "Random": 5,
    "Scratch": 6,
    "Near-full": 7,
    "none": 8,
}

def unwrap_label(x):
    """Handles nested lists/arrays like [[Center]] or [[]]"""
    while isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return "none"  # Empty list -> treat as 'none'
        if len(x) == 1:
            x = x[0]
        else:
            # Multiple labels - take first one or handle differently
            x = x[0]
    return x if x else "none"  # Handle None or empty string

def to_64x64_normalized(a):
    """
    Convert wafer map to 64x64 and normalize to 0-255 range for proper image processing.
    
    WM811K stores wafer maps with values:
    - 0: background
    - 1: normal dies
    - 2: defective dies
    
    We scale these to proper grayscale intensities for image processing.
    """
    arr = np.array(a, dtype=np.float32)  # Use float32 for processing
    if arr.ndim != 2:
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"waferMap not 2D: {arr.shape}")
    
    # Resize to 64x64
    arr = cv2.resize(arr, (64, 64), interpolation=cv2.INTER_NEAREST)
    
    # Normalize to 0-255 range for proper image processing
    # This ensures compatibility with ImageNet pretrained models
    # Method 1: Map discrete values to grayscale intensities
    # 0 → 0 (background/black)
    # 1 → 127 (normal dies/gray) 
    # 2 → 255 (defective dies/white)
    if arr.max() <= 2:
        # Standard WM811K encoding
        arr = arr * 127.5  # 0→0, 1→127.5, 2→255
    else:
        # Some wafer maps might have value 3 or other encodings
        # Scale proportionally to 0-255
        arr = (arr / arr.max()) * 255.0
    
    return arr.astype(np.uint8)

# Load and clean
df = pd.read_pickle(PKL_PATH)
print(f"Original size: {len(df)}")

# Drop rows with missing waferMap or failureType
df = df.dropna(subset=["waferMap"]).reset_index(drop=True)
print(f"After dropping NaN waferMap: {len(df)}")

# Check for empty failureType
df = df[df["failureType"].apply(lambda x: x is not None and 
                                 (not isinstance(x, (list, np.ndarray)) or len(x) > 0))].reset_index(drop=True)
print(f"After dropping empty failureType: {len(df)}")

# Resize and normalize wafer maps
print("\nProcessing wafer maps...")
X = np.stack(df["waferMap"].apply(to_64x64_normalized).to_list(), axis=0)

# Process labels
y_raw = df["failureType"].apply(unwrap_label).to_list()

# Convert to integers
if isinstance(y_raw[0], (int, np.integer)):
    y = np.array([int(v) for v in y_raw], dtype=np.int64)
else:
    y_str = np.array([str(v) for v in y_raw], dtype=object)
    
    # Check for unknown labels
    unknown = sorted(set(y_str) - set(STR_TO_ID.keys()))
    if unknown:
        print(f"WARNING: Unknown labels found: {unknown}")
        print("Mapping them to 'none' (class 8)")
        # Map unknown labels to 'none'
        y_str = np.array([s if s in STR_TO_ID else 'none' for s in y_str])
    
    y = np.array([STR_TO_ID[s] for s in y_str], dtype=np.int64)

# Print statistics
print("\n" + "="*60)
print("Conversion Summary:")
print("="*60)
print(f"X shape: {X.shape}, dtype: {X.dtype}")
print(f"  Value range: [{X.min()}, {X.max()}]")
print(f"  Expected: [0, 255] for proper image processing")
print(f"\ny shape: {y.shape}, dtype: {y.dtype}")
print(f"  Unique classes: {np.unique(y)}")
print("\nClass distribution:")
unique, counts = np.unique(y, return_counts=True)
for cls_id, count in zip(unique, counts):
    # Reverse lookup class name
    cls_name = [k for k, v in STR_TO_ID.items() if v == cls_id][0]
    print(f"  Class {cls_id} ({cls_name:12s}): {count:6d} samples")

# Verify pixel value distribution
print("\nPixel value distribution (first 1000 images):")
unique_vals, val_counts = np.unique(X[:1000].flatten(), return_counts=True)
for val, count in zip(unique_vals[:10], val_counts[:10]):  # Show first 10 unique values
    print(f"  Value {val:3.0f}: {count:8d} pixels")

# Save
np.savez_compressed(OUT_NPZ, data=X, label=y)
print(f"\n✓ Saved to: {OUT_NPZ}")
print("  Data is now in uint8 format with values 0-255")
print("  Ready for PIL Image processing and ImageNet normalization")
print("="*60)