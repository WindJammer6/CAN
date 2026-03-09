# Using the "wmconvert" virtual environment since it requires different versions of numpy and pandas library

# Filtering out wafer maps with only 1 defect class (aka non-mixed) 
import numpy as np
import cv2

INPUT_NPZ = "MixedWM38/Wafer_Map_Datasets.npz"
OUTPUT_NPZ = "MixedWM38/mixed2m38-64-single.npz"

# MixedWM38 defect class mapping (aligned with WM811K convention)
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

# Load the data
print("Loading MixedWM38 dataset...")
data = np.load(INPUT_NPZ)

X_raw = data['arr_0']  # (38015, 52, 52)
y_multihot = data['arr_1']  # (38015, 8)

print(f"\nOriginal dataset:")
print(f"  X shape: {X_raw.shape}, dtype: {X_raw.dtype}")
print(f"  X value range: [{X_raw.min()}, {X_raw.max()}]")
print(f"  y shape: {y_multihot.shape}, dtype: {y_multihot.dtype}")

# Check label format
print(f"\nFirst 10 labels:")
for i in range(min(10, len(y_multihot))):
    n_defects = np.sum(y_multihot[i])
    print(f"  Sample {i}: {y_multihot[i]} (num_defects: {n_defects})")

# Analyze defect distribution
defect_counts = np.sum(y_multihot, axis=1)
print(f"\nDefect type distribution:")
print(f"  Single-type (sum=1): {np.sum(defect_counts == 1)} samples")
print(f"  Mixed-type (sum>1):  {np.sum(defect_counts > 1)} samples")
print(f"  No defects (sum=0):  {np.sum(defect_counts == 0)} samples")

# Filter: keep single-type defects AND no-defect (none) class
single_or_none_mask = (defect_counts == 1) | (defect_counts == 0)

print(f"\n{'='*60}")
print("Filtering to single-type defects + 'none' class...")
print(f"{'='*60}")

X_filtered_52 = X_raw[single_or_none_mask]
y_filtered_multihot = y_multihot[single_or_none_mask]

print(f"Kept: {len(X_filtered_52)} samples")
print(f"Removed (mixed-type): {len(X_raw) - len(X_filtered_52)} samples")

# Convert multi-hot to integer labels
# [0, 1, 0, 0, 0, 0, 0, 0] -> 1
# [0, 0, 0, 0, 0, 0, 0, 0] -> 8 (none class)
y_filtered = np.zeros(len(y_filtered_multihot), dtype=np.int64)

for i in range(len(y_filtered_multihot)):
    if np.sum(y_filtered_multihot[i]) == 0:
        # All zeros = "none" class (class 8)
        y_filtered[i] = 8
    else:
        # Single defect = use argmax to get class 0-7
        y_filtered[i] = np.argmax(y_filtered_multihot[i])

# Remap classes to align with WM811K convention:
# Original MixedWM38: Near-full=5, Random=7  
# Target WM811K:      Random=5, Near-full=7
# We need to swap classes 5 and 7
print("\nRemapping classes to align with WM811K convention...")
print("  Swapping: class 5 (Near-full) <-> class 7 (Random)")
remap_mask_5_to_7 = (y_filtered == 5)  # Original Near-full samples
remap_mask_7_to_5 = (y_filtered == 7)  # Original Random samples
y_filtered[remap_mask_5_to_7] = 7  # Near-full -> class 7 (WM811K convention)
y_filtered[remap_mask_7_to_5] = 5  # Random -> class 5 (WM811K convention)

# Resize from 52x52 to 64x64 AND normalize to 0-255
print(f"\nResizing images from 52x52 to 64x64 and normalizing to 0-255...")
X_filtered = []
for img in X_filtered_52:
    # Resize
    img_resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_NEAREST)
    
    # Normalize to 0-255 range for proper image processing
    # MixedWM38 uses similar encoding to WM811K (0=background, 1=normal, 2/3=defect)
    img_normalized = img_resized.astype(np.float32)
    
    if img_normalized.max() <= 3:
        # Scale discrete values (0,1,2,3) to full grayscale range
        # This ensures compatibility with ImageNet pretrained models
        img_normalized = (img_normalized / img_normalized.max()) * 255.0
    elif img_normalized.max() < 255:
        # Already in some intermediate range, scale to full 0-255
        img_normalized = (img_normalized / img_normalized.max()) * 255.0
    
    X_filtered.append(img_normalized.astype(np.uint8))

X_filtered = np.array(X_filtered)

# Print filtered statistics
print("\n" + "="*60)
print("Filtered Dataset Summary:")
print("="*60)
print(f"X shape: {X_filtered.shape}, dtype: {X_filtered.dtype}")
print(f"  Value range: [{X_filtered.min()}, {X_filtered.max()}]")
print(f"  Expected: [0, 255] for proper image processing")
print(f"y shape: {y_filtered.shape}, dtype: {y_filtered.dtype}")
print(f"  Value range: [{y_filtered.min()}, {y_filtered.max()}]")

print("\nClass distribution (single-type + none):")
ID_TO_STR = {v: k for k, v in STR_TO_ID.items()}
unique, counts = np.unique(y_filtered, return_counts=True)
total = len(y_filtered)
for cls_id, count in zip(unique, counts):
    cls_name = ID_TO_STR.get(cls_id, 'Unknown')
    pct = 100 * count / total
    print(f"  Class {cls_id} ({cls_name:12s}): {count:6d} samples ({pct:5.2f}%)")

# Verify pixel value distribution
print("\nPixel value distribution (first 1000 images):")
unique_vals, val_counts = np.unique(X_filtered[:1000].flatten(), return_counts=True)
for val, count in zip(unique_vals[:10], val_counts[:10]):
    print(f"  Value {val:3.0f}: {count:8d} pixels")

# Save filtered data with keys matching training code expectations
np.savez_compressed(OUTPUT_NPZ, data=X_filtered, label=y_filtered)
print(f"\n✓ Saved filtered dataset to: {OUTPUT_NPZ}")
print("  Keys: 'data', 'label'")
print("  Image size: 64x64 (resized from 52x52)")
print("  Data type: uint8 with values 0-255")
print(f"  Labels: 0-8 (9 classes, including 'none' class)")
print("  Classes aligned with WM811K convention")
print("="*60)