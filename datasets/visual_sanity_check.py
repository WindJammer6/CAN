import numpy as np
from PIL import Image
import os
from collections import defaultdict
import random

def load_and_visualize_datasets():
    """Load both datasets and save sample images for each class"""
    
    # Set random seed for reproducible sampling
    random.seed(42)
    np.random.seed(42)
    
    # Load datasets
    print("Loading datasets...")
    wm811k = np.load('../wm811k-64.npz')
    mixed = np.load('../mixed2m38-64-single.npz')
    
    wm_data = wm811k['data']
    wm_labels = wm811k['label']
    mixed_data = mixed['data']
    mixed_labels = mixed['label']
    
    # Class names (should be same for both datasets now)
    class_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'none']
    
    # Create output directory
    output_dir = 'visual_samples'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Organize samples by class and save images
    def save_samples_by_class(data, labels, dataset_name, n_samples=3):
        samples_by_class = defaultdict(list)
        
        # Collect samples for each class
        for i, label in enumerate(labels):
            if len(samples_by_class[label]) < n_samples:
                samples_by_class[label].append((i, data[i]))
        
        # Save samples
        for class_id, samples in samples_by_class.items():
            class_name = class_names[class_id]
            print(f"  Class {class_id} ({class_name}): {len(samples)} samples saved")
            
            for i, (idx, img) in enumerate(samples):
                # Normalize image to 0-255 range for PIL
                img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                
                # Convert to PIL and save
                pil_img = Image.fromarray(img_normalized, mode='L')  # 'L' for grayscale
                filename = f"{output_dir}/{dataset_name}_class{class_id:02d}_{class_name}_sample{i+1}.png"
                pil_img.save(filename)
        
        return samples_by_class
    
    print("\nSaving WM811K samples...")
    wm_samples = save_samples_by_class(wm_data, wm_labels, "WM811K", 3)
    
    print("\nSaving MixedWM38 samples...")
    mixed_samples = save_samples_by_class(mixed_data, mixed_labels, "MixedWM38", 3)
    
    # Print sample counts for verification
    print("\n" + "="*60)
    print("SAMPLE COUNTS VERIFICATION")
    print("="*60)
    
    print("\nWM811K Dataset:")
    wm_unique, wm_counts = np.unique(wm_labels, return_counts=True)
    for cls_id, count in zip(wm_unique, wm_counts):
        print(f"  Class {cls_id} ({class_names[cls_id]:12s}): {count:7d} samples")
    
    print("\nMixedWM38 Dataset:")
    mixed_unique, mixed_counts = np.unique(mixed_labels, return_counts=True)
    for cls_id, count in zip(mixed_unique, mixed_counts):
        print(f"  Class {cls_id} ({class_names[cls_id]:12s}): {count:7d} samples")
    
    # Check for suspicious identical counts
    print("\n" + "="*60)
    print("SANITY CHECK RESULTS")
    print("="*60)
    
    suspicious = []
    for cls_id in range(9):
        wm_count = wm_counts[wm_unique == cls_id][0] if cls_id in wm_unique else 0
        mixed_count = mixed_counts[mixed_unique == cls_id][0] if cls_id in mixed_unique else 0
        
        if wm_count == mixed_count and wm_count > 0:
            suspicious.append((cls_id, class_names[cls_id], wm_count))
    
    if suspicious:
        print("⚠️  SUSPICIOUS: Identical sample counts detected!")
        for cls_id, cls_name, count in suspicious:
            print(f"   Class {cls_id} ({cls_name}): Both datasets have exactly {count} samples")
        print("\n   This suggests the MixedWM38 dataset may not have been regenerated correctly.")
    else:
        print("✅ GOOD: No identical sample counts detected between datasets")
    
    print(f"\n📁 Sample images saved in: {output_dir}/")
    print("   Format: [DATASET]_class[ID]_[NAME]_sample[N].png")
    print("   Compare visually to verify class mappings are correct!")

if __name__ == '__main__':
    load_and_visualize_datasets()

if __name__ == '__main__':
    load_and_visualize_datasets()