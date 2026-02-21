# test_fixed_features.py
import sys
sys.path.append('.')

from utils.data_loader import DataLoader
import numpy as np
import pandas as pd

print("Testing fixed feature extraction...")
data_loader = DataLoader(dataset_path="Dataset")

# Test with just one video
print("\n1. Testing single video feature extraction...")
test_video = "Dataset/BackhandLob_Dataset/Good Performance/BackhandLob1_Good.mp4"
df, features = data_loader.extract_features_from_video(test_video, max_frames=10)

if features:
    print(f"✅ Successfully extracted {len(features)} features")
    print(f"First 5 feature names: {list(features.keys())[:5]}")
    print(f"First 5 feature values: {list(features.values())[:5]}")
    
    # Check if all values are numeric
    all_numeric = all(isinstance(v, (int, float)) for v in features.values())
    print(f"✅ All features are numeric: {all_numeric}")
else:
    print("❌ Failed to extract features")

# Test preparing small dataset
print("\n2. Testing prepare_dataset with small sample...")
X, y, y_action, paths = data_loader.prepare_dataset(
    max_videos_per_class=1,  # Just 1 video per class for testing
    feature_type='aggregated'
)

if X is not None:
    print(f"✅ Success! Loaded {len(X)} samples")
    print(f"Features shape: {X.shape}")
    print(f"Labels: {y}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Check if X contains only numeric values
    print(f"X dtype: {X.dtype}")
    print(f"X contains NaN: {np.isnan(X).any()}")
    print(f"X contains Inf: {np.isinf(X).any()}")
else:
    print("❌ Failed to load dataset")