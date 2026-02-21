# test_data_loading.py
import sys
sys.path.append('.')

from utils.data_loader import DataLoader
import numpy as np

print("Testing data loading...")
data_loader = DataLoader(dataset_path="Dataset")

# Test getting video paths
print("\n1. Testing get_video_paths...")
videos, labels = data_loader.get_video_paths()
print(f"Found {len(videos)} total videos")
if len(videos) > 0:
    print(f"First video: {videos[0]}")
    print(f"First label: {labels[0]}")

# Test preparing dataset
print("\n2. Testing prepare_dataset...")
X, y, y_action, paths = data_loader.prepare_dataset(
    max_videos_per_class=2,  # Small number for testing
    feature_type='aggregated'
)

if X is not None:
    print(f"✅ Success! Loaded {len(X)} samples")
    print(f"Features shape: {X.shape}")
    print(f"Labels: {y}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Sample paths: {paths[:3]}")
else:
    print("❌ Failed to load data")
    
    # Debug: Check directory structure
    print("\n3. Debugging directory structure:")
    import os
    from pathlib import Path
    
    base_path = Path("Dataset")
    print(f"Absolute path: {base_path.absolute()}")
    
    if base_path.exists():
        print("✅ Dataset folder exists")
        
        # List action folders
        action_folders = [f for f in base_path.iterdir() if f.is_dir()]
        print(f"\nFound {len(action_folders)} action folders:")
        for folder in action_folders[:5]:  # Show first 5
            print(f"  📁 {folder.name}/")
            
            # Check performance level folders
            for level in ["Good Performance", "Moderate Performance", "Bad Performance"]:
                level_path = folder / level
                if level_path.exists():
                    videos = list(level_path.glob("*.mp4"))
                    print(f"    ✅ {level}: {len(videos)} videos")
                else:
                    print(f"    ❌ {level}: Not found")
    else:
        print(f"❌ Dataset folder not found at {base_path.absolute()}")