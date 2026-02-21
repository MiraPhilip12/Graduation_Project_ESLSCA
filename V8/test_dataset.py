# test_dataset.py
import os
from pathlib import Path

# Print current directory
print(f"Current directory: {os.getcwd()}")

# Check if Dataset folder exists
dataset_path = Path("Dataset")
if not dataset_path.exists():
    print(f"❌ Dataset folder not found at: {dataset_path.absolute()}")
    exit(1)

print(f"✅ Dataset folder found at: {dataset_path.absolute()}")

# List all items in Dataset
print("\nContents of Dataset folder:")
for item in dataset_path.iterdir():
    if item.is_dir():
        print(f"  📁 {item.name}/")
    else:
        print(f"  📄 {item.name}")

# Look for performance level folders
performance_levels = ["Good Performance", "Moderate Performance", "Bad Performance"]

print("\nChecking for performance level folders:")
for action_folder in dataset_path.iterdir():
    if action_folder.is_dir():
        print(f"\n📁 {action_folder.name}:")
        for level in performance_levels:
            level_path = action_folder / level
            if level_path.exists():
                # Count videos
                videos = list(level_path.glob("*.mp4"))
                print(f"  ✅ {level}: {len(videos)} videos")
            else:
                print(f"  ❌ {level}: Not found")