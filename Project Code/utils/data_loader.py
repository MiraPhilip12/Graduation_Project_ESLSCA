# utils/data_loader.py
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .feature_extractor import FeatureExtractor

class DataLoader:
    """Load and prepare data for training"""
    
    def __init__(self, dataset_path="Dataset"):
        self.dataset_path = Path(dataset_path)
        self.feature_extractor = FeatureExtractor()
        
        # Your exact action folder names
        self.action_folders = [
            "BackhandLob_Dataset", "Backhand_Dataset", "CheckTheWatch_Dataset",
            "Clapping_Dataset", "ForehandLob_Dataset", "Forehand_Dataset",
            "Handshake_Dataset", "PhoneCall_Dataset", "Smash_Dataset", "ThumbsUp_Dataset"
        ]
        
        # Map folder names to action names for display
        self.action_names = {
            "BackhandLob_Dataset": "Backhand Lob",
            "Backhand_Dataset": "Backhand",
            "CheckTheWatch_Dataset": "Check The Watch",
            "Clapping_Dataset": "Clapping",
            "ForehandLob_Dataset": "Forehand Lob",
            "Forehand_Dataset": "Forehand",
            "Handshake_Dataset": "Handshake",
            "PhoneCall_Dataset": "Phone Call",
            "Smash_Dataset": "Smash",
            "ThumbsUp_Dataset": "Thumbs Up"
        }
        
        # Your performance levels (exact folder names)
        self.performance_levels = {
            "Good Performance": 2,
            "Moderate Performance": 1,
            "Bad Performance": 0
        }
        
        # Reverse mapping for labels
        self.performance_labels = {0: "Bad Performance", 1: "Moderate Performance", 2: "Good Performance"}
        self.feature_cache = {}
        
    def get_video_paths(self, action_folder=None, level=None):
        """Get all video paths matching criteria"""
        video_paths = []
        labels = []
        
        # Determine which action folders to search
        if action_folder:
            if action_folder in self.action_folders:
                search_folders = [action_folder]
            else:
                search_folders = []
        else:
            search_folders = self.action_folders
        
        # Walk through directory structure
        for action_folder_name in search_folders:
            action_path = self.dataset_path / action_folder_name
            
            if not action_path.exists():
                continue
            
            # If level is specified, only look in that level folder
            if level and level in self.performance_levels:
                level_path = action_path / level
                if level_path.exists():
                    for video_file in level_path.glob("*.*"):
                        if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.wmv', '.mkv']:
                            video_paths.append(str(video_file))
                            labels.append({
                                'path': str(video_file),
                                'action_folder': action_folder_name,
                                'action_name': self.action_names.get(action_folder_name, action_folder_name),
                                'level': level,
                                'label': self.performance_levels[level],
                                'action_label': self.action_folders.index(action_folder_name) if action_folder_name in self.action_folders else -1
                            })
            else:
                # Look in all level folders
                for level_name, level_label in self.performance_levels.items():
                    level_path = action_path / level_name
                    if level_path.exists():
                        for video_file in level_path.glob("*.*"):
                            if video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.wmv', '.mkv']:
                                video_paths.append(str(video_file))
                                labels.append({
                                    'path': str(video_file),
                                    'action_folder': action_folder_name,
                                    'action_name': self.action_names.get(action_folder_name, action_folder_name),
                                    'level': level_name,
                                    'label': level_label,
                                    'action_label': self.action_folders.index(action_folder_name) if action_folder_name in self.action_folders else -1
                                })
        
        return video_paths, labels
    
    def extract_features_from_video(self, video_path, max_frames=15):
        """Extract features from a single video (optimized with caching)"""
        
        # Check cache first
        cache_key = f"{video_path}_{max_frames}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None, None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling - more aggressive sampling
        if total_frames > max_frames:
            sample_rate = total_frames // max_frames
        else:
            sample_rate = 1
        
        all_features = []
        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                try:
                    # Extract features from frame
                    pose_features, _ = self.feature_extractor.extract_pose_features(frame)
                    face_features, _ = self.feature_extractor.extract_face_features(frame)
                    hand_features, _ = self.feature_extractor.extract_hand_features(frame)
                    
                    # Combine features
                    frame_features = {}
                    
                    # Select only the most important features
                    # For pose, take every 5th landmark
                    pose_keys = sorted(pose_features.keys())
                    for key in pose_keys[::5]:  # Take every 5th
                        frame_features[key] = float(pose_features[key])
                    
                    # For face, take fewer landmarks
                    face_keys = sorted(face_features.keys())
                    emotion_keys = [k for k in face_keys if 'emotion_' in k]
                    landmark_keys = [k for k in face_keys if 'face_' in k]
                    
                    # Take all emotions (they're few)
                    for key in emotion_keys:
                        frame_features[key] = float(face_features[key])
                    
                    # Take every 10th face landmark
                    for key in landmark_keys[::10]:
                        frame_features[key] = float(face_features[key])
                    
                    # For hands, take every 5th landmark
                    hand_keys = sorted(hand_features.keys())
                    for key in hand_keys[::5]:
                        frame_features[key] = float(hand_features[key])
                    
                    if frame_features:
                        all_features.append(frame_features)
                        processed_frames += 1
                        
                except Exception as e:
                    print(f"Error extracting features from frame: {e}")
            
            frame_count += 1
        
        cap.release()
        
        if not all_features:
            return None, None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        df = df.fillna(0)
        
        # Calculate aggregated features
        aggregated = {}
        for col in df.columns:
            try:
                col_data = pd.to_numeric(df[col], errors='coerce').fillna(0)
                aggregated[f'{col}_mean'] = float(col_data.mean())
                aggregated[f'{col}_std'] = float(col_data.std())
                # Skip max and min to reduce features
                # aggregated[f'{col}_max'] = float(col_data.max())
                # aggregated[f'{col}_min'] = float(col_data.min())
            except Exception as e:
                continue
        
        # Cache the result
        self.feature_cache[cache_key] = (df, aggregated)
        
        return df, aggregated
    
    def prepare_dataset(self, actions=None, max_videos_per_class=5, feature_type='aggregated'):
        """Prepare dataset for training"""
        all_features = []
        all_labels = []
        all_action_labels = []
        all_paths = []
        
        # Determine which action folders to process
        if actions:
            search_folders = []
            for action in actions:
                if action in self.action_folders:
                    search_folders.append(action)
                else:
                    # Try to find matching folder
                    for folder in self.action_folders:
                        if action.lower() in folder.lower() or action.lower() in self.action_names.get(folder, '').lower():
                            search_folders.append(folder)
                            break
        else:
            search_folders = self.action_folders
        
        # Get all videos
        all_videos = []
        for folder in search_folders:
            videos, labels = self.get_video_paths(action_folder=folder)
            all_videos.extend(list(zip(videos, labels)))
        
        # Group by (action, level) for balanced sampling
        class_groups = {}
        for path, info in all_videos:
            class_key = (info['action_folder'], info['level'])
            if class_key not in class_groups:
                class_groups[class_key] = []
            class_groups[class_key].append((path, info))
        
        # Sample videos per class
        for class_key, videos in class_groups.items():
            if max_videos_per_class and len(videos) > max_videos_per_class:
                videos = random.sample(videos, max_videos_per_class)
            
            for path, info in videos:
                print(f"Processing {info['action_name']} - {info['level']}: {Path(path).name}")
                _, features = self.extract_features_from_video(path, max_frames=15)
                
                if features:
                    all_features.append(features)
                    all_labels.append(info['label'])
                    all_action_labels.append(info['action_label'])
                    all_paths.append(path)
        
        if not all_features:
            print("No features extracted!")
            return None, None, None, None
        
        # Ensure all feature dictionaries have the same keys
        print("Aligning features...")
        feature_keys = set()
        for feat in all_features:
            feature_keys.update(feat.keys())
        
        feature_keys = sorted(list(feature_keys))
        print(f"Total features: {len(feature_keys)}")
        
        # Create X array with consistent features
        X_list = []
        for feat in all_features:
            feat_vec = [feat.get(key, 0) for key in feature_keys]
            X_list.append(feat_vec)
        
        X = np.array(X_list)
        y = np.array(all_labels)
        y_action = np.array(all_action_labels)
        
        print(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y)}")
        
        return X, y, y_action, all_paths
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_val  # Fixed: last should be y_val