import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

class PoseAnalyzer:
    """Pose and gesture analysis"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Define key poses for acting (simplified)
        self.key_poses = {
            'neutral': [0.5, 0.5, 0.5],  # Placeholder
            'open': [0.7, 0.5, 0.3],
            'closed': [0.3, 0.5, 0.7],
            'leaning_forward': [0.5, 0.3, 0.5],
            'leaning_back': [0.5, 0.7, 0.5]
        }
        
    def extract_landmarks(self, frame):
        """Extract pose landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        landmarks_dict = {}
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            for idx, landmark in enumerate(landmarks):
                landmarks_dict[f'pose_{idx}_x'] = landmark.x
                landmarks_dict[f'pose_{idx}_y'] = landmark.y
                landmarks_dict[f'pose_{idx}_z'] = landmark.z
                landmarks_dict[f'pose_{idx}_visibility'] = landmark.visibility
        
        return landmarks_dict, results
    
    def calculate_pose_features(self, landmarks_dict):
        """Calculate meaningful features from pose landmarks"""
        features = {}
        
        if landmarks_dict:
            # Extract key points (simplified)
            # Nose: 0, Shoulders: 11,12, Hips: 23,24, etc.
            
            # Posture (vertical alignment)
            nose_y = landmarks_dict.get('pose_0_y', 0)
            left_shoulder_y = landmarks_dict.get('pose_11_y', 0)
            right_shoulder_y = landmarks_dict.get('pose_12_y', 0)
            avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2 if left_shoulder_y and right_shoulder_y else 0
            
            features['posture_straightness'] = 1 - min(abs(nose_y - avg_shoulder_y), 1)
            
            # Arm openness
            left_shoulder_x = landmarks_dict.get('pose_11_x', 0)
            right_shoulder_x = landmarks_dict.get('pose_12_x', 0)
            left_elbow_x = landmarks_dict.get('pose_13_x', 0)
            right_elbow_x = landmarks_dict.get('pose_14_x', 0)
            left_wrist_x = landmarks_dict.get('pose_15_x', 0)
            right_wrist_x = landmarks_dict.get('pose_16_x', 0)
            
            shoulder_width = abs(right_shoulder_x - left_shoulder_x) if left_shoulder_x and right_shoulder_x else 0
            arm_span = abs(right_wrist_x - left_wrist_x) if left_wrist_x and right_wrist_x else 0
            
            if shoulder_width > 0:
                features['arm_openness'] = min(arm_span / (shoulder_width * 3), 1.0)
            else:
                features['arm_openness'] = 0.5
            
            # Movement speed (will be calculated across frames)
            features['body_tension'] = self.calculate_tension(landmarks_dict)
            
            # Gesture detection (simplified)
            features['gesture_intensity'] = self.calculate_gesture_intensity(landmarks_dict)
            
        return features
    
    def calculate_tension(self, landmarks_dict):
        """Calculate body tension based on joint angles"""
        # Simplified tension calculation
        tension = 0.5  # Default medium tension
        
        # Check if elbows are bent (high tension if straight)
        left_shoulder = np.array([landmarks_dict.get('pose_11_x', 0),
                                  landmarks_dict.get('pose_11_y', 0)])
        left_elbow = np.array([landmarks_dict.get('pose_13_x', 0),
                               landmarks_dict.get('pose_13_y', 0)])
        left_wrist = np.array([landmarks_dict.get('pose_15_x', 0),
                               landmarks_dict.get('pose_15_y', 0)])
        
        if np.any(left_shoulder) and np.any(left_elbow) and np.any(left_wrist):
            # Calculate angle at elbow
            v1 = left_elbow - left_shoulder
            v2 = left_wrist - left_elbow
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                tension = 1 - (angle / np.pi)  # Straight = high tension
        
        return tension
    
    def calculate_gesture_intensity(self, landmarks_dict):
        """Calculate gesture intensity based on movement range"""
        # Simplified - based on hand positions relative to body
        intensity = 0
        
        left_wrist_x = landmarks_dict.get('pose_15_x', 0)
        left_wrist_y = landmarks_dict.get('pose_15_y', 0)
        right_wrist_x = landmarks_dict.get('pose_16_x', 0)
        right_wrist_y = landmarks_dict.get('pose_16_y', 0)
        
        # Distance from center
        center_x, center_y = 0.5, 0.5
        left_dist = np.sqrt((left_wrist_x - center_x)**2 + (left_wrist_y - center_y)**2)
        right_dist = np.sqrt((right_wrist_x - center_x)**2 + (right_wrist_y - center_y)**2)
        
        intensity = (left_dist + right_dist) / 2
        
        return min(intensity * 2, 1.0)  # Normalize
    
    def analyze_video(self, video_path, sample_rate=5):
        """Analyze pose throughout video"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        all_data = []
        frame_count = 0
        previous_landmarks = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                landmarks_dict, results = self.extract_landmarks(frame)
                
                if landmarks_dict:
                    features = self.calculate_pose_features(landmarks_dict)
                    
                    # Calculate movement speed if previous frame exists
                    if previous_landmarks and previous_landmarks:
                        speed = self.calculate_movement_speed(landmarks_dict, previous_landmarks)
                        features['movement_speed'] = speed
                    else:
                        features['movement_speed'] = 0
                    
                    features['frame'] = frame_count
                    features['timestamp'] = frame_count / fps
                    
                    all_data.append(features)
                    previous_landmarks = landmarks_dict
            
            frame_count += 1
        
        cap.release()
        
        # Calculate statistics
        df = pd.DataFrame(all_data)
        
        if len(df) > 0:
            statistics = {
                'average_posture': df['posture_straightness'].mean(),
                'posture_variance': df['posture_straightness'].var(),
                'average_arm_openness': df['arm_openness'].mean(),
                'average_gesture_intensity': df['gesture_intensity'].mean(),
                'gestile_range': df['gesture_intensity'].max() - df['gesture_intensity'].min(),
                'average_movement_speed': df['movement_speed'].mean() if 'movement_speed' in df else 0,
                'movement_variance': df['movement_speed'].var() if 'movement_speed' in df else 0,
                'average_tension': df['body_tension'].mean()
            }
            
            # Calculate performance score based on pose dynamics
            # Good acting has varied, expressive poses
            pose_expression_score = (
                df['gesture_intensity'].std() * 0.4 +
                df['arm_openness'].mean() * 0.3 +
                (1 - abs(df['posture_straightness'].mean() - 0.7)) * 0.3  # Prefer moderately straight posture
            )
            statistics['pose_expression_score'] = min(pose_expression_score, 1.0)
            
        else:
            statistics = {
                'average_posture': 0,
                'posture_variance': 0,
                'average_arm_openness': 0,
                'average_gesture_intensity': 0,
                'gestile_range': 0,
                'average_movement_speed': 0,
                'movement_variance': 0,
                'average_tension': 0,
                'pose_expression_score': 0
            }
        
        return df, statistics
    
    def calculate_movement_speed(self, current, previous):
        """Calculate movement speed between frames"""
        speed = 0
        count = 0
        
        # Compare key landmarks
        key_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # Nose, shoulders, elbows, wrists, hips
        
        for idx in key_indices:
            curr_x = current.get(f'pose_{idx}_x', 0)
            curr_y = current.get(f'pose_{idx}_y', 0)
            prev_x = previous.get(f'pose_{idx}_x', 0)
            prev_y = previous.get(f'pose_{idx}_y', 0)
            
            if curr_x and prev_x:
                dist = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                speed += dist
                count += 1
        
        if count > 0:
            return speed / count
        return 0
    
    def classify_pose(self, landmarks_dict):
        """Classify the current pose into a category"""
        if not landmarks_dict:
            return 'unknown'
        
        features = self.calculate_pose_features(landmarks_dict)
        
        # Simple rule-based classification
        if features['arm_openness'] > 0.7:
            return 'open'
        elif features['arm_openness'] < 0.3:
            return 'closed'
        elif features['posture_straightness'] < 0.3:
            return 'leaning'
        else:
            return 'neutral'
    
    def draw_pose_on_frame(self, frame, pose_results):
        """Draw pose landmarks on frame"""
        if pose_results and pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Add pose classification text
            if hasattr(pose_results, 'pose_classification'):
                cv2.putText(frame, f"Pose: {pose_results.pose_classification}", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        return frame