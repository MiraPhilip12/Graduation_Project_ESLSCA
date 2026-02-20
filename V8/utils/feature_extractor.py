import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import librosa
import speech_recognition as sr
from typing import Dict, List, Tuple, Optional
import pandas as pd

class FeatureExtractor:
    """Extract features from video/audio for acting performance analysis"""
    
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
    def extract_pose_features(self, frame):
        """Extract pose landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        features = {}
        if results.pose_landmarks:
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark
            for idx, landmark in enumerate(landmarks):
                features[f'pose_{idx}_x'] = landmark.x
                features[f'pose_{idx}_y'] = landmark.y
                features[f'pose_{idx}_z'] = landmark.z
                features[f'pose_{idx}_visibility'] = landmark.visibility
        else:
            # Return zeros if no pose detected
            for idx in range(33):  # 33 pose landmarks
                features[f'pose_{idx}_x'] = 0
                features[f'pose_{idx}_y'] = 0
                features[f'pose_{idx}_z'] = 0
                features[f'pose_{idx}_visibility'] = 0
        
        return features, results
    
    def extract_face_features(self, frame):
        """Extract face landmarks and emotions"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face landmarks
        face_results = self.face_mesh.process(rgb_frame)
        
        features = {}
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            for idx, landmark in enumerate(landmarks[:100]):  # Limit to 100 landmarks
                features[f'face_{idx}_x'] = landmark.x
                features[f'face_{idx}_y'] = landmark.y
                features[f'face_{idx}_z'] = landmark.z
        
        # Emotion analysis (using DeepFace)
        try:
            emotion_analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(emotion_analysis, list):
                emotion_analysis = emotion_analysis[0]
            
            emotions = emotion_analysis.get('emotion', {})
            for emotion, score in emotions.items():
                features[f'emotion_{emotion}'] = score / 100.0  # Normalize to 0-1
            
            # DON'T add dominant_emotion as a feature (it's a string)
            # features['dominant_emotion'] = emotion_analysis.get('dominant_emotion', '')
            
        except Exception as e:
            print(f"Emotion analysis error: {e}")
            # Default values
            default_emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            for emotion in default_emotions:
                features[f'emotion_{emotion}'] = 0
        
        return features, face_results
    
    def extract_hand_features(self, frame):
        """Extract hand landmarks for gestures"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        features = {}
        
        # Left hand
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            hand_landmarks = results.multi_hand_landmarks[0].landmark
            for idx, landmark in enumerate(hand_landmarks):
                features[f'hand_left_{idx}_x'] = landmark.x
                features[f'hand_left_{idx}_y'] = landmark.y
                features[f'hand_left_{idx}_z'] = landmark.z
        else:
            for idx in range(21):  # 21 hand landmarks
                features[f'hand_left_{idx}_x'] = 0
                features[f'hand_left_{idx}_y'] = 0
                features[f'hand_left_{idx}_z'] = 0
        
        # Right hand
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
            hand_landmarks = results.multi_hand_landmarks[1].landmark
            for idx, landmark in enumerate(hand_landmarks):
                features[f'hand_right_{idx}_x'] = landmark.x
                features[f'hand_right_{idx}_y'] = landmark.y
                features[f'hand_right_{idx}_z'] = landmark.z
        else:
            for idx in range(21):  # 21 hand landmarks
                features[f'hand_right_{idx}_x'] = 0
                features[f'hand_right_{idx}_y'] = 0
                features[f'hand_right_{idx}_z'] = 0
        
        return features, results
    
    def extract_eye_gaze_features(self, frame, face_landmarks=None):
        """Extract eye gaze direction features"""
        features = {}
        
        if face_landmarks and face_landmarks.multi_face_landmarks:
            landmarks = face_landmarks.multi_face_landmarks[0].landmark
            
            # Eye landmarks indices (MediaPipe Face Mesh)
            LEFT_EYE = [33, 133, 157, 158, 159, 160, 161, 173]
            RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 466]
            
            # Calculate eye aspect ratio and gaze
            left_eye_points = [landmarks[i] for i in LEFT_EYE]
            right_eye_points = [landmarks[i] for i in RIGHT_EYE]
            
            # Eye openness (EAR - Eye Aspect Ratio)
            left_ear = self.calculate_ear(left_eye_points)
            right_ear = self.calculate_ear(right_eye_points)
            
            features['left_eye_openness'] = left_ear
            features['right_eye_openness'] = right_ear
            features['avg_eye_openness'] = (left_ear + right_ear) / 2
            
            # Gaze direction (simplified)
            left_eye_center = np.mean([[p.x, p.y] for p in left_eye_points], axis=0)
            right_eye_center = np.mean([[p.x, p.y] for p in right_eye_points], axis=0)
            
            features['gaze_x'] = (left_eye_center[0] + right_eye_center[0]) / 2
            features['gaze_y'] = (left_eye_center[1] + right_eye_center[1]) / 2
        
        return features
    
    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio"""
        # Simplified EAR calculation
        if len(eye_points) < 6:
            return 0.5
        
        # Vertical distances
        v1 = np.linalg.norm([eye_points[1].x - eye_points[5].x, 
                             eye_points[1].y - eye_points[5].y])
        v2 = np.linalg.norm([eye_points[2].x - eye_points[4].x, 
                             eye_points[2].y - eye_points[4].y])
        
        # Horizontal distance
        h = np.linalg.norm([eye_points[0].x - eye_points[3].x, 
                           eye_points[0].y - eye_points[3].y])
        
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
        return min(ear, 1.0)  # Normalize
    
    def extract_audio_features(self, audio_path):
        """Extract features from audio file"""
        features = {}
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
                features[f'mfcc_{i}_std'] = np.std(mfcc[i])
            
            # Pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            features['pitch_mean'] = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            features['pitch_std'] = np.std(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            # Energy
            energy = np.sum(y**2) / len(y)
            features['energy'] = energy
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = np.mean(zcr)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            
            # Speech recognition
            try:
                # Save audio temporarily for speech recognition
                temp_audio = "temp_audio.wav"
                import soundfile as sf
                sf.write(temp_audio, y, sr)
                
                with sr.AudioFile(temp_audio) as source:
                    audio = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio)
                    features['transcribed_text'] = text
                    features['speech_detected'] = 1
                    
                import os
                os.remove(temp_audio)
            except:
                features['transcribed_text'] = ''
                features['speech_detected'] = 0
                
        except Exception as e:
            print(f"Audio feature extraction error: {e}")
            # Default values
            for i in range(13):
                features[f'mfcc_{i}_mean'] = 0
                features[f'mfcc_{i}_std'] = 0
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['energy'] = 0
            features['zcr_mean'] = 0
            features['spectral_centroid_mean'] = 0
            features['transcribed_text'] = ''
            features['speech_detected'] = 0
        
        return features
    
    def process_video(self, video_path, sample_rate=30):
        """Process entire video and extract features frame by frame"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_features = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_count % sample_rate == 0:
                frame_features = {}
                
                # Extract all features
                pose_features, pose_results = self.extract_pose_features(frame)
                face_features, face_results = self.extract_face_features(frame)
                hand_features, hand_results = self.extract_hand_features(frame)
                gaze_features = self.extract_eye_gaze_features(frame, face_results)
                
                # Combine all features
                frame_features.update(pose_features)
                frame_features.update(face_features)
                frame_features.update(hand_features)
                frame_features.update(gaze_features)
                frame_features['frame_number'] = frame_count
                frame_features['timestamp'] = frame_count / fps
                
                all_features.append(frame_features)
            
            frame_count += 1
        
        cap.release()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Calculate aggregated statistics
        aggregated_features = {}
        for col in df.columns:
            if col not in ['frame_number', 'timestamp']:
                aggregated_features[f'{col}_mean'] = df[col].mean()
                aggregated_features[f'{col}_std'] = df[col].std()
                aggregated_features[f'{col}_max'] = df[col].max()
                aggregated_features[f'{col}_min'] = df[col].min()
        
        return df, aggregated_features
    
    def draw_landmarks(self, frame, pose_results, face_results, hand_results):
        """Draw landmarks on frame for visualization"""
        # Draw pose landmarks
        if pose_results and pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        # Draw face mesh
        if face_results and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS)
        
        # Draw hand landmarks
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return frame