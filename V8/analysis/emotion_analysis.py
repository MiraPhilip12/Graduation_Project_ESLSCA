import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import mediapipe as mp

class EmotionAnalyzer:
    """Emotion analysis from facial expressions"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
    def analyze_frame(self, frame):
        """Analyze emotions in a single frame"""
        # DeepFace emotion analysis
        emotion_data = {
            'angry': 0,
            'disgust': 0,
            'fear': 0,
            'happy': 0,
            'sad': 0,
            'surprise': 0,
            'neutral': 0,
            'dominant_emotion': 'neutral',
            'face_confidence': 0
        }
        
        try:
            # Analyze with DeepFace
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result.get('emotion', {})
            for emotion, score in emotions.items():
                if emotion in emotion_data:
                    emotion_data[emotion] = score / 100  # Normalize to 0-1
            
            emotion_data['dominant_emotion'] = result.get('dominant_emotion', 'neutral')
            emotion_data['face_confidence'] = result.get('face_confidence', 0)
            
        except Exception as e:
            print(f"Emotion analysis error: {e}")
        
        return emotion_data
    
    def analyze_video(self, video_path, sample_rate=10):
        """Analyze emotions throughout video"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        all_data = []
        frame_count = 0
        emotion_counts = {emotion: 0 for emotion in self.emotion_labels}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                emotion_data = self.analyze_frame(frame)
                
                # Count dominant emotions
                dominant = emotion_data['dominant_emotion']
                if dominant in emotion_counts:
                    emotion_counts[dominant] += 1
                
                emotion_data['frame'] = frame_count
                emotion_data['timestamp'] = frame_count / fps
                all_data.append(emotion_data)
            
            frame_count += 1
        
        cap.release()
        
        # Calculate statistics
        df = pd.DataFrame(all_data)
        
        if len(df) > 0:
            statistics = {
                'dominant_emotion_overall': max(emotion_counts, key=emotion_counts.get),
                'emotion_variance': {emotion: df[emotion].var() for emotion in self.emotion_labels},
                'average_emotion_scores': {emotion: df[emotion].mean() for emotion in self.emotion_labels},
                'emotional_range': max(df[self.emotion_labels].max()) - min(df[self.emotion_labels].min()),
                'emotion_transitions': self.count_emotion_transitions(df['dominant_emotion'].tolist()),
                'face_presence': (df['face_confidence'] > 0.5).mean() * 100
            }
            
            # Calculate performance score based on emotional expression
            # For acting, more emotional variation is better
            emotional_expression_score = (
                df[self.emotion_labels].std().mean() * 0.5 +
                df['face_confidence'].mean() * 0.3 +
                (1 - df['neutral'].mean()) * 0.2
            )
            statistics['emotional_expression_score'] = min(emotional_expression_score, 1.0)
            
        else:
            statistics = {
                'dominant_emotion_overall': 'neutral',
                'emotion_variance': {},
                'average_emotion_scores': {},
                'emotional_range': 0,
                'emotion_transitions': 0,
                'face_presence': 0,
                'emotional_expression_score': 0
            }
        
        return df, statistics
    
    def count_emotion_transitions(self, emotions):
        """Count number of emotion transitions"""
        transitions = 0
        for i in range(1, len(emotions)):
            if emotions[i] != emotions[i-1]:
                transitions += 1
        return transitions
    
    def get_emotion_timeline(self, df):
        """Create emotion timeline for visualization"""
        timeline = []
        for _, row in df.iterrows():
            timeline.append({
                'timestamp': row['timestamp'],
                'emotion': row['dominant_emotion'],
                'intensity': max([row[e] for e in self.emotion_labels])
            })
        return timeline
    
    def calculate_emotion_appropriateness(self, emotions_df, scene_type=None):
        """Calculate how appropriate emotions are for a given scene type"""
        # This is a placeholder - in real implementation, you'd compare
        # detected emotions with expected emotions for the scene
        if scene_type == 'happy':
            score = emotions_df['happy'].mean()
        elif scene_type == 'sad':
            score = emotions_df['sad'].mean()
        elif scene_type == 'angry':
            score = emotions_df['angry'].mean()
        elif scene_type == 'neutral':
            score = emotions_df['neutral'].mean()
        else:
            # For general acting, emotional range is good
            score = emotions_df[self.emotion_labels].std().mean()
        
        return min(score, 1.0)
    
    def draw_emotion_on_frame(self, frame, emotion_data):
        """Draw emotion information on frame"""
        dominant = emotion_data['dominant_emotion']
        scores = {e: emotion_data[e] for e in self.emotion_labels}
        
        # Draw dominant emotion
        cv2.putText(frame, f"Emotion: {dominant}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw emotion bars
        y_start = 120
        for i, (emotion, score) in enumerate(scores.items()):
            bar_width = int(score * 200)
            cv2.rectangle(frame, (10, y_start + i*20), 
                         (10 + bar_width, y_start + i*20 + 15), 
                         (255, 0, 0), -1)
            cv2.putText(frame, f"{emotion[:3]}: {score:.2f}", 
                       (220, y_start + i*20 + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame