import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

class EyeTracker:
    """Eye tracking and gaze analysis"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices
        self.LEFT_EYE_INDICES = [33, 133, 157, 158, 159, 160, 161, 173]
        self.RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 466]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio (EAR)"""
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
        return min(ear, 1.0)
    
    def calculate_gaze_ratio(self, eye_landmarks, iris_center, frame_width):
        """Calculate gaze direction ratio"""
        eye_center = np.mean(eye_landmarks, axis=0)
        
        # Calculate horizontal gaze ratio
        if eye_center[0] > 0:
            gaze_ratio = (iris_center[0] - eye_center[0]) / eye_center[0]
        else:
            gaze_ratio = 0
        
        return gaze_ratio
    
    def analyze_frame(self, frame):
        """Analyze eye movements in a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        eye_data = {
            'left_ear': 0,
            'right_ear': 0,
            'avg_ear': 0,
            'left_gaze_ratio': 0,
            'right_gaze_ratio': 0,
            'avg_gaze_ratio': 0,
            'blink_detected': False,
            'eye_contact_score': 0
        }
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Get eye landmarks
            left_eye = np.array([[landmarks.landmark[i].x * w, 
                                  landmarks.landmark[i].y * h] 
                                 for i in self.LEFT_EYE_INDICES])
            right_eye = np.array([[landmarks.landmark[i].x * w, 
                                   landmarks.landmark[i].y * h] 
                                  for i in self.RIGHT_EYE_INDICES])
            
            # Get iris landmarks (if available)
            if hasattr(landmarks, 'landmark') and len(landmarks.landmark) > 477:
                left_iris = np.mean([[landmarks.landmark[i].x * w, 
                                      landmarks.landmark[i].y * h] 
                                     for i in self.LEFT_IRIS], axis=0)
                right_iris = np.mean([[landmarks.landmark[i].x * w, 
                                       landmarks.landmark[i].y * h] 
                                      for i in self.RIGHT_IRIS], axis=0)
            else:
                # Approximate iris position as eye center
                left_iris = np.mean(left_eye, axis=0)
                right_iris = np.mean(right_eye, axis=0)
            
            # Calculate EAR
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2
            
            # Calculate gaze ratios
            left_gaze = self.calculate_gaze_ratio(left_eye, left_iris, w)
            right_gaze = self.calculate_gaze_ratio(right_eye, right_iris, w)
            avg_gaze = (left_gaze + right_gaze) / 2
            
            # Detect blink (EAR threshold)
            blink_threshold = 0.2
            blink_detected = avg_ear < blink_threshold
            
            # Calculate eye contact score (based on gaze centeredness)
            eye_contact_score = 1 - min(abs(avg_gaze), 1)
            
            eye_data = {
                'left_ear': left_ear,
                'right_ear': right_ear,
                'avg_ear': avg_ear,
                'left_gaze_ratio': left_gaze,
                'right_gaze_ratio': right_gaze,
                'avg_gaze_ratio': avg_gaze,
                'blink_detected': blink_detected,
                'eye_contact_score': eye_contact_score
            }
        
        return eye_data
    
    def analyze_video(self, video_path, sample_rate=5):
        """Analyze entire video for eye tracking"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        all_data = []
        frame_count = 0
        blinks = 0
        eye_contact_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                eye_data = self.analyze_frame(frame)
                
                # Count blinks
                if eye_data['blink_detected']:
                    blinks += 1
                
                # Count eye contact frames
                if eye_data['eye_contact_score'] > 0.7:
                    eye_contact_frames += 1
                
                eye_data['frame'] = frame_count
                eye_data['timestamp'] = frame_count / fps
                all_data.append(eye_data)
            
            frame_count += 1
        
        cap.release()
        
        # Calculate statistics
        df = pd.DataFrame(all_data)
        
        if len(df) > 0:
            statistics = {
                'avg_eye_contact_score': df['eye_contact_score'].mean(),
                'avg_ear': df['avg_ear'].mean(),
                'blink_rate_per_minute': (blinks * 60 * sample_rate) / (frame_count / fps) if frame_count > 0 else 0,
                'total_blinks': blinks,
                'eye_contact_percentage': (eye_contact_frames / len(df)) * 100 if len(df) > 0 else 0,
                'gaze_variance': df['avg_gaze_ratio'].var(),
                'left_right_balance': abs(df['left_gaze_ratio'].mean() - df['right_gaze_ratio'].mean())
            }
        else:
            statistics = {
                'avg_eye_contact_score': 0,
                'avg_ear': 0,
                'blink_rate_per_minute': 0,
                'total_blinks': 0,
                'eye_contact_percentage': 0,
                'gaze_variance': 0,
                'left_right_balance': 0
            }
        
        return df, statistics
    
    def draw_eye_tracking(self, frame, eye_data):
        """Draw eye tracking visualization on frame"""
        if eye_data['avg_ear'] > 0:
            # Draw eye openness indicator
            openness_text = f"Eyes: {eye_data['avg_ear']:.2f}"
            cv2.putText(frame, openness_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw gaze direction
            gaze_x = int(frame.shape[1] * (0.5 + eye_data['avg_gaze_ratio']))
            gaze_y = int(frame.shape[0] * 0.5)
            
            # Draw gaze point
            cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 255, 255), -1)
            cv2.circle(frame, (gaze_x, gaze_y), 15, (255, 255, 255), 2)
            
            # Draw eye contact score
            score_text = f"Eye Contact: {eye_data['eye_contact_score']:.2f}"
            cv2.putText(frame, score_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        return frame