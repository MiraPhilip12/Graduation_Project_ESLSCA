import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from collections import deque

class GestureRecognizer:
    """Gesture recognition for specific actions"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_pose = mp.solutions.pose
        
        # Define gestures for each action
        self.action_gestures = {
            'Paddle_forehand': ['arm_swing_right', 'wrist_flex'],
            'Forehand_lob': ['arm_swing_up', 'wrist_extension'],
            'Backhand': ['arm_swing_left', 'wrist_flex'],
            'Backhand_lob': ['arm_swing_left_up', 'wrist_extension'],
            'Smash': ['arm_raise', 'wrist_snap'],
            'Phone_call': ['hand_to_ear', 'fingers_curl'],
            'Checking_watch': ['wrist_rotate', 'arm_lower'],
            'Clapping': ['hands_together', 'palms_face'],
            'Hand_shake': ['hand_extend', 'grasp'],
            'Thumbs_up': ['thumb_up', 'fingers_curl']
        }
        
        # Store recent frames for gesture detection
        self.recent_landmarks = deque(maxlen=10)
        
    def extract_hand_landmarks(self, frame):
        """Extract hand landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        hand_data = {
            'left_hand_present': False,
            'right_hand_present': False,
            'left_hand_landmarks': [],
            'right_hand_landmarks': [],
            'gestures': []
        }
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine if left or right hand
                if idx == 0:
                    hand_data['right_hand_present'] = True
                    hand_data['right_hand_landmarks'] = hand_landmarks
                elif idx == 1:
                    hand_data['left_hand_present'] = True
                    hand_data['left_hand_landmarks'] = hand_landmarks
                
                # Detect gestures for this hand
                gestures = self.detect_hand_gestures(hand_landmarks)
                hand_data['gestures'].extend(gestures)
        
        return hand_data, results
    
    def detect_hand_gestures(self, hand_landmarks):
        """Detect specific gestures from hand landmarks"""
        gestures = []
        
        # Get landmark positions
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        
        landmarks = np.array(landmarks)
        
        # Check for thumb up
        if self.is_thumb_up(landmarks):
            gestures.append('thumb_up')
        
        # Check for fist
        if self.is_fist(landmarks):
            gestures.append('fist')
        
        # Check for open hand
        if self.is_open_hand(landmarks):
            gestures.append('open_hand')
        
        # Check for pointing
        if self.is_pointing(landmarks):
            gestures.append('pointing')
        
        # Check for peace sign
        if self.is_peace_sign(landmarks):
            gestures.append('peace_sign')
        
        # Check for hand to ear (phone)
        if self.is_hand_to_ear(landmarks):
            gestures.append('hand_to_ear')
        
        return gestures
    
    def is_thumb_up(self, landmarks):
        """Detect thumb up gesture"""
        # Thumb tip y should be less than thumb IP y (higher up)
        thumb_tip_y = landmarks[4][1]
        thumb_ip_y = landmarks[3][1]
        
        # Other fingers should be curled (tip y > pip y)
        other_fingers_curled = True
        for finger_start in [5, 9, 13, 17]:  # Index, middle, ring, pinky MCP
            tip_y = landmarks[finger_start + 4][1]
            pip_y = landmarks[finger_start + 1][1]
            if tip_y < pip_y:  # Finger extended
                other_fingers_curled = False
                break
        
        return thumb_tip_y < thumb_ip_y and other_fingers_curled
    
    def is_fist(self, landmarks):
        """Detect fist gesture"""
        # All fingers curled
        all_curled = True
        for finger_start in [1, 5, 9, 13, 17]:  # Thumb to pinky MCP
            tip_y = landmarks[finger_start + 3][1]
            pip_y = landmarks[finger_start + 1][1] if finger_start > 1 else landmarks[finger_start + 2][1]
            if tip_y < pip_y:  # Finger extended
                all_curled = False
                break
        
        return all_curled
    
    def is_open_hand(self, landmarks):
        """Detect open hand gesture"""
        # All fingers extended
        all_extended = True
        for finger_start in [1, 5, 9, 13, 17]:  # Thumb to pinky MCP
            tip_y = landmarks[finger_start + 3][1]
            pip_y = landmarks[finger_start + 1][1] if finger_start > 1 else landmarks[finger_start + 2][1]
            if tip_y > pip_y:  # Finger curled
                all_extended = False
                break
        
        return all_extended
    
    def is_pointing(self, landmarks):
        """Detect pointing gesture"""
        # Index finger extended, others curled
        index_tip = landmarks[8][1]
        index_pip = landmarks[6][1]
        index_extended = index_tip < index_pip
        
        others_curled = True
        for finger_start in [1, 9, 13, 17]:  # Thumb, middle, ring, pinky
            tip_y = landmarks[finger_start + 3][1]
            pip_y = landmarks[finger_start + 1][1] if finger_start > 1 else landmarks[finger_start + 2][1]
            if tip_y < pip_y:  # Finger extended
                others_curled = False
                break
        
        return index_extended and others_curled
    
    def is_peace_sign(self, landmarks):
        """Detect peace sign (index and middle extended)"""
        # Index and middle extended, others curled
        index_tip = landmarks[8][1]
        index_pip = landmarks[6][1]
        middle_tip = landmarks[12][1]
        middle_pip = landmarks[10][1]
        
        index_extended = index_tip < index_pip
        middle_extended = middle_tip < middle_pip
        
        others_curled = True
        for finger_start in [1, 13, 17]:  # Thumb, ring, pinky
            tip_y = landmarks[finger_start + 3][1]
            pip_y = landmarks[finger_start + 1][1] if finger_start > 1 else landmarks[finger_start + 2][1]
            if tip_y < pip_y:  # Finger extended
                others_curled = False
                break
        
        return index_extended and middle_extended and others_curled
    
    def is_hand_to_ear(self, landmarks):
        """Detect hand to ear gesture (simplified)"""
        # Check if hand is near head region
        wrist = landmarks[0]
        
        # In a real implementation, you'd combine with pose detection
        # to know where the ear is. This is simplified.
        hand_y = wrist[1]
        hand_x = wrist[0]
        
        # Assume ear is at approximately (0.5, 0.3-0.4)
        ear_x, ear_y = 0.5, 0.35
        
        distance = np.sqrt((hand_x - ear_x)**2 + (hand_y - ear_y)**2)
        
        return distance < 0.2
    
    def analyze_video(self, video_path, target_action=None):
        """Analyze gestures throughout video"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        all_data = []
        frame_count = 0
        gesture_counts = {}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 5 == 0:  # Sample every 5 frames
                hand_data, results = self.extract_hand_landmarks(frame)
                
                # Count gestures
                for gesture in hand_data['gestures']:
                    gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
                
                data_point = {
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'left_hand': hand_data['left_hand_present'],
                    'right_hand': hand_data['right_hand_present'],
                    'gestures': hand_data['gestures']
                }
                all_data.append(data_point)
            
            frame_count += 1
        
        cap.release()
        
        # Calculate statistics
        df = pd.DataFrame(all_data)
        
        if len(df) > 0:
            statistics = {
                'total_frames_with_hands': (df['left_hand'] | df['right_hand']).sum(),
                'left_hand_percentage': df['left_hand'].mean() * 100,
                'right_hand_percentage': df['right_hand'].mean() * 100,
                'unique_gestures': len(gesture_counts),
                'gesture_frequency': gesture_counts,
                'most_common_gesture': max(gesture_counts, key=gesture_counts.get) if gesture_counts else 'none'
            }
            
            # Calculate gesture accuracy for target action
            if target_action and target_action in self.action_gestures:
                expected_gestures = self.action_gestures[target_action]
                detected_gestures = set(gesture_counts.keys())
                
                if expected_gestures:
                    matches = len([g for g in expected_gestures if g in detected_gestures])
                    statistics['gesture_accuracy'] = matches / len(expected_gestures)
                else:
                    statistics['gesture_accuracy'] = 0
            else:
                statistics['gesture_accuracy'] = 0
            
            # Calculate overall gesture performance score
            gesture_score = (
                min(len(gesture_counts) / 5, 1.0) * 0.4 +  # Variety of gestures
                statistics['gesture_accuracy'] * 0.6        # Accuracy for target action
            )
            statistics['gesture_performance_score'] = min(gesture_score, 1.0)
            
        else:
            statistics = {
                'total_frames_with_hands': 0,
                'left_hand_percentage': 0,
                'right_hand_percentage': 0,
                'unique_gestures': 0,
                'gesture_frequency': {},
                'most_common_gesture': 'none',
                'gesture_accuracy': 0,
                'gesture_performance_score': 0
            }
        
        return df, statistics
    
    def draw_gestures_on_frame(self, frame, hand_results):
        """Draw hand landmarks and gestures on frame"""
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Detect and display gestures
                gestures = self.detect_hand_gestures(hand_landmarks)
                if gestures:
                    # Get wrist position for text
                    wrist = hand_landmarks.landmark[0]
                    h, w, _ = frame.shape
                    cx, cy = int(wrist.x * w), int(wrist.y * h)
                    
                    cv2.putText(frame, f"Gestures: {', '.join(gestures)}", 
                               (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 0), 2)
        
        return frame