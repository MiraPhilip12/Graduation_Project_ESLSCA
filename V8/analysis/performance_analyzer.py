import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

from .eye_tracking import EyeTracker
from .emotion_analysis import EmotionAnalyzer
from .pose_estimation import PoseAnalyzer
from .gesture_recognition import GestureRecognizer
from .voice_analysis import VoiceAnalyzer

class PerformanceAnalyzer:
    """Main analysis pipeline for acting performance"""
    
    def __init__(self):
        self.eye_tracker = EyeTracker()
        self.emotion_analyzer = EmotionAnalyzer()
        self.pose_analyzer = PoseAnalyzer()
        self.gesture_recognizer = GestureRecognizer()
        self.voice_analyzer = VoiceAnalyzer()
        
    def analyze_video(self, video_path, target_action=None):
        """Run full analysis pipeline on video"""
        print(f"Starting analysis of {video_path}...")
        
        results = {
            'video_path': video_path,
            'target_action': target_action,
            'analysis_time': datetime.now().isoformat(),
            'eye_tracking': {},
            'emotion_analysis': {},
            'pose_analysis': {},
            'gesture_analysis': {},
            'voice_analysis': {},
            'overall_scores': {}
        }
        
        # Eye tracking analysis
        print("Analyzing eye movements...")
        try:
            eye_df, eye_stats = self.eye_tracker.analyze_video(video_path)
            results['eye_tracking'] = {
                'statistics': eye_stats,
                'data': eye_df.to_dict('records') if not eye_df.empty else []
            }
        except Exception as e:
            print(f"Eye tracking error: {e}")
            results['eye_tracking']['error'] = str(e)
        
        # Emotion analysis
        print("Analyzing emotions...")
        try:
            emotion_df, emotion_stats = self.emotion_analyzer.analyze_video(video_path)
            results['emotion_analysis'] = {
                'statistics': emotion_stats,
                'data': emotion_df.to_dict('records') if not emotion_df.empty else []
            }
        except Exception as e:
            print(f"Emotion analysis error: {e}")
            results['emotion_analysis']['error'] = str(e)
        
        # Pose analysis
        print("Analyzing pose and posture...")
        try:
            pose_df, pose_stats = self.pose_analyzer.analyze_video(video_path)
            results['pose_analysis'] = {
                'statistics': pose_stats,
                'data': pose_df.to_dict('records') if not pose_df.empty else []
            }
        except Exception as e:
            print(f"Pose analysis error: {e}")
            results['pose_analysis']['error'] = str(e)
        
        # Gesture analysis
        print("Analyzing gestures...")
        try:
            gesture_df, gesture_stats = self.gesture_recognizer.analyze_video(video_path, target_action)
            results['gesture_analysis'] = {
                'statistics': gesture_stats,
                'data': gesture_df.to_dict('records') if not gesture_df.empty else []
            }
        except Exception as e:
            print(f"Gesture analysis error: {e}")
            results['gesture_analysis']['error'] = str(e)
        
        # Voice analysis
        print("Analyzing voice...")
        try:
            voice_df, voice_stats = self.voice_analyzer.analyze_video(video_path)
            results['voice_analysis'] = {
                'statistics': voice_stats,
                'data': voice_df.to_dict('records') if not voice_df.empty else []
            }
        except Exception as e:
            print(f"Voice analysis error: {e}")
            results['voice_analysis']['error'] = str(e)
        
        # Calculate overall scores
        results['overall_scores'] = self.calculate_overall_scores(results)
        
        # Determine performance level
        results['performance_level'] = self.get_performance_level(results['overall_scores'].get('final_score', 0))
        
        print("Analysis complete!")
        return results
    
    def calculate_overall_scores(self, results):
        """Calculate overall performance scores"""
        scores = {}
        
        # Eye contact score (from eye tracking)
        if 'eye_tracking' in results and 'statistics' in results['eye_tracking']:
            eye_stats = results['eye_tracking']['statistics']
            scores['eye_contact'] = eye_stats.get('avg_eye_contact_score', 0)
        else:
            scores['eye_contact'] = 0
        
        # Emotional expression score
        if 'emotion_analysis' in results and 'statistics' in results['emotion_analysis']:
            emotion_stats = results['emotion_analysis']['statistics']
            scores['emotional_expression'] = emotion_stats.get('emotional_expression_score', 0)
        else:
            scores['emotional_expression'] = 0
        
        # Pose expression score
        if 'pose_analysis' in results and 'statistics' in results['pose_analysis']:
            pose_stats = results['pose_analysis']['statistics']
            scores['physical_expression'] = pose_stats.get('pose_expression_score', 0)
        else:
            scores['physical_expression'] = 0
        
        # Gesture performance score
        if 'gesture_analysis' in results and 'statistics' in results['gesture_analysis']:
            gesture_stats = results['gesture_analysis']['statistics']
            scores['gesture_accuracy'] = gesture_stats.get('gesture_performance_score', 0)
        else:
            scores['gesture_accuracy'] = 0
        
        # Voice performance score
        if 'voice_analysis' in results and 'statistics' in results['voice_analysis']:
            voice_stats = results['voice_analysis']['statistics']
            scores['voice_quality'] = voice_stats.get('overall_voice_score', 0)
        else:
            scores['voice_quality'] = 0
        
        # Weighted final score (priorities from director)
        # 1. Eye movement (30%)
        # 2. Emotional analysis (30%)
        # 3. Voice (25%)
        # 4. Gestures and posture (15%)
        
        scores['final_score'] = (
            scores.get('eye_contact', 0) * 0.30 +
            scores.get('emotional_expression', 0) * 0.30 +
            scores.get('voice_quality', 0) * 0.25 +
            (scores.get('physical_expression', 0) + scores.get('gesture_accuracy', 0)) / 2 * 0.15
        )
        
        return scores
    
    def get_performance_level(self, score):
        """Convert numerical score to performance level"""
        if score >= 0.7:
            return "Good Performance"
        elif score >= 0.4:
            return "Moderate Performance"
        else:
            return "Bad Performance"
    
    def generate_summary(self, results):
        """Generate a summary of the analysis"""
        summary = {
            'actor_name': results.get('actor_name', 'Unknown'),
            'action': results.get('target_action', 'Unknown'),
            'performance_level': results['performance_level'],
            'final_score': results['overall_scores']['final_score'],
            'key_metrics': {}
        }
        
        # Extract key metrics
        if 'eye_tracking' in results and 'statistics' in results['eye_tracking']:
            eye_stats = results['eye_tracking']['statistics']
            summary['key_metrics']['eye_contact_percentage'] = eye_stats.get('eye_contact_percentage', 0)
        
        if 'emotion_analysis' in results and 'statistics' in results['emotion_analysis']:
            emotion_stats = results['emotion_analysis']['statistics']
            summary['key_metrics']['dominant_emotion'] = emotion_stats.get('dominant_emotion_overall', 'neutral')
            summary['key_metrics']['emotional_range'] = emotion_stats.get('emotional_range', 0)
        
        if 'gesture_analysis' in results and 'statistics' in results['gesture_analysis']:
            gesture_stats = results['gesture_analysis']['statistics']
            summary['key_metrics']['gesture_accuracy'] = gesture_stats.get('gesture_accuracy', 0)
        
        if 'voice_analysis' in results and 'statistics' in results['voice_analysis']:
            voice_stats = results['voice_analysis']['statistics']
            summary['key_metrics']['words_spoken'] = voice_stats.get('words_spoken', 0)
            summary['key_metrics']['vocal_energy'] = voice_stats.get('vocal_energy', 0)
        
        return summary
    
    def save_results(self, results, output_path):
        """Save analysis results to file"""
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            else:
                return obj
        
        # Recursively convert
        serializable_results = json.loads(
            json.dumps(results, default=convert_to_serializable)
        )
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {output_path}")
        
        return output_path