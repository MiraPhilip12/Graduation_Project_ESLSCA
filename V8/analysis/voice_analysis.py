import librosa
import numpy as np
import pandas as pd
import speech_recognition as sr
import soundfile as sf
import os
import tempfile

class VoiceAnalyzer:
    """Voice and speech analysis"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def extract_audio_from_video(self, video_path):
        """Extract audio from video file"""
        try:
            import subprocess
            import os
            
            # Create temp audio file
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()
            
            # Use ffmpeg to extract audio (you need ffmpeg installed)
            cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
                   '-ar', '16000', '-ac', '1', temp_audio_path, '-y']
            
            subprocess.run(cmd, capture_output=True, text=True)
            
            return temp_audio_path
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return None
    
    def analyze_audio_file(self, audio_path):
        """Analyze audio features"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=16000)
            
            features = {}
            
            # Basic features
            features['duration'] = len(y) / sr
            features['rms_energy'] = np.mean(librosa.feature.rms(y=y))
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get non-zero pitches
            pitch_values = pitches[pitches > 0]
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_min'] = np.min(pitch_values)
                features['pitch_max'] = np.max(pitch_values)
                features['pitch_range'] = features['pitch_max'] - features['pitch_min']
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_min'] = 0
                features['pitch_max'] = 0
                features['pitch_range'] = 0
            
            # MFCC features (timbre)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            
            # Tempo/rhythm
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # Harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            features['harmonic_ratio'] = np.sum(y_harmonic**2) / np.sum(y**2) if np.sum(y**2) > 0 else 0
            
            return features
            
        except Exception as e:
            print(f"Error analyzing audio: {e}")
            return {}
    
    def transcribe_speech(self, audio_path):
        """Transcribe speech from audio"""
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                
                # Try different recognition methods
                text = ""
                confidence = 0
                
                try:
                    # Google Speech Recognition
                    text = self.recognizer.recognize_google(audio)
                    confidence = 0.8  # Approximate confidence
                except sr.UnknownValueError:
                    text = "[Speech not recognized]"
                    confidence = 0
                except sr.RequestError:
                    text = "[Recognition service unavailable]"
                    confidence = 0
                
                return text, confidence
                
        except Exception as e:
            print(f"Error transcribing speech: {e}")
            return "", 0
    
    def analyze_voice_quality(self, audio_path):
        """Analyze voice quality for acting"""
        features = self.analyze_audio_file(audio_path)
        
        if not features:
            return {}
        
        # Voice quality metrics for acting
        quality_metrics = {
            'vocal_energy': min(features.get('rms_energy', 0) * 10, 1.0),
            'pitch_variety': min(features.get('pitch_std', 0) / 100, 1.0),
            'vocal_range': min(features.get('pitch_range', 0) / 300, 1.0),
            'clarity': min(1 - features.get('zero_crossing_rate', 0) * 10, 1.0),
            'rhythm_consistency': 1 - min(abs(features.get('tempo', 120) - 120) / 120, 1.0)
        }
        
        # Overall voice performance score
        voice_score = (
            quality_metrics['vocal_energy'] * 0.25 +
            quality_metrics['pitch_variety'] * 0.3 +
            quality_metrics['vocal_range'] * 0.2 +
            quality_metrics['clarity'] * 0.15 +
            quality_metrics['rhythm_consistency'] * 0.1
        )
        
        quality_metrics['overall_voice_score'] = min(voice_score, 1.0)
        
        return quality_metrics
    
    def analyze_video(self, video_path):
        """Analyze voice from video"""
        # Extract audio from video
        audio_path = self.extract_audio_from_video(video_path)
        
        if not audio_path or not os.path.exists(audio_path):
            return pd.DataFrame(), {}
        
        # Analyze audio features
        audio_features = self.analyze_audio_file(audio_path)
        
        # Transcribe speech
        transcript, confidence = self.transcribe_speech(audio_path)
        
        # Analyze voice quality
        voice_quality = self.analyze_voice_quality(audio_path)
        
        # Create time-series data (simplified - sample every second)
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        
        time_data = []
        for t in np.arange(0, duration, 1):
            start_sample = int(t * sr)
            end_sample = int(min((t + 1) * sr, len(y)))
            
            if end_sample > start_sample:
                segment = y[start_sample:end_sample]
                
                # Calculate features for this segment
                rms = np.mean(librosa.feature.rms(y=segment))
                zcr = np.mean(librosa.feature.zero_crossing_rate(segment))
                
                time_data.append({
                    'timestamp': t,
                    'energy': rms,
                    'zero_crossing_rate': zcr,
                    'speech_present': rms > 0.01
                })
        
        df = pd.DataFrame(time_data)
        
        # Clean up temp file
        try:
            os.unlink(audio_path)
        except:
            pass
        
        # Compile statistics
        statistics = {
            'duration': duration,
            'transcript': transcript,
            'transcription_confidence': confidence,
            'words_spoken': len(transcript.split()) if transcript else 0,
            'speech_percentage': df['speech_present'].mean() * 100 if len(df) > 0 else 0,
            **voice_quality,
            **{k: v for k, v in audio_features.items() if k not in voice_quality}
        }
        
        return df, statistics