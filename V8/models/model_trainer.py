import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import json
from datetime import datetime
import pickle

from .cnn_lstm_model import CNNLSTMModel
from .lstm_model import LSTMModel
from .bilstm_model import BiLSTMModel
from .gru_model import GRUModel
from .rnn_model import RNNModel
from .cnn_rnn_model import CNNRNNModel
from .keras_model import KerasModel
from .pycaret_model import PyCaretModel

class ModelTrainer:
    """Train and compare multiple models"""
    
    def __init__(self, models_dir="saved_models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def initialize_models(self, input_shape, num_classes=3):
        """Initialize all models"""
        
        # For sequence models (LSTM, CNN, etc.) with flat features, we need to reshape
        if len(input_shape) == 1:  # Flat features (features,)
            # For sequence models, we'll reshape to (1, features) to simulate a single timestep
            sequence_shape = (1, input_shape[0])
            
            models = {
                'CNN_LSTM': CNNLSTMModel(sequence_shape, num_classes),
                'LSTM': LSTMModel(sequence_shape, num_classes),
                'BiLSTM': BiLSTMModel(sequence_shape, num_classes),
                'GRU': GRUModel(sequence_shape, num_classes),
                'RNN': RNNModel(sequence_shape, num_classes),
                'CNN_RNN': CNNRNNModel(sequence_shape, num_classes),
                'Keras_Deep': KerasModel(input_shape, num_classes, name="Keras_Deep", architecture='deep'),
                'Keras_Medium': KerasModel(input_shape, num_classes, name="Keras_Medium", architecture='medium')
            }
        else:  # For traditional ML (flat features)
            models = {
                'PyCaret': PyCaretModel("PyCaret", num_classes),
                'Keras_Deep': KerasModel(input_shape, num_classes, name="Keras_Deep", architecture='deep')
            }
        
        self.models = models
        return models
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, 
                     X_test=None, y_test=None, epochs=50, batch_size=32):
        """Train all initialized models"""
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            print('='*50)
            history = model.train(X_train, y_train, X_val, y_val, epochs, batch_size)


            # After training, evaluate if test data provided
            if X_test is not None and y_test is not None:
                try:
                    # For sequence models with flat features, reshape the data
                    if name in ['CNN_LSTM', 'LSTM', 'BiLSTM', 'GRU', 'RNN', 'CNN_RNN']:
                        X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
                    else:
                        X_test_reshaped = X_test
                    
                    # Get predictions first to check shape
                    y_pred = model.predict(X_test_reshaped)
                    
                    # Ensure y_test and y_pred have same length
                    if len(y_test) != len(y_pred):
                        print(f"Warning: Shape mismatch - y_test: {len(y_test)}, y_pred: {len(y_pred)}")
                        # Take the minimum length
                        min_len = min(len(y_test), len(y_pred))
                        y_test_trimmed = y_test[:min_len]
                        X_test_trimmed = X_test_reshaped[:min_len] if X_test_reshaped is not None else None
                    else:
                        y_test_trimmed = y_test
                        X_test_trimmed = X_test_reshaped
                    
                    metrics = model.evaluate(X_test_trimmed, y_test_trimmed)
                    
                    results[name] = {
                        'metrics': metrics,
                        'history': history.history if history else None,
                        'model': model
                    }
                    
                    print(f"\n{name} Results:")
                    print(f"Accuracy: {metrics['accuracy']:.4f}")
                    print(f"Precision: {metrics['precision']:.4f}")
                    print(f"Recall: {metrics['recall']:.4f}")
                    print(f"F1-Score: {metrics['f1_score']:.4f}")
                    
                except Exception as e:
                    print(f"Error evaluating {name}: {str(e)}")
                    results[name] = {
                        'metrics': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0},
                        'error': str(e)
                    }
        
        self.results = results
        return results
    
    def compare_models(self, metric='accuracy'):
        """Compare all models based on a specific metric"""
        comparison = {}
        
        for name, result in self.results.items():
            if 'metrics' in result:
                comparison[name] = result['metrics'][metric]
            else:
                comparison[name] = 0
        
        # Sort by metric value
        comparison = dict(sorted(comparison.items(), key=lambda x: x[1], reverse=True))
        
        return comparison
    
    def get_best_model(self, metric='accuracy'):
        """Get the best performing model"""
        comparison = self.compare_models(metric)
        if comparison:
            best_name = list(comparison.keys())[0]
            return best_name, self.results[best_name]['model']
        return None, None
    
    def save_all_models(self):
        """Save all trained models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, result in self.results.items():
            if 'model' in result:
                model_path = os.path.join(self.models_dir, f"{name}_{timestamp}")
                result['model'].save_model(model_path)
        
        # Save results summary
        summary = {}
        for name, result in self.results.items():
            if 'metrics' in result:
                summary[name] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                    for k, v in result['metrics'].items() if k != 'confusion_matrix'
                }
        
        summary_path = os.path.join(self.models_dir, f"results_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nResults summary saved to {summary_path}")
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, f"scaler_{timestamp}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return summary
    
    def load_model(self, model_name, model_path):
        """Load a specific model"""
        if model_name in self.models:
            self.models[model_name].load_model(model_path)
            return self.models[model_name]
        else:
            print(f"Model {model_name} not found in initialized models")
            return None
    
    def create_ensemble_prediction(self, X, weights=None):
        """Create ensemble prediction from all models"""
        predictions = []
        probabilities = []
        valid_models = []
        
        for name, result in self.results.items():
            if 'model' in result and result['model'] is not None:
                try:
                    # Handle reshaping for sequence models
                    if name in ['CNN_LSTM', 'LSTM', 'BiLSTM', 'GRU', 'RNN', 'CNN_RNN']:
                        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
                    else:
                        X_reshaped = X
                    
                    # Get probabilities
                    proba = result['model'].predict_proba(X_reshaped)
                    
                    # Ensure proba is 2D
                    if len(proba.shape) == 1:
                        proba = proba.reshape(-1, 1)
                    
                    probabilities.append(proba)
                    
                    # Get predictions
                    pred = result['model'].predict(X_reshaped)
                    predictions.append(pred)
                    valid_models.append(name)
                    
                except Exception as e:
                    print(f"Error with model {name} in ensemble: {e}")
        
        if not probabilities:
            return None
        
        # Ensure all probability arrays have same shape
        min_samples = min([p.shape[0] for p in probabilities])
        probabilities = [p[:min_samples] for p in probabilities]
        
        # Weighted average of probabilities
        if weights is None:
            # Equal weights
            weights = [1/len(probabilities)] * len(probabilities)
        
        weighted_proba = np.zeros_like(probabilities[0])
        for proba, weight in zip(probabilities, weights):
            weighted_proba += weight * proba
        
        # Final prediction
        ensemble_pred = np.argmax(weighted_proba, axis=1)
        
        # Voting ensemble
        if predictions:
            # Trim predictions to same length
            predictions = [p[:min_samples] for p in predictions]
            predictions = np.array(predictions)
            voting_pred = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(), 
                axis=0, 
                arr=predictions
            )
        else:
            voting_pred = ensemble_pred
        
        return {
            'weighted_proba': weighted_proba,
            'ensemble_pred': ensemble_pred,
            'voting_pred': voting_pred,
            'individual_predictions': predictions if predictions else [],
            'valid_models': valid_models
        }