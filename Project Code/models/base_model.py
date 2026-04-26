import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, name="base_model", input_shape=None, num_classes=3):
        self.name = name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, verbose=1):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Convert labels to categorical if needed
        if len(y_train.shape) == 1 or y_train.shape[1] != self.num_classes:
            y_train_cat = keras.utils.to_categorical(y_train, self.num_classes)
            y_val_cat = keras.utils.to_categorical(y_val, self.num_classes) if y_val is not None else None
        else:
            y_train_cat = y_train
            y_val_cat = y_val
        
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat) if y_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get predictions
        y_pred = self.predict(X_test)
        
        # Convert y_test to labels if categorical
        if len(y_test.shape) > 1 and y_test.shape[1] == self.num_classes:
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test
        
        # Ensure same length (take min if mismatched)
        min_len = min(len(y_test_labels), len(y_pred))
        if min_len < len(y_test_labels):
            print(f"Warning: Truncating evaluation from {len(y_test_labels)} to {min_len} samples")
            y_test_labels = y_test_labels[:min_len]
            y_pred = y_pred[:min_len]
        
        # Calculate metrics
        try:
            metrics = {
                'accuracy': accuracy_score(y_test_labels, y_pred),
                'precision': precision_score(y_test_labels, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test_labels, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test_labels, y_pred, average='weighted', zero_division=0)
            }
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            }
        
        # Confusion matrix
        try:
            metrics['confusion_matrix'] = confusion_matrix(y_test_labels, y_pred)
        except:
            metrics['confusion_matrix'] = np.zeros((self.num_classes, self.num_classes))
        
        return metrics
    
    def plot_confusion_matrix(self, y_test, y_pred, class_names=None):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = ['Bad', 'Moderate', 'Good']
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train')
        if 'val_accuracy' in self.history.history:
            axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Train')
        if 'val_loss' in self.history.history:
            axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        plt.tight_layout()
        return fig
    
    def save_model(self, path):
        """Save the model"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if isinstance(self.model, keras.Model):
            self.model.save(f"{path}.h5")
        else:
            joblib.dump(self.model, f"{path}.pkl")
        
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path):
        """Load the model"""
        if os.path.exists(f"{path}.h5"):
            self.model = keras.models.load_model(f"{path}.h5")
        elif os.path.exists(f"{path}.pkl"):
            self.model = joblib.load(f"{path}.pkl")
        else:
            raise FileNotFoundError(f"No model found at {path}")
        
        print(f"✅ Model loaded from {path}")