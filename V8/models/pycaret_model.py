import numpy as np
import pandas as pd
from pycaret.classification import *
from .base_model import BaseModel
import joblib
import os

class PyCaretModel(BaseModel):
    """PyCaret model for classical ML approaches"""
    
    def __init__(self, name="PyCaret_Model", num_classes=3):
        super().__init__(name, None, num_classes)
        self.setup_done = False
        self.best_model = None
        
    def build_model(self):
        """PyCaret doesn't need traditional build"""
        pass
    
    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train using PyCaret"""
        
        # Prepare data
        if isinstance(X_train, np.ndarray):
            # Convert to DataFrame
            columns = [f'feature_{i}' for i in range(X_train.shape[1])]
            train_df = pd.DataFrame(X_train, columns=columns)
            train_df['label'] = y_train
            
            if X_val is not None:
                val_df = pd.DataFrame(X_val, columns=columns)
                val_df['label'] = y_val
        else:
            train_df = X_train.copy()
            train_df['label'] = y_train
            
            if X_val is not None:
                val_df = X_val.copy()
                val_df['label'] = y_val
        
        # Setup PyCaret - remove silent parameter
        exp_name = setup(
            data=train_df,
            target='label',
            session_id=123,
            log_experiment=False,
            html=False,  # Use html=False instead of silent
            verbose=False
        )
        self.setup_done = True
        
        # Compare models and get best
        best = compare_models(
            include=['lr', 'rf', 'xgboost', 'lightgbm', 'dt', 'nb'],
            n_select=1,
            verbose=False
        )
        
        self.best_model = best
        
        # Train specific models for ensemble
        models_to_train = ['lr', 'rf', 'xgboost', 'lightgbm']
        self.trained_models = []
        
        for model_name in models_to_train:
            try:
                model = create_model(model_name, verbose=False)
                self.trained_models.append(model)
            except:
                pass
        
        # Create ensemble
        if len(self.trained_models) > 1:
            self.best_model = blend_models(
                estimator_list=self.trained_models,
                verbose=False
            )
        
        # Finalize model
        self.best_model = finalize_model(self.best_model)
        
        return self.best_model
    
    def predict(self, X):
        """Make predictions"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        if isinstance(X, np.ndarray):
            columns = [f'feature_{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=columns)
        else:
            X_df = X.copy()
        
        predictions = predict_model(self.best_model, data=X_df, verbose=False)
        return predictions['prediction_label'].values
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        
        if isinstance(X, np.ndarray):
            columns = [f'feature_{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=columns)
        else:
            X_df = X.copy()
        
        predictions = predict_model(self.best_model, data=X_df, verbose=False)
        
        # Get probability columns
        prob_cols = [col for col in predictions.columns if 'prediction_score' in col]
        if prob_cols:
            probas = predictions[prob_cols].values
        else:
            # Fallback
            probas = np.zeros((len(predictions), self.num_classes))
            for i, row in predictions.iterrows():
                probas[i, int(row['prediction_label'])] = 1.0
        
        return probas
    
    def save_model(self, path):
        """Save the PyCaret model"""
        if self.best_model is None:
            raise ValueError("No model to save!")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_model(self.best_model, path, verbose=False)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the PyCaret model"""
        self.best_model = load_model(path, verbose=False)
        print(f"Model loaded from {path}")