import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .base_model import BaseModel

class CNNLSTMModel(BaseModel):
    """CNN-LSTM hybrid model for action recognition"""
    
    def __init__(self, input_shape, num_classes=3, name="CNN_LSTM"):
        super().__init__(name, input_shape, num_classes)
        
    def build_model(self):
        """Build CNN-LSTM architecture adapted for single timestep"""
        model = keras.Sequential()
        
        # For single timestep, we need to adjust the architecture
        # Instead of multiple conv+pool layers, use a single conv layer
        # and then reshape for LSTM
        
        # CNN layers for spatial feature extraction
        model.add(layers.Conv1D(filters=64, kernel_size=1, activation='relu', 
                               input_shape=self.input_shape, padding='same'))
        model.add(layers.BatchNormalization())
        # Remove pooling for single timestep
        model.add(layers.Dropout(0.25))
        
        # Reshape to have timesteps dimension for LSTM
        model.add(layers.Reshape((1, 64)))
        
        # LSTM layers for temporal dependencies
        model.add(layers.LSTM(64, return_sequences=False))
        model.add(layers.Dropout(0.3))
        
        # Dense layers for classification
        model.add(layers.Dense(64, activation='relu', kernel_regularizer='l2'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model