import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .base_model import BaseModel

class LSTMModel(BaseModel):
    """LSTM model for temporal sequence analysis"""
    
    def __init__(self, input_shape, num_classes=3, name="LSTM"):
        super().__init__(name, input_shape, num_classes)
        
    def build_model(self):
        """Build LSTM architecture"""
        model = keras.Sequential()
        
        # First LSTM layer
        model.add(layers.LSTM(128, return_sequences=True, 
                             input_shape=self.input_shape))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        
        # Second LSTM layer
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        
        # Third LSTM layer
        model.add(layers.LSTM(32))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        
        # Dense layers
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(32, activation='relu'))
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