import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .base_model import BaseModel

class BiLSTMModel(BaseModel):
    """Bidirectional LSTM model for sequence analysis"""
    
    def __init__(self, input_shape, num_classes=3, name="BiLSTM"):
        super().__init__(name, input_shape, num_classes)
        
    def build_model(self):
        """Build Bidirectional LSTM architecture"""
        model = keras.Sequential()
        
        # First BiLSTM layer
        model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True),
                                      input_shape=self.input_shape))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        
        # Second BiLSTM layer
        model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        
        # Third BiLSTM layer
        model.add(layers.Bidirectional(layers.LSTM(32)))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        
        # Dense layers
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(64, activation='relu'))
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