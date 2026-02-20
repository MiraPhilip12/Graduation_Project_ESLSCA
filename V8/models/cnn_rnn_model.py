import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .base_model import BaseModel

class CNNRNNModel(BaseModel):
    """CNN-RNN hybrid model"""
    
    def __init__(self, input_shape, num_classes=3, name="CNN_RNN"):
        super().__init__(name, input_shape, num_classes)
        
    def build_model(self):
        """Build CNN-RNN architecture adapted for single timestep"""
        model = keras.Sequential()
        
        # CNN layers
        model.add(layers.Conv1D(filters=64, kernel_size=1, activation='relu',
                               input_shape=self.input_shape, padding='same'))
        # Remove pooling for single timestep
        model.add(layers.Dropout(0.2))
        
        # Reshape for RNN
        model.add(layers.Reshape((1, 64)))
        
        # RNN layers
        model.add(layers.SimpleRNN(64, return_sequences=False))
        model.add(layers.Dropout(0.3))
        
        # Dense layers
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