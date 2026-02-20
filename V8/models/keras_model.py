import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .base_model import BaseModel

class KerasModel(BaseModel):
    """Flexible Keras model with configurable architecture"""
    
    def __init__(self, input_shape, num_classes=3, name="Keras_Model", 
                 architecture='deep'):
        super().__init__(name, input_shape, num_classes)
        self.architecture = architecture
        
    def build_model(self):
        """Build Keras model with different architectures"""
        
        if self.architecture == 'shallow':
            model = self._build_shallow()
        elif self.architecture == 'medium':
            model = self._build_medium()
        else:  # deep
            model = self._build_deep()
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def _build_shallow(self):
        """Build shallow architecture"""
        model = keras.Sequential()
        model.add(layers.Flatten(input_shape=self.input_shape))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        return model
    
    def _build_medium(self):
        """Build medium architecture"""
        model = keras.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=self.input_shape))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        return model
    
    def _build_deep(self):
        """Build deep architecture"""
        model = keras.Sequential()
        model.add(layers.Dense(256, activation='relu', input_shape=self.input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        return model