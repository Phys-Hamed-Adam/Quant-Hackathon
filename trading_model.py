import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GRU, Dropout, Conv1D, GlobalAveragePooling1D, BatchNormalization, SpatialDropout1D, Bidirectional, LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt

class TradingModel(Model):
    def __init__(self, input_shape=None):
        super().__init__()
        self.history = None

        # 1. Bottleneck the Conv block and increase L2 / Dropout
        self.conv1 = Conv1D(
            filters=64,             # <-- Reduced from 128
            kernel_size=5, 
            padding="same",
            kernel_regularizer=l2(1e-3) # <-- Increased penalty
        )
        self.leaky_conv = LeakyReLU(alpha=0.01)
        self.bn1 = BatchNormalization()
        self.spatial_drop = SpatialDropout1D(0.4) # <-- Increased to 40%

        # 2. Shrink GRUs and increase Dropout
        self.gru1 = Bidirectional(GRU(32, return_sequences=True, kernel_regularizer=l2(1e-3))) # <-- Reduced to 32 units, increased L2
        self.dropout1 = Dropout(0.5) # <-- Increased to 50%

        self.gru2 = Bidirectional(GRU(32, kernel_regularizer=l2(1e-3)))
        self.dropout2 = Dropout(0.5)

        # 3. Shrink Head
        self.dense1 = Dense(16, kernel_regularizer=l2(1e-3)) # <-- Reduced to 16
        
        # BINARY OUTPUT: 0 (Down) or 1 (Up)
        self.out = Dense(1, activation='sigmoid')  

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.spatial_drop(x, training=training)

        x = self.gru1(x)
        x = self.dropout1(x, training=training)

        x = self.gru2(x)
        x = self.dropout2(x, training=training)

        x = self.dense1(x)
        return self.out(x)
    
    def compile(self, optimizer=None, loss='binary_crossentropy', metrics=['accuracy']):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def fit(self, x, y, epochs=100, batch_size=32, validation_split=0.2, validation_data=None, callbacks=None):
        if callbacks is None:
            callbacks = []
            
        lr_callback = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        )
        early_stop = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )

        callbacks.extend([lr_callback, early_stop])
        
        # --- THE MOST IMPORTANT FIX FOR THE 55% CEILING ---
        # Calculate weights to penalize the model for just guessing "1" (Up)
        zero_count = np.sum(y == 0)
        one_count = np.sum(y == 1)
        total = len(y)
        
        weight_for_0 = (1 / zero_count) * (total / 2.0)
        weight_for_1 = (1 / one_count) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        
        print(f"Applying Class Weights: 0 (Down): {weight_for_0:.2f}, 1 (Up): {weight_for_1:.2f}")

        self.history = super().fit(
            x, y, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            callbacks=callbacks,
            class_weight=class_weight,
            shuffle=True
        )
        return self.history
    
    def plot_model(self):
        plot_model(self, show_shapes=True, show_layer_names=True)
    
    def plot_training_history(self, history=None):
        if history is None:
            history = self.history

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')

        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Val Loss')

        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        if 'accuracy' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            if 'val_accuracy' in history.history:
                plt.plot(history.history['val_accuracy'], label='Val Accuracy')
            plt.title('Accuracy Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

        plt.tight_layout()
        plt.show()