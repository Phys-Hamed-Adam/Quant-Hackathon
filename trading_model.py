from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt

class TradingModel(Model):
    def __init__(self):
        super().__init__()

        # fields
        self.history = None

        # Conv block
        self.conv1 = Conv1D(64, 3, activation='relu')
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPooling1D(2)

        # GRU stack
        self.gru1 = GRU(50, return_sequences=True)
        self.dropout1 = Dropout(0.2)

        self.gru2 = GRU(50)
        self.dropout2 = Dropout(0.2)

        # Head
        self.dense1 = Dense(25, activation='relu')
        self.out = Dense(1, activation='sigmoid')   # Binary classification output (up/down)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        x = self.gru1(x)
        x = self.dropout1(x, training=training)

        x = self.gru2(x)
        x = self.dropout2(x, training=training)

        x = self.dense1(x)
        return self.out(x)
    
    def compile(self, optimizer='adam', loss='binary_crossentropy'):
        super().compile(optimizer=optimizer, loss=loss)
    
    def fit(self, x, y=None, epochs=1, batch_size=32, callbacks=[]):
        lr_callback = ReduceLROnPlateau(
            monitor='val_loss',      # metric to monitor
            factor=0.5,              # multiply LR by this factor
            patience=5,              # wait this many epochs before reducing
            min_lr=1e-6,             # lower bound for LR
            verbose=1
        )

        callbacks.append(lr_callback)
        self.history = super().fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
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