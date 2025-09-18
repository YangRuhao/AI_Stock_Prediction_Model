# src/train.py
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from src.preprocess import load_data, scale_data, create_sequences

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

if __name__ == "__main__":
    import tensorflow as tf
    print("✅ TensorFlow version:", tf.__version__)

    # Load and preprocess
    df = load_data("AAPL")
    scaled_data, scaler = scale_data(df)
    X, y = create_sequences(scaled_data)

    # Reshape for LSTM: [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train
    es = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(X, y, batch_size=32, epochs=20, callbacks=[es])

    # Save model
    os.makedirs(DATA_DIR, exist_ok=True)
    model.save(os.path.join(DATA_DIR, "lstm_stock_model.h5"))
    print("💾 Model saved to models/lstm_stock_model.h5")
