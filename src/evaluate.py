# src/evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.preprocess import load_data, scale_data, create_sequences

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "lstm_stock_model.h5")

if __name__ == "__main__":
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Model not trained yet. Run `python -m src.train` first!")

    model = load_model(MODEL_PATH)

    # Load data
    df = load_data("AAPL")
    scaled_data, scaler = scale_data(df)
    X, y = create_sequences(scaled_data)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Predict
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_true = scaler.inverse_transform(y.reshape(-1, 1))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="True Prices")
    plt.plot(predictions, label="Predicted Prices")
    plt.legend()
    plt.title("LSTM Stock Price Prediction (AAPL)")
    plt.show()
