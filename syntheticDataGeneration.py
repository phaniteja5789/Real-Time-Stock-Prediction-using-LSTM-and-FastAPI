import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib
import os

# Constants
MODEL_DIR = "saved_models"
MODEL_NAME = "lstm_stock_model.keras"
DAYS = 365

# Load the scaler and model
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
model = load_model(os.path.join(MODEL_DIR, MODEL_NAME))

def generate_dummy_data(days=DAYS):
    np.random.seed(0)  # for reproducibility
    time = np.arange(days)
    # Generate a trend
    trend = time * 0.02
    # Add seasonality
    seasonal = 10 * np.sin(time * 2 * np.pi / 180)
    # Add noise
    noise = 2 * np.random.normal(size=time.size)
    prices = 100 + trend + seasonal + noise
    return prices

def prepare_data(prices):
    df = pd.DataFrame(prices, columns=['Close'])
    scaled_data = scaler.transform(df.values)
    X = scaled_data.reshape(1, DAYS, 1)  # Reshape for the LSTM input
    return X

def make_prediction(X):
    prediction = model.predict(X)
    predicted_prices = scaler.inverse_transform(prediction)  # Inverse transform to get actual prices
    return predicted_prices.flatten()

def main():
    dummy_prices = generate_dummy_data()
    
    X = prepare_data(dummy_prices)
    print(dummy_prices.tolist())
    predicted_prices = make_prediction(X)
    print(predicted_prices)
    print("Predicted Prices for the next 7 days:", predicted_prices)
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(DAYS), dummy_prices, label='Dummy Closing Prices')
    plt.plot(np.arange(DAYS, DAYS + 7), predicted_prices, label='Predicted Next 7 Days', marker='o')
    plt.title('Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
