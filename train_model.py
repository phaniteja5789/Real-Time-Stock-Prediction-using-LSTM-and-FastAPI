import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

MODEL_DIR = "saved_models"
MODEL_NAME = "lstm_stock_model.keras"

def download_stock_data(ticker="AAPL", period="10y"):  # Increased period for more data
    df = yf.download(ticker, period=period, interval="1d")
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

def create_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size - 7 + 1):
        X.append(series[i:(i + window_size)])
        y.append(series[(i + window_size):(i + window_size + 7)])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(7)  # Predicting 7 days
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def main():
    df = download_stock_data(ticker="AAPL", period="10y")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    
    # Using a different train-test split
    train_size = int(len(scaled_data) * 0.7)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]
    
    window_size = 365  # 1 year window
    X_train, y_train = create_dataset(train_data, window_size)
    X_test, y_test = create_dataset(test_data, window_size)

    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Insufficient data for training set.")
    if X_test.size == 0 or y_test.size == 0:
        raise ValueError("Insufficient data for testing set.")
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_lstm_model((window_size, 1))
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, MODEL_NAME))

    import joblib
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    main()
