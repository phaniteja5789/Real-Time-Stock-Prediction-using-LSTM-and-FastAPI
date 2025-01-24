# Real-Time-Stock-Prediction-using-LSTM-and-FastAPI

Project Overview

This repository demonstrates a workflow for predicting stock closing prices using Long Short-Term Memory (LSTM) neural networks.
It includes a FastAPI backend for model inference.
Key Features

Automated Data Collection: Retrieves historical stock prices via yfinance.
Data Preprocessing & Scaling: Uses MinMaxScaler to transform price data.
LSTM Deep Learning Model: Captures time-series dependencies with sliding windows of 60 days.
FastAPI Inference Endpoint: Exposes a /predict route to get next-day price predictions.


The model uses a 2-layer LSTM:
LSTM layer (50 units) with return_sequences=True
Dropout (20%)
LSTM layer (50 units)
Dropout (20%)
Dense output (7 unit for final prediction) ==> To predict 7 days prediction
Loss Function: Mean Squared Error (MSE)
Optimizer: Adam

Outputs:
![image](https://github.com/user-attachments/assets/1b86a4f8-cd7d-46dc-9349-ebda781282f1)

![image](https://github.com/user-attachments/assets/c1db5aaf-8770-4a24-ba0a-d9ae508ce4de)

Network metrics can be visualized in the Tensor Board with the help of callbacks of Tensor Flow

![image](https://github.com/user-attachments/assets/acbdb7e3-28ec-490d-b1a1-1c9c83d24490)


