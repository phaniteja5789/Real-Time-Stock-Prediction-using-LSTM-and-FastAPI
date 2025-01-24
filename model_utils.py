import os
import joblib
import numpy as np
import tensorflow as tf

MODEL_DIR = "saved_models"
MODEL_NAME = "lstm_stock_model.keras"

def load_model_and_scaler():
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def prepare_input_for_inference(recent_data, window_size, scaler):
    recent_data_array = np.array(recent_data).reshape(-1, 1)
    scaled_data = scaler.transform(recent_data_array)
    
    X_input = scaled_data.reshape(1, window_size, 1)
    return X_input
