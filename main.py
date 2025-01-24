from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from model_utils import load_model_and_scaler, prepare_input_for_inference

# FastAPI initialization
app = FastAPI()

# Load model/scaler at startup
model, scaler = load_model_and_scaler()

# We need to define a Pydantic schema for incoming data
class StockData(BaseModel):
    recent_closes: list[float]

@app.api_route("/predict", methods=["GET", "POST"])
def predict_stock_price(data: StockData):
    try:
        window_size = 365  # must match training window
        if len(data.recent_closes) < window_size:
            return {
                "error": f"Insufficient data! Provide at least {window_size} prices."
            }
        
        # Prepare the input for LSTM
        X_input = prepare_input_for_inference(
            recent_data=data.recent_closes[-window_size:],  # last 1 year closes
            window_size=window_size,
            scaler=scaler
        )
        
        # Model inference
        prediction_scaled = model.predict(X_input)
        import numpy as np
        predicted_prices = scaler.inverse_transform(prediction_scaled).flatten()
        predicted_prices = [f"{price:.2f}" for price in predicted_prices]
        return {"predicted_prices": ", ".join(predicted_prices)}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
