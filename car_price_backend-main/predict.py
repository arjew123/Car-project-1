import os
import joblib
import pandas as pd

MODEL_FILE = "car_price_model.pkl"

def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found. Please run train_model.py first.")
    return joblib.load(MODEL_FILE)

def make_prediction(vehicle_data: dict) -> float:
    """
    Accepts a dictionary of vehicle features and returns the estimated price.
    """
    model = load_model()
    
    # Convert the single dictionary into a DataFrame with one row
    df = pd.DataFrame([vehicle_data])
    
    # The pipeline handles all scaling and encoding automatically
    predicted_price = model.predict(df)[0]
    return max(45000.0, float(predicted_price))

if __name__ == "__main__":
    print("--- Vehicle Resale Price Predictor (CLI Test) ---")
    
    # Sample Test Case 1: Standard Commuter
    test_car_1 = {
        "brand": "Hyundai",
        "model": "Creta",
        "year": 2020,
        "mileage": 55000,
        "condition": "Good",
        "fuel_type": "Diesel",
        "transmission": "Automatic",
        "owners": 1
    }
    
    # Sample Test Case 2: Heavily Depreciated
    test_car_2 = {
        "brand": "Maruti Suzuki",
        "model": "Swift",
        "year": 2014,
        "mileage": 120000,
        "condition": "Poor",
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "owners": 3
    }

    try:
        price_1 = make_prediction(test_car_1)
        print(f"\nTest Car 1 (2020 Hyundai Creta Diesel Auto): ₹{price_1:,.2f}")
        
        price_2 = make_prediction(test_car_2)
        print(f"Test Car 2 (2014 Maruti Swift Petrol Manual): ₹{price_2:,.2f}")
    except Exception as e:
        print(f"Prediction failed: {e}")