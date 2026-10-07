import os
import joblib
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vision_module import router as vision_router
from chat_module import router as chat_router

app = FastAPI(title="Used Car Resale Value Prediction Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(vision_router)
app.include_router(chat_router)

MODEL_FILE = "car_price_model.pkl"
DATA_FILE = "used_cars_real.csv"
model = None
valid_cars_db = {}

# 1. Load ML Model
if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
    except Exception as e:
        print(f"Warning: Failed to load {MODEL_FILE}: {e}")

# 2. Build In-Memory Database from CSV for Validation
if os.path.exists(DATA_FILE):
    try:
        df = pd.read_csv(DATA_FILE)
        df.columns = df.columns.str.strip().str.lower()
        for brand, group in df.groupby("brand"):
            # Map every model to its corresponding brand in lowercase
            valid_cars_db[str(brand).strip().lower()] = set(str(m).strip().lower() for m in group["model"].unique())
        print(f"Car database loaded: {len(valid_cars_db)} brands indexed.")
    except Exception as e:
        print(f"Warning: Could not build validation database: {e}")
else:
    # Fallback to defaults if CSV is missing
    valid_cars_db = {
        "maruti suzuki": {"swift", "baleno", "wagonr"},
        "hyundai": {"i20", "creta", "venue"},
        "tata": {"nexon", "harrier", "tiago"},
        "mahindra": {"thar", "xuv700", "scorpio"},
        "honda": {"city", "amaze", "civic"},
        "toyota": {"innova", "fortuner", "glanza"}
    }

class VehicleInput(BaseModel):
    brand: str
    model: str
    year: int
    mileage: int
    condition: str = "Good"
    fuel_type: str = "Petrol"
    transmission: str = "Manual"
    owners: int = 1

def validate_vehicle(vehicle: VehicleInput):
    brand = vehicle.brand.strip().lower()
    model = vehicle.model.strip().lower()
    
    if brand not in valid_cars_db:
        raise HTTPException(status_code=404, detail=f"Brand '{vehicle.brand}' not found in our market database.")
        
    if model not in valid_cars_db[brand]:
        supported = [m.title() for m in valid_cars_db[brand]]
        # Show a few examples of valid models for that brand
        hint = ", ".join(supported[:5]) + ("..." if len(supported) > 5 else "")
        raise HTTPException(status_code=404, detail=f"Model '{vehicle.model}' not found for brand '{vehicle.brand}'. Known models: {hint}")

def compute_depreciation(vehicle: VehicleInput, target_year: int, current_year: int = 2026) -> float:
    age = max(0, target_year - vehicle.year)
    base_price = 850000.0  
    depreciated = base_price * ((1.0 - 0.085) ** age)

    if target_year <= current_year:
        years_elapsed = max(1, current_year - vehicle.year)
        sim_mileage = (vehicle.mileage / years_elapsed) * age
    else:
        sim_mileage = vehicle.mileage + (12000 * (target_year - current_year))

    mileage_deduction = (sim_mileage / 10000.0) * 15000.0
    owner_deduction = max(0, vehicle.owners - 1) * 35000.0

    condition_mult = {"Excellent": 1.10, "Good": 1.0, "Fair": 0.82, "Poor": 0.65}.get(vehicle.condition, 1.0)
    fuel_mult = {"Diesel": 1.04, "Petrol": 1.0, "EV": 1.12, "CNG": 0.94}.get(vehicle.fuel_type, 1.0)
    trans_mult = {"Automatic": 1.08, "Manual": 1.0}.get(vehicle.transmission, 1.0)

    statutory_penalty = 1.0
    if vehicle.fuel_type == "Diesel" and age >= 10:
        statutory_penalty = 0.55
    elif age >= 15:
        statutory_penalty = 0.60

    net_value = (depreciated - mileage_deduction - owner_deduction) * condition_mult * fuel_mult * trans_mult * statutory_penalty
    return max(45000.0, net_value)

@app.get("/api/car-database")
def get_car_database():
    # Convert sets to lists for JSON serialization and sort them alphabetically
    return {brand: sorted(list(models)) for brand, models in valid_cars_db.items()}

@app.get("/")
def serve_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "API Running."}

@app.post("/api/predict")
def predict_price(vehicle: VehicleInput):
    # Run the strict database check first
    validate_vehicle(vehicle)
    
    current_year = 2026
    if model is not None:
        try:
            df = pd.DataFrame([{
                "brand": vehicle.brand,
                "model": vehicle.model,
                "year": vehicle.year,
                "mileage": vehicle.mileage,
                "condition": vehicle.condition,
                "fuel_type": vehicle.fuel_type,
                "transmission": vehicle.transmission,
                "owners": vehicle.owners
            }])
            predicted_value = float(model.predict(df)[0])
            return {
                "estimated_price": round(max(45000.0, predicted_value), 2),
                "currency": "INR",
                "status": "success"
            }
        except Exception:
            pass

    val = compute_depreciation(vehicle, target_year=current_year, current_year=current_year)
    return {
        "estimated_price": round(val, 2),
        "currency": "INR",
        "status": "fallback_estimation"
    }

@app.post("/api/visual-analysis")
def get_visual_analysis(vehicle: VehicleInput):
    # Ensure visual analysis also validates the input
    validate_vehicle(vehicle)
    
    current_year = 2026
    end_year = current_year + 5
    years = list(range(vehicle.year, end_year + 1))

    prices = []
    yoy_drops = []
    context_events = []

    macro_timeline = {
        2024: "National Scrappage Policy incentives enacted.",
        2025: "EV battery health certification standardizations.",
        2026: "Current Valuation Benchmark.",
        2027: "Projected standard YoY fleet depreciation.",
        2028: "Projected urban clean-air regulatory tightening.",
        2029: "Projected 10-year lifecycle threshold regulations.",
        2030: "Projected mass market EV penetration impact.",
        2031: "Mandatory automated testing station (ATS) test."
    }

    for y in years:
        val = compute_depreciation(vehicle, target_year=y, current_year=current_year)
        prices.append(round(val, 0))
        context_events.append(macro_timeline.get(y, "Standard algorithmic depreciation baseline."))

    for i in range(len(prices)):
        if i == 0:
            yoy_drops.append("Initial")
        else:
            drop = prices[i-1] - prices[i]
            pct = (drop / prices[i-1]) * 100 if prices[i-1] > 0 else 0
            yoy_drops.append(f"-{pct:.1f}% YoY" if pct > 0 else f"+{abs(pct):.1f}% YoY")

    try:
        current_idx = years.index(current_year)
    except ValueError:
        current_idx = 0
        
    current_val = prices[current_idx]
    final_val = prices[-1]
    
    if current_val > 0 and (len(prices) - 1 - current_idx) > 0:
        total_drop_pct = ((current_val - final_val) / current_val) * 100
        years_diff = len(prices) - 1 - current_idx
        avg_depreciation = round(total_drop_pct / years_diff, 1)
    else:
        avg_depreciation = 8.5

    specs = {
        "engine": "Electric / Li-ion" if vehicle.fuel_type == "EV" else f"Inline {vehicle.fuel_type}",
        "efficiency": "N/A" if vehicle.fuel_type == "EV" else "Standard ARAI",
        "defects": "Visual wear profiling applied.",
        "avg_depreciation": f"{avg_depreciation}% annually"
    }

    return {
        "years": years,
        "prices": prices,
        "yoy_drops": yoy_drops,
        "context": context_events,
        "specs": specs
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)