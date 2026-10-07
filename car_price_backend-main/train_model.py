import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DATA_FILE = "used_cars_real.csv"
MODEL_FILE = "car_price_model.pkl"

def generate_synthetic_indian_market_data(samples=2000):
    """Generates a realistic dataset if the real CSV is missing."""
    print("Generating synthetic Indian market dataset for training...")
    np.random.seed(42)
    
    brands = ["Maruti Suzuki", "Hyundai", "Tata", "Mahindra", "Honda", "Toyota"]
    models_map = {
        "Maruti Suzuki": ["Swift", "Baleno", "WagonR"],
        "Hyundai": ["i20", "Creta", "Venue"],
        "Tata": ["Nexon", "Harrier", "Tiago"],
        "Mahindra": ["Thar", "XUV700", "Scorpio"],
        "Honda": ["City", "Amaze", "Civic"],
        "Toyota": ["Innova", "Fortuner", "Glanza"]
    }
    
    data = []
    for _ in range(samples):
        brand = np.random.choice(brands)
        model = np.random.choice(models_map[brand])
        year = np.random.randint(2012, 2026)
        age = 2026 - year
        mileage = int(np.random.normal(12000 * age, 5000))
        mileage = max(1000, mileage) 
        
        condition = np.random.choice(["Excellent", "Good", "Fair", "Poor"], p=[0.2, 0.5, 0.2, 0.1])
        fuel = np.random.choice(["Petrol", "Diesel", "CNG", "EV"], p=[0.5, 0.35, 0.1, 0.05])
        trans = np.random.choice(["Manual", "Automatic"], p=[0.6, 0.4])
        owners = np.random.choice([1, 2, 3, 4], p=[0.6, 0.3, 0.08, 0.02])
        
        # Base pricing logic for realistic targets
        base_price = np.random.uniform(600000, 1500000)
        depreciated = base_price * ((1.0 - 0.085) ** age)
        mileage_ded = (mileage / 10000) * 12000
        cond_mult = {"Excellent": 1.1, "Good": 1.0, "Fair": 0.85, "Poor": 0.65}[condition]
        owner_ded = (owners - 1) * 30000
        
        price = max(50000, (depreciated - mileage_ded - owner_ded) * cond_mult)
        
        data.append([brand, model, year, mileage, condition, fuel, trans, owners, int(price)])
        
    df = pd.DataFrame(data, columns=["brand", "model", "year", "mileage", "condition", "fuel_type", "transmission", "owners", "price"])
    df.to_csv(DATA_FILE, index=False)
    return df

def train():
    if not os.path.exists(DATA_FILE):
        df = generate_synthetic_indian_market_data()
    else:
        df = pd.read_csv(DATA_FILE)

    print(f"Dataset loaded with {len(df)} records.")

    # Strip hidden whitespaces and enforce lowercase to prevent KeyErrors
    df.columns = df.columns.str.strip().str.lower()

    # 1. Separate Features (X) and Target (y)
    X = df.drop("price", axis=1)
    y = df["price"]

    # 2. Define Preprocessing Steps
    numeric_features = ["year", "mileage", "owners"]
    categorical_nominal = ["brand", "model", "fuel_type", "transmission"]
    categorical_ordinal = ["condition"]

    # Condition mapping: Poor (0) to Excellent (3)
    condition_categories = [["Poor", "Fair", "Good", "Excellent"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat_nom", OneHotEncoder(handle_unknown="ignore"), categorical_nominal),
            ("cat_ord", OrdinalEncoder(categories=condition_categories), categorical_ordinal)
        ]
    )

    # 3. Build the Pipeline with a Random Forest Regressor
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15))
    ])

    # 4. Split and Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model pipeline (this may take a moment)...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate Accuracy
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Model Performance ---")
    print(f"Mean Absolute Error: ₹{mae:,.2f}")
    print(f"R-squared Score: {r2:.4f} (Closer to 1.0 is better)")

    # 6. Save the fully contained pipeline
    joblib.dump(pipeline, MODEL_FILE)
    print(f"\nModel successfully saved to {MODEL_FILE}")

if __name__ == "__main__":
    train()