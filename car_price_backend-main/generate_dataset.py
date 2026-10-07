"""
generate_dataset.py
--------------------
Creates a synthetic used-car dataset that mimics the structure of the
popular CarDekho used-car dataset (Kaggle).

NOTE: If you have a real CSV (e.g. downloaded from Kaggle's "CarDekho
Used Car Data"), just skip this file and point `DATA_PATH` in
train_model.py to your CSV instead. The column names used here match
common CarDekho columns so swapping in a real file needs minimal changes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000  # number of synthetic rows

brands_models = {
    "Maruti": ["Swift", "Baleno", "Alto", "Dzire", "Ertiga"],
    "Hyundai": ["i20", "Creta", "Venue", "i10", "Verna"],
    "Honda": ["City", "Amaze", "Jazz", "WR-V"],
    "Toyota": ["Innova", "Fortuner", "Glanza"],
    "Tata": ["Nexon", "Altroz", "Tiago", "Harrier"],
    "Ford": ["EcoSport", "Figo"],
}

fuel_types = ["Petrol", "Diesel", "CNG", "Electric"]
transmissions = ["Manual", "Automatic"]
owner_types = ["First Owner", "Second Owner", "Third Owner"]

rows = []
for _ in range(N):
    brand = np.random.choice(list(brands_models.keys()))
    model = np.random.choice(brands_models[brand])
    year = np.random.randint(2008, 2024)
    age = 2024 - year

    fuel = np.random.choice(fuel_types, p=[0.55, 0.30, 0.10, 0.05])
    transmission = np.random.choice(transmissions, p=[0.7, 0.3])
    owner = np.random.choice(owner_types, p=[0.6, 0.3, 0.1])

    engine_cc = np.random.choice([998, 1197, 1497, 1998, 2494])
    mileage_kmpl = round(np.random.uniform(12, 26), 1)
    km_driven = int(max(500, np.random.normal(50000, 25000) + age * 3000))

    # A rough, believable price formula (in INR, scaled to lakhs for readability)
    base_price = 800000  # base new-car-ish price in INR
    brand_premium = {"Maruti": 0.9, "Hyundai": 1.0, "Honda": 1.1,
                      "Toyota": 1.3, "Tata": 0.95, "Ford": 0.9}[brand]
    price = base_price * brand_premium
    price *= (1 - 0.08) ** age                 # depreciation per year
    price -= km_driven * 0.4                    # wear from usage
    price *= 1.15 if transmission == "Automatic" else 1.0
    price *= 0.9 if owner == "Second Owner" else (0.75 if owner == "Third Owner" else 1.0)
    price *= 1.1 if fuel == "Diesel" else (0.95 if fuel == "CNG" else 1.0)
    price += np.random.normal(0, 30000)          # random market noise
    price = max(price, 60000)                    # floor price

    # Introduce some realistic missing values
    if np.random.rand() < 0.03:
        mileage_kmpl = np.nan
    if np.random.rand() < 0.02:
        engine_cc = np.nan

    rows.append({
        "brand": brand,
        "model": model,
        "year": year,
        "km_driven": km_driven,
        "fuel_type": fuel,
        "transmission": transmission,
        "owner": owner,
        "engine_cc": engine_cc,
        "mileage_kmpl": mileage_kmpl,
        "selling_price": round(price, -2),  # round to nearest 100
    })

df = pd.DataFrame(rows)
df.to_csv("used_cars.csv", index=False)
print(f"Synthetic dataset created: used_cars.csv  ({len(df)} rows)")
print(df.head())
