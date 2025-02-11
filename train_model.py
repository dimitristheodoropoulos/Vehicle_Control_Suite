import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from data_generator import VehicleSimulator  

# Regenerate synthetic data if the CSV file is missing
try:
    df = pd.read_csv('vehicle_data.csv')
except FileNotFoundError:
    print("vehicle_data.csv not found. Regenerating data...")
    simulator = VehicleSimulator()
    data = [simulator.update() for _ in range(5000)]  # Generate 5000 data points
    df = pd.DataFrame(data)
    df.to_csv('vehicle_data.csv', index=False)
    print("Data has been generated and saved to 'vehicle_data.csv'")

# Features and target
X = df[['speed_kmh', 'engine_rpm', 'throttle_position', 'coolant_temp']].values
y = df['fuel_level'].values

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained Random Forest model
joblib.dump(rf_model, "random_forest_model.pkl")
print("Random Forest model trained and saved as 'random_forest_model.pkl'")

# Evaluate the model on the validation set
y_val_pred = rf_model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)
print(f"Validation MSE: {val_mse}, R2: {val_r2}")
