import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from data_generator import VehicleSimulator
import joblib

# Generate synthetic training data
simulator = VehicleSimulator()
data = [simulator.update() for _ in range(5000)]  # Create 5000 samples
df = pd.DataFrame(data)

# Select features & target variable
X = df[['speed_kmh', 'engine_rpm', 'throttle_position', 'coolant_temp']].values
y = df['fuel_level'].values  # Predict fuel consumption

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model
joblib.dump(rf_model, "random_forest_model.pkl")
print("Random Forest model trained and saved as 'random_forest_model.pkl'")

# Evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse}, R2: {r2}")
