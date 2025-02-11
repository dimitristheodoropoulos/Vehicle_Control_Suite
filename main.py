from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from model import hybrid_predict
from data_generator import VehicleSimulator
from database import InfluxDBManager
from alerts import AlertSystem
from pid_controller import PIDController
import numpy as np

# Initialize components
app = FastAPI()

simulator = VehicleSimulator()
db_manager = InfluxDBManager()
alert_system = AlertSystem()

# Pydantic model for vehicle data input
class VehicleData(BaseModel):
    speed_kmh: float
    engine_rpm: int
    throttle_position: int
    coolant_temp: float

# Pydantic model for PID gains
class PIDGains(BaseModel):
    Kp: float
    Ki: float
    Kd: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fuel Prediction API!"}

@app.post("/set_pid_gains")
def set_pid_gains(gains: PIDGains):
    try:
        simulator.pid = PIDController(Kp=gains.Kp, Ki=gains.Ki, Kd=gains.Kd)
        return {"message": "PID gains updated", "Kp": gains.Kp, "Ki": gains.Ki, "Kd": gains.Kd}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error setting PID gains: {str(e)}")

@app.get("/simulate")
def simulate(num_steps: int = 100):
    try:
        data = simulator.generate_data(num_steps)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in simulation: {str(e)}")

@app.post("/predict_fuel/")
async def predict_fuel(data: VehicleData):
    try:
        # Convert input data to numpy array
        input_data = np.array([
            [data.speed_kmh, data.engine_rpm, data.throttle_position, data.coolant_temp]
        ], dtype=np.float32)

        # Convert numpy array to tensor
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        # Log input tensor
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Input tensor type: {input_tensor.dtype}")

        # Ensure input_tensor is on the same device as the model (if using GPU)
        input_tensor = input_tensor.to(torch.float32)

        # Get the refined prediction using the hybrid approach
        predicted_fuel = hybrid_predict(input_tensor)

        # If predicted_fuel is a tensor, extract the value as float
        if isinstance(predicted_fuel, torch.Tensor):
            predicted_fuel = predicted_fuel.item()  # Extract scalar from tensor

        return {"predicted_fuel": predicted_fuel}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/simulate_vehicle/")
async def simulate_vehicle():
    try:
        # Generate real-time vehicle data
        data = simulator.update()
        print(f"🚗 Data: {data}")

        # Convert data to tensor for fuel prediction
        input_data = torch.tensor([
            [data['speed_kmh'], data['engine_rpm'], data['throttle_position'], data['coolant_temp']]
        ], dtype=torch.float32)

        # Get the refined prediction using the hybrid approach
        predicted_fuel = hybrid_predict(input_data)
        print(f"🔮 Predicted Fuel Level: {predicted_fuel:.2f} L")

        # Insert data into InfluxDB
        db_manager.write_data(data, predicted_fuel)

        # Check for alerts (e.g., low fuel)
        alert_system.check_alerts(data, predicted_fuel)

        return {"simulated_data": data, "predicted_fuel": predicted_fuel}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in simulation: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Ensure the model is loaded at startup."""
    try:
        # You might want to load the neural network model here
        print("Models loaded successfully at startup.")
    except Exception as e:
        print(f"Error loading models at startup: {str(e)}")
        exit(1)

@app.on_event("shutdown")
def shutdown_event():
    """Close any open resources on shutdown."""
    print("Shutting down the FastAPI app.")
