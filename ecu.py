import numpy as np
import random
from pid_controller import PIDController

class SoftwareECU:
    def __init__(self, fuel_capacity=50.0, target_fuel_efficiency=15.0):
        """
        Simulates an ECU with fuel management, OBD-II support, and PID control.
        """
        self.fuel_capacity = fuel_capacity
        self.fuel_level = fuel_capacity
        self.alert_triggered = False
        self.target_fuel_efficiency = target_fuel_efficiency

        # Tunable PID Controller
        self.pid = PIDController(kp=0.1, ki=0.01, kd=0.05)  # Default values

    def calculate_fuel_consumption(self, speed_kmh, throttle_position, engine_rpm):
        """
        Calculates fuel consumption dynamically.
        """
        base_consumption = (speed_kmh * throttle_position * 0.01) * (engine_rpm * 0.00005)
        return base_consumption * np.random.uniform(0.95, 1.05)

    def adjust_throttle(self, speed_kmh, fuel_efficiency):
        """
        Uses PID to adjust throttle for fuel efficiency.
        """
        error = self.target_fuel_efficiency - fuel_efficiency
        throttle_adjustment = self.pid.compute(error)
        return max(0, min(100, throttle_adjustment))

    def update_fuel_level(self, fuel_consumed):
        self.fuel_level = max(0, self.fuel_level - fuel_consumed)
        return self.fuel_level

    def check_fuel_alert(self):
        if self.fuel_level < (self.fuel_capacity * 0.1) and not self.alert_triggered:
            self.alert_triggered = True
            return "⚠️ Warning: Low Fuel Level!"
        return None

    def simulate_obd_data(self):
        """
        Simulates OBD-II responses.
        """
        return {
            "speed_kmh": random.randint(30, 120),
            "engine_rpm": random.randint(1000, 5000),
            "throttle_position": random.randint(10, 90),
            "coolant_temp": random.uniform(70, 100),
            "fuel_level": round(self.fuel_level, 2),
        }

    def simulate_ecu(self):
        """
        Runs the ECU logic including:
        - Fuel consumption
        - PID-based throttle adjustment
        - OBD-II data simulation
        """
        obd_data = self.simulate_obd_data()
        fuel_consumed = self.calculate_fuel_consumption(obd_data["speed_kmh"], obd_data["throttle_position"], obd_data["engine_rpm"])
        fuel_efficiency = obd_data["speed_kmh"] / (fuel_consumed + 1e-5)

        obd_data["adjusted_throttle"] = self.adjust_throttle(obd_data["speed_kmh"], fuel_efficiency)
        obd_data["fuel_consumed"] = round(fuel_consumed, 4)
        obd_data["fuel_efficiency"] = round(fuel_efficiency, 2)
        obd_data["alert"] = self.check_fuel_alert()

        self.update_fuel_level(fuel_consumed)

        return obd_data