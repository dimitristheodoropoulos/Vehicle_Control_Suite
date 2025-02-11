import numpy as np
from typing import Dict

class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, target: float, current: float, dt: float) -> float:
        error = target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

class VehicleSimulator:
    """
    Simulates vehicle data and uses a PID controller to regulate throttle position.
    """

    def __init__(self, initial_speed: float = 0, target_speed: float = 60, dt: float = 0.1, pid: PIDController = None):
        self.speed = initial_speed
        self.target_speed = target_speed
        self.dt = dt
        self.time_elapsed = 0
        self.throttle_position = 0
        self.pid = pid or PIDController(Kp=1.0, Ki=0.1, Kd=0.01)

    def update(self) -> Dict[str, float]:
        """
        Simulate one time step of vehicle data.

        :return: Dictionary containing simulated vehicle data.
        """
        self.throttle_position = self.pid.update(self.target_speed, self.speed, self.dt)

        # Simple dynamics model
        self.speed += self.throttle_position * self.dt - 0.1 * self.speed * self.dt
        self.speed += np.random.normal(0, 0.1)  # Adding noise for realism

        engine_rpm = 3000 + 10 * self.speed
        coolant_temp = 90 + 0.1 * self.speed
        fuel_level = 100 - 0.1 * self.time_elapsed

        self.time_elapsed += self.dt

        return {
            "time": self.time_elapsed,
            "speed_kmh": self.speed,
            "throttle_position": self.throttle_position,
            "target_speed": self.target_speed,
            "engine_rpm": engine_rpm,
            "coolant_temp": coolant_temp,
            "fuel_level": fuel_level
        }

    def set_pid_gains(self, Kp: float, Ki: float, Kd: float):
        self.pid.Kp = Kp
        self.pid.Ki = Ki
        self.pid.Kd = Kd
