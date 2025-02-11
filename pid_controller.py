import numpy as np

class PIDController:
    """
    A simple PID controller for throttle position regulation.
    """

    def __init__(self, Kp: float, Ki: float, Kd: float, output_limits: tuple = (0, 100)):
        """
        Initialize the PID controller with gains and output limits.

        :param Kp: Proportional gain.
        :param Ki: Integral gain.
        :param Kd: Derivative gain.
        :param output_limits: Tuple of (min, max) values to clamp the control output.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        self.previous_error = 0
        self.integral = 0

    def update(self, setpoint: float, measured_value: float, dt: float) -> float:
        """
        Update the PID controller and compute the control output.

        :param setpoint: Desired value (e.g., desired speed).
        :param measured_value: Actual value (e.g., current speed).
        :param dt: Time step (delta time).
        :return: Control output (e.g., throttle position), clamped within output limits.
        """
        error = setpoint - measured_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error

        # Clamp the output to the defined limits (e.g., throttle position between 0 and 100)
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        return output

    def reset(self):
        """
        Reset the PID controller's integral and previous error.
        """
        self.integral = 0
        self.previous_error = 0

    def set_gains(self, Kp: float, Ki: float, Kd: float):
        """
        Update the PID controller's gains dynamically.

        :param Kp: New proportional gain.
        :param Ki: New integral gain.
        :param Kd: New derivative gain.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd