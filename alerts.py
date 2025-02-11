class AlertSystem:
    def check_alerts(self, data, predicted_fuel, pid_error=None):
        if predicted_fuel < 5:  # Fuel is too low!
            self.send_alert(f"⚠️ Fuel critically low: {predicted_fuel:.2f} L!")

        # Check PID error (if exists) for abnormal throttle control
        if pid_error and abs(pid_error) > 10:  # Example threshold
            self.send_alert(f"⚠️ High PID error: {pid_error:.2f}, check throttle control.")

    def send_alert(self, message):
        print(f"[ALERT] {message}")  # Log the alert
