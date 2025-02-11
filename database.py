from influxdb_client import InfluxDBClient, Point, WritePrecision, WriteOptions
import time

class InfluxDBManager:
    def __init__(self):
        """
        Initializes the InfluxDB client for real-time ECU logging.
        """
        self.client = InfluxDBClient(url="http://localhost:8086", token="your-token", org="your-org")
        self.write_api = self.client.write_api(write_options=WriteOptions(batch_size=1))  # Synchronous writing
        self.bucket = "car_data"  # Ensure the correct bucket is used

    def write_data(self, data, predicted_fuel=None, timestamp=None):
        """
        Writes simulated car ECU & OBD-II data to InfluxDB.
        """

        # Default to current timestamp if not provided (milliseconds)
        timestamp = timestamp or int(time.time() * 1000)

        # Create an InfluxDB Point with additional ECU parameters
        point = (
            Point("vehicle_data")
            .tag("vehicle_id", "vehicle_001")
            .field("speed_kmh", data["speed_kmh"])
            .field("engine_rpm", data["engine_rpm"])
            .field("throttle_position", data["throttle_position"])
            .field("adjusted_throttle", data["adjusted_throttle"])  # ✅ Fixed
            .field("fuel_level", data["fuel_level"])
            .field("coolant_temp", data["coolant_temp"])
            .field("fuel_consumed", data["fuel_consumed"])
            .field("fuel_efficiency", data["fuel_efficiency"])
            .field("error_code", data.get("error_code", 0))  # ✅ Fixed
            .field("tire_pressure", data.get("tire_pressure", 32))
            .field("predicted_fuel", predicted_fuel if predicted_fuel is not None else 0)
            .time(timestamp, WritePrecision.MS)
        )

        try:
            # Write data to InfluxDB
            self.write_api.write(bucket=self.bucket, record=point)
            print(f"✅ Data written to InfluxDB: {point}")
        except Exception as e:
            print(f"❌ Error writing to InfluxDB: {e}")

    def close(self):
        """
        Closes the InfluxDB connection.
        """
        self.client.close()
