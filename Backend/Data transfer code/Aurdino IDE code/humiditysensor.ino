#include <ESP8266WiFi.h>

const int soilMoistureSensorPin = A0; // Soil Moisture Sensor connected to A0

void setup() {
  Serial.begin(9600); // Start the Serial communication
  pinMode(soilMoistureSensorPin, INPUT); // Initialize the soil moisture sensor pin as an input
}

void loop() {
  int sensorValue = analogRead(soilMoistureSensorPin); // Read the soil moisture sensor value
  int moisturePercent = map(sensorValue, 0, 1023, 100, 0); // Map the sensor value to a percentage (assuming 0 = 100% and 1023 = 0%)

  // Print the moisture percentage
  Serial.print("Soil Moisture: ");
  Serial.print(moisturePercent);
  Serial.println("%");

  delay(2000); // Wait for 2 seconds before the next reading
}
