#include <ESP8266WiFi.h>
// Define LM35 pin
const int lm35_pin = A0; //Analog pin A0 on NodeMCU

void setup() {
  Serial.begin(9600);
}

void loop() {
  float temperature = readTemperature();
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");
  delay(5000); // Read temperature every 5 seconds
}

float readTemperature() {
  int adc_value = analogRead(lm35_pin); // Read ADC value
  float temperature = (adc_value * 3.3 / 1024) * 100; // Convert ADC value to temperature in Celsius
  return temperature-9;
}
