#include <ESP8266WiFi.h>
#include <DHT.h>

#define DHTPIN D4     // Pin connected to the DHT sensor
#define DHTTYPE DHT11 // DHT 11

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);
  dht.begin();
}

void loop() {
  delay(5000);  // Wait for 2 seconds

  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();
  temperature = temperature + 16;

  if (!isnan(humidity) && !isnan(temperature)) {
    Serial.print("Temp: ");
    Serial.print(temperature);
    Serial.print(" C, ");
    //Serial.print(temperature * 9 / 5 + 32); // Convert to Fahrenheit
    //Serial.print(" F, Hum: ");
    Serial.print(humidity);
    Serial.println("%");
  } else {
    Serial.println("Failed to read from DHT sensor!");
  }
}
