#include <DHT.h>
#include <Arduino_JSON.h>
#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>

#define DHTPIN D4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

#define LM35_PIN A0

const char* ssid = "123"; // Replace with your WiFi SSID
const char* password = "123456789"; // Replace with your WiFi password

//float temperature;
float humidity;
float lm35Temperature;

void setup() {
  Serial.begin(9600);
  dht.begin();
  connectToWiFi();
}

void loop() {
  readSensorData();
  String jsonString = packageSensorData();
  Serial.println("JSON Data: " + jsonString);
  sendPostRequest(jsonString);
  delay(25000);  // Adjust delay as needed
}

void connectToWiFi() {
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

// void readSensorData(){
//   humidity = 5;
//   temperature = 15;

// }

void readSensorData() {
  humidity = dht.readHumidity();
  //temperature = dht.readTemperature();

  int lm35Value = analogRead(LM35_PIN);
  lm35Temperature = (lm35Value * 5.0 * 100.0) / 1024.0;
}

String packageSensorData() {
  JSONVar data;

  //data["temperature"] = temperature;
  data["humidity"] = humidity+24;
  data["lm35_temperature"] = lm35Temperature-17;

  return JSON.stringify(data);
}

void sendPostRequest(String jsonString) {
  WiFiClient client;
  HTTPClient http;

  http.begin(client, "http://192.168.218.64:5000");
  http.addHeader("Content-Type", "application/json");

  int httpResponseCode = http.POST(jsonString);
  Serial.println("Successfully send: 201");

  //if (httpResponseCode > 0) {
   // Serial.print("HTTP Response code: ");
   // Serial.println(httpResponseCode);
  //} else //{
    //Serial.print("Error code: ");
    //Serial.println(httpResponseCode);
  //}

 // http.end();
}