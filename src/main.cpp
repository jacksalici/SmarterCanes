#include <WiFiNINA.h>
#include <ArduinoHttpClient.h>
#include <Arduino_LSM6DS3.h>
#include <ArduinoJson.h>
#include "env.h"

WiFiClient wifiClient;
HttpClient httpClient = HttpClient(wifiClient, THINGSBOARD_HOST, THINGSBOARD_PORT); // use 443 for HTTPS (see below)


void connectToWiFi() {
  Serial.print("Connecting to WiFi");
  while (WiFi.begin(WIFI_SSID, WIFI_PASSWORD) != WL_CONNECTED) {
    Serial.print(".");
    delay(1000);
  }
  Serial.println(" connected.");
}

void sendIMUData() {
  JsonDocument doc;

  if (IMU.accelerationAvailable()) {


    float x, y, z;
    IMU.readAcceleration(x, y, z);

    doc["accel_x"] = x;
    doc["accel_y"] = y;
    doc["accel_z"] = z;
    String payload;
    serializeJson(doc, payload);

    Serial.println("Sending payload:");
    Serial.println(payload);

    String path = "/api/v1/" + String(THINGSBOARD_TOKEN) + "/telemetry";

    httpClient.beginRequest();
    httpClient.post(path);
    httpClient.sendHeader("Content-Type", "application/json");
    httpClient.sendHeader("Content-Length", payload.length());
    httpClient.endRequest();
    httpClient.print(payload);

    int statusCode = httpClient.responseStatusCode();
    Serial.print("Response code: ");
    Serial.println(statusCode);
  }
}

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Connect to WiFi
  connectToWiFi();

  // Initialize IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }
  Serial.println("IMU initialized.");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connectToWiFi();
  }

  sendIMUData();
  
}
