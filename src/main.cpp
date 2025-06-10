#include <WiFiNINA.h>
#include <ArduinoHttpClient.h>
#include <Arduino_LSM6DS3.h>
#include "env.h"

WiFiClient wifiClient;
HttpClient httpClient = HttpClient(wifiClient, THINGSBOARD_HOST, 8080); // use 443 for HTTPS (see below)

unsigned long lastSend = 0;

void connectToWiFi() {
  Serial.print("Connecting to WiFi");
  while (WiFi.begin(WIFI_SSID, WIFI_PASSWORD) != WL_CONNECTED) {
    Serial.print(".");
    delay(1000);
  }
  Serial.println(" connected.");
}

void sendIMUData() {
  float x, y, z;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);

    String payload = "{";
    payload += "\"accel_x\": " + String(x, 4) + ",";
    payload += "\"accel_y\": " + String(y, 4) + ",";
    payload += "\"accel_z\": " + String(z, 4);
    payload += "}";

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
    String response = httpClient.responseBody();

    Serial.print("Response code: ");
    Serial.println(statusCode);
    Serial.print("Response: ");
    Serial.println(response);
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

  if (millis() - lastSend > 10) {
    sendIMUData();
    lastSend = millis();
  }
}
