#include <WiFiNINA.h>
#include <Arduino_LSM6DS3.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "env.h"

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

unsigned long lastTelemetryTime = 0;
const unsigned long TELEMETRY_INTERVAL = 1000;

bool wifiConnected = false;
bool mqttConnected = false;

float accelX, accelY, accelZ;

String telemetryTopic = "v1/devices/me/telemetry";
String attributesTopic = "v1/devices/me/attributes";

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message received on topic: ");
  Serial.println(topic);
}

void readAndSendSensorData() {
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(accelX, accelY, accelZ);

    Serial.print("Acceleration - X: ");
    Serial.print(accelX);
    Serial.print(" g, Y: ");
    Serial.print(accelY);
    Serial.print(" g, Z: ");
    Serial.print(accelZ);
    Serial.println(" g");

    float magnitude = sqrt(accelX * accelX + accelY * accelY + accelZ * accelZ);

    StaticJsonDocument<200> telemetryDoc;
    telemetryDoc["accel_x"] = accelX;
    telemetryDoc["accel_y"] = accelY;
    telemetryDoc["accel_z"] = accelZ;
    telemetryDoc["accel_magnitude"] = magnitude;
    telemetryDoc["timestamp"] = millis();

    String telemetryPayload;
    serializeJson(telemetryDoc, telemetryPayload);

    if (mqttClient.publish(telemetryTopic.c_str(), telemetryPayload.c_str())) {
      Serial.println("Telemetry data sent to ThingsBoard");
      Serial.println("Payload: " + telemetryPayload);
    } else {
      Serial.println("Failed to send telemetry data");
    }
  } else {
    Serial.println("No new IMU data available");
  }
}

void sendDeviceAttributes() {
  StaticJsonDocument<150> attributesDoc;
  attributesDoc["device_type"] = "Arduino_LSM6DS3";
  attributesDoc["firmware_version"] = "1.0.0";
  attributesDoc["sample_rate"] = IMU.accelerationSampleRate();
  attributesDoc["wifi_rssi"] = WiFi.RSSI();

  String attributesPayload;
  serializeJson(attributesDoc, attributesPayload);

  if (mqttClient.publish(attributesTopic.c_str(), attributesPayload.c_str())) {
    Serial.println("Device attributes sent to ThingsBoard");
    Serial.println("Payload: " + attributesPayload);
  } else {
    Serial.println("Failed to send device attributes");
  }
}

void connectToWiFi() {
  Serial.print("Connecting to Wi-Fi network: ");
  Serial.println(WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    Serial.println();
    Serial.println("Wi-Fi connected successfully!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal strength (RSSI): ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
  } else {
    wifiConnected = false;
    Serial.println();
    Serial.println("Failed to connect to Wi-Fi!");
    delay(5000);
  }
}

void connectToMQTT() {
  if (!wifiConnected) {
    Serial.println("Cannot connect to MQTT: No Wi-Fi connection");
    return;
  }

  Serial.print("Connecting to ThingsBoard MQTT server: ");
  Serial.println(TB_SERVER);

  if (mqttClient.connect(TB_CLIENT_ID, TB_USERNAME, TB_PASSWORD)) {
    mqttConnected = true;
    Serial.println("Connected to ThingsBoard MQTT successfully!");
    sendDeviceAttributes();
  } else {
    mqttConnected = false;
    Serial.print("Failed to connect to MQTT! State: ");
    Serial.println(mqttClient.state());
    delay(5000);
  }
}

void setup() {
  Serial.begin(9600);
  while (!Serial) { ; }

  Serial.println("Starting Arduino LSM6DS3 to ThingsBoard MQTT client...");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.println("IMU initialized successfully");
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");

  mqttClient.setServer(TB_SERVER, TB_PORT);
  mqttClient.setCallback(mqttCallback);

  connectToWiFi();
  connectToMQTT();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    wifiConnected = false;
    Serial.println("Wi-Fi connection lost. Attempting to reconnect...");
    connectToWiFi();
  }

  if (!mqttClient.connected()) {
    mqttConnected = false;
    Serial.println("MQTT connection lost. Attempting to reconnect...");
    connectToMQTT();
  }

  mqttClient.loop();

  if (millis() - lastTelemetryTime > TELEMETRY_INTERVAL) {
    if (wifiConnected && mqttConnected) {
      readAndSendSensorData();
    }
    lastTelemetryTime = millis();
  }

  delay(100);
}
