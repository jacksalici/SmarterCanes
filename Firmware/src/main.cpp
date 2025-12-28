#include <Arduino.h>
#include <ArduinoBLE.h>
#include <Arduino_BHY2.h>

SensorXYZ gyroscope(SENSOR_ID_GYRO);
SensorXYZ accelerometer(SENSOR_ID_ACC);
Sensor temperature(SENSOR_ID_TEMP);
Sensor humidity(SENSOR_ID_HUM);
Sensor pressure(SENSOR_ID_BARO);
Sensor gas(SENSOR_ID_GAS);

BLEService sensorService("eb7f25c3-8d96-4311-92c9-45e90f6b6f5b");
BLEStringCharacteristic sensorGyroData("a94090de-f49a-49f4-97c0-a95abc6cbb95", BLERead | BLENotify, 50);
BLEStringCharacteristic sensorAccelData("ca52c70a-3eb6-4043-add4-df23393e387f", BLERead | BLENotify, 50);
BLEStringCharacteristic sensorEnvData("d3e4f5a6-7b8c-9d0e-1f2a-3b4c5d6e7f8a", BLERead | BLENotify, 100);

float maxVel = 150;            // Maximum expected rotation value
float movementThreshold = 0.8; // Threshold to consider as movement

long sendInterval = 50;

boolean bleInitialized = false; // BLE initialization state
boolean isConnected = false;    // Track connection state

void onBLEConnected(BLEDevice central)
{
  Serial.print("Connected to central: ");
  Serial.println(central.address());
  isConnected = true;
}

void onBLEDisconnected(BLEDevice central)
{
  Serial.print("Disconnected from central: ");
  Serial.println(central.address());
  isConnected = false;

  BLE.advertise();
  Serial.println("Restarted advertising after disconnect");
}

int state = 0; // state machine for walking animation

boolean initBLE()
{
  if (!BLE.begin())
  {
    Serial.println("Starting BLE failed!");
    return false;
  }

  BLE.setLocalName("Better Walking Cane");
  BLE.setDeviceName("Better Walking Cane");

  BLE.setConnectionInterval(0x0006, 0x0C80); // 7.5ms to 4s
  BLE.setSupervisionTimeout(0x0C80);         // 20 seconds

  BLE.setAdvertisedService(sensorService);
  sensorService.addCharacteristic(sensorGyroData);
  sensorService.addCharacteristic(sensorAccelData);
  sensorService.addCharacteristic(sensorEnvData);

  BLE.addService(sensorService);

  BLE.setEventHandler(BLEConnected, onBLEConnected);
  BLE.setEventHandler(BLEDisconnected, onBLEDisconnected);
  BLE.setPairable(1);

  BLE.advertise();
  Serial.println("BLE device is now advertising...");
  return true;
}

void sendBLE(String data, BLEStringCharacteristic& characteristic)
{
  if (isConnected && BLE.connected())
  {
    characteristic.writeValue(data);
  }
}

void setup()
{
  Serial.begin(115200);

  // Wait a moment for serial to initialize
  delay(1000);

  // Initialize BHY2 sensor system
  BHY2.begin();
  gyroscope.begin();
  accelerometer.begin();
  temperature.begin();
  humidity.begin();
  pressure.begin();
  gas.begin();

  gyroscope.setRange(2000);    // Set gyro range to +/-2000 dps
  accelerometer.setRange(8);   // Set accelerometer range to +/-8g

  bleInitialized = initBLE(); // Initialize BLE

  if (bleInitialized)
  {
    Serial.println("BLE initialized successfully");
  }
  else
  {
    Serial.println("BLE initialization failed");
  }

  Serial.println("Setup complete!");
}

void loop()
{
  BLE.poll();
  BHY2.update();

  // Check and send accelerometer data when available
  if (accelerometer.dataAvailable())
  {
    float accelX = accelerometer.x();
    float accelY = accelerometer.y();
    float accelZ = accelerometer.z();
    String data = String(accelX, 3) + "," + String(accelY, 3) + "," + String(accelZ, 3);
    sendBLE(data, sensorAccelData);
    accelerometer.clearDataAvailFlag();
  }

  // Check and send gyroscope data when available
  if (gyroscope.dataAvailable())
  {
    float gyroX = gyroscope.x();
    float gyroY = gyroscope.y();
    float gyroZ = gyroscope.z();
    
    String data = String(gyroX, 3) + "," + String(gyroY, 3) + "," + String(gyroZ, 3);
    sendBLE(data, sensorGyroData);
    gyroscope.clearDataAvailFlag();
  }

  // Check and send environmental data when available
  static auto lastEnvSendTime = millis();
  if (millis() - lastEnvSendTime >= sendInterval)
  {
    lastEnvSendTime = millis();
    
    float temp = 0, hum = 0, press = 0, gasVal = 0;
    
    if (temperature.dataAvailable())
    {
      temp = temperature.value();
      temperature.clearDataAvailFlag();
    }
    
    if (humidity.dataAvailable())
    {
      hum = humidity.value();
      humidity.clearDataAvailFlag();
    }
    
    if (pressure.dataAvailable())
    {
      press = pressure.value();
      pressure.clearDataAvailFlag();
    }
    
    if (gas.dataAvailable())
    {
      gasVal = gas.value();
      gas.clearDataAvailFlag();
    }
    
    String data = String(temp, 2) + "," + String(hum, 2) + "," + String(press, 2) + "," + String(gasVal, 2);
    sendBLE(data, sensorEnvData);
  }

  BLE.poll();
  delay(50);
}