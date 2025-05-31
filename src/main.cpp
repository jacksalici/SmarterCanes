#include <Arduino.h>
#include <Arduino_LSM6DS3.h>

void setup() {
  Serial.begin(9600);
  while (!Serial);

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  Serial.println("IMU initialized successfully.");
}

void loop() {
  float x, y, z;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);

    Serial.print("X: ");
    Serial.print(x);
    Serial.print(" g, Y: ");
    Serial.print(y);
    Serial.print(" g, Z: ");
    Serial.print(z);
    Serial.println(" g");
  }

  delay(500); // Adjust delay for your needs
}
