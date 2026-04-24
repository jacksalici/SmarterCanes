#include <SPI.h>
#include <ISM330DLCSensor.h>

constexpr uint8_t CS_PIN = 10;

// UNO R4 exposes the hardware SPI bus as a global instance.
auto &dev_spi = SPI;

// Create SPI sensor object (CS pin 10)
ISM330DLCSensor AccGyr(&dev_spi, CS_PIN);

void setup() {
  Serial.begin(115200);
  delay(1000);

  dev_spi.begin();

  if (AccGyr.begin() != 0) {
    Serial.println("Sensor init failed!");
    while (1);
  }

  AccGyr.Enable_X(); // accelerometer
  AccGyr.Enable_G(); // gyroscope

  Serial.println("ISM330DLC (SPI) ready");
}

void loop() {
  int32_t acc[3];
  int32_t gyro[3];

  AccGyr.Get_X_Axes(acc);
  AccGyr.Get_G_Axes(gyro);

  Serial.print("ACC [mg]: ");
  Serial.print(acc[0]); Serial.print(" ");
  Serial.print(acc[1]); Serial.print(" ");
  Serial.print(acc[2]); Serial.print("   ");

  Serial.print("GYRO [mdps]: ");
  Serial.print(gyro[0]); Serial.print(" ");
  Serial.print(gyro[1]); Serial.print(" ");
  Serial.print(gyro[2]);

  Serial.println();

  delay(200);
}