#include <Arduino.h>
#include <SD.h>
#include <Wire.h>
#include <ISM330DLCSensor.h>
#include <Modulino.h>

#include "Button.h"
#include "ImuRecorder.h"
#include "StatusLed.h"

namespace pins {
constexpr uint8_t kSdCs = 5;
constexpr uint8_t kImuSda = 26;
constexpr uint8_t kImuScl = 25;
constexpr uint8_t kButton = 12;
constexpr uint8_t kLed = 32;
}

ISM330DLCSensor imu(&Wire, ISM330DLC_ACC_GYRO_I2C_ADDRESS_LOW);
ModulinoDistance distance;
Button button(pins::kButton);
StatusLed statusLed(pins::kLed);
ImuRecorder recorder(imu, distance);

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n[Main] Lolin32 Lite IMU Recorder starting...");

  Serial.println("[Main] Initializing I2C...");
  Wire.begin(pins::kImuSda, pins::kImuScl);
  Modulino.begin(Wire);

  Serial.println("[Main] Initializing button...");
  button.begin();

  Serial.println("[Main] Initializing status LED...");
  statusLed.begin();
  statusLed.blinkBlocking(3, 100);

  Serial.println("[Main] Initializing SD card...");
  if (!SD.begin(pins::kSdCs)) {
    Serial.println("[Main] ERROR: SD card initialization failed");
    while (true) delay(1000);
  }
  Serial.println("[Main] SD card ready");

  Serial.println("[Main] Initializing IMU...");
  if (!recorder.begin()) {
    Serial.println("[Main] ERROR: IMU initialization failed");
    while (true) delay(1000);
  }

  Serial.println("[Main] ========================================");
  Serial.println("[Main] Ready - waiting for button input");
  Serial.println("[Main] Click: toggle recording");
  Serial.println("[Main] Double-click: stop with annotation 0");
  Serial.println("[Main] Long-press: stop with annotation 1");
  Serial.println("[Main] ========================================");
}

void loop() {
  switch (button.update()) {
    case ButtonEvent::Click:
      if (recorder.isRecording()) {
        recorder.stop(-1);
      } else {
        recorder.start();
      }
      break;

    case ButtonEvent::DoubleClick:
      if (recorder.isRecording()) {
        recorder.stop(0);
      }
      break;

    case ButtonEvent::LongPress:
      if (recorder.isRecording()) {
        recorder.stop(1);
      }
      break;

    default:
      break;
  }

  recorder.poll();

  statusLed.setActive(recorder.isRecording());
  statusLed.update();
}
