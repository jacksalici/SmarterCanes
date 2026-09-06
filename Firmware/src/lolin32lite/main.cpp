#include <Arduino.h>
#include <SD.h>
#include <Wire.h>
#include <WiFi.h>
#include <ISM330DLCSensor.h>
#include <Modulino.h>

#include "Button.h"
#include "ImuRecorder.h"
#include "StatusLed.h"
#include "Dashboard.h"

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
Dashboard dashboard(recorder);

namespace {
bool trySdBegin() { return SD.begin(pins::kSdCs); }
bool tryRecorderBegin() { return recorder.begin(); }

// Retries a boot-critical init step forever instead of hanging silently, so
// a bad connection (SD card, IMU) that clears up - e.g. the card gets
// reseated - recovers without a power cycle. The LED goes solid while
// stuck, the same "fault" signal poll() uses for a mid-recording SD issue,
// so a boot failure is visible even with no serial monitor attached.
void waitFor(const char *label, bool (*attempt)()) {
  while (!attempt()) {
    Serial.print("[Main] ERROR: ");
    Serial.print(label);
    Serial.println(" failed, retrying...");
    statusLed.setFault(true);
    statusLed.update();
    delay(1000);
  }
  statusLed.setFault(false);
  statusLed.update();
}
}  // namespace

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
  waitFor("SD card initialization", trySdBegin);
  Serial.println("[Main] SD card ready");

  Serial.println("[Main] Initializing IMU...");
  waitFor("IMU initialization", tryRecorderBegin);
  Serial.println("[Main] IMU ready");

  Serial.println("[Main] Connecting to WiFi...");
  if (dashboard.begin()) {
    Serial.print("[Main] Dashboard ready at http://");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("[Main] WiFi unavailable - continuing without dashboard");
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
  dashboard.poll();

  statusLed.setActive(recorder.isRecording());
  statusLed.setFault(recorder.hasFault());
  statusLed.update();
}
