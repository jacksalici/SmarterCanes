#include <Arduino.h>
#include <Wire.h>
#include <Modulino.h>

// Standalone bring-up test for the Modulino Distance sensor (VL53L4CD),
// independent of the main `bwc` firmware. Prints one distance reading per
// line so the Arduino IDE / PlatformIO Serial Plotter can graph it live -
// useful for checking wiring/mounting without recording a whole session.
//
// Build/upload: `pio run -e distance_test -t upload`
// Plot: PlatformIO Home > Serial Plotter (or Arduino IDE's), 115200 baud.

namespace pins {
// Same I2C bus as the main firmware (shared with the IMU there; this test
// doesn't touch the IMU at all).
constexpr uint8_t kSda = 26;
constexpr uint8_t kScl = 25;
}  // namespace pins

ModulinoDistance distance;

void setup() {
  Serial.begin(115200);
  delay(500);

  Wire.begin(pins::kSda, pins::kScl);
  Modulino.begin(Wire);

  if (!distance.begin()) {
    Serial.println("[DistanceTest] ERROR: sensor initialization failed");
    while (true) delay(1000);
  }
  Serial.println("[DistanceTest] Sensor ready - printing dist_mm");
}

void loop() {
  if (distance.available()) {
    Serial.println(distance.get(), 1);
  }
}
