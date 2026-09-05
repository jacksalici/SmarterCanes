#pragma once

#include <Arduino.h>

// Non-blocking status LED: off when idle, blinks at a fixed rate while
// active (e.g. to indicate an ongoing recording). Call update() every loop
// iteration.
class StatusLed {
public:
  explicit StatusLed(uint8_t pin);

  void begin();
  void blinkBlocking(uint8_t times, unsigned long intervalMs);
  void setActive(bool active);
  void update();

private:
  static constexpr unsigned long kBlinkIntervalMs = 300;

  uint8_t pin_;
  bool active_ = false;
  bool ledOn_ = false;
  unsigned long lastToggleMs_ = 0;
};
