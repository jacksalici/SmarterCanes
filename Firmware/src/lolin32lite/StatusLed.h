#pragma once

#include <Arduino.h>

// Non-blocking status LED: off when idle, blinks at a fixed rate while
// active (e.g. to indicate an ongoing recording). A fault (e.g. an SD write
// failure) overrides that with a solid-on light, distinguishable at a
// glance from the normal blink, for as long as the fault persists. Call
// update() every loop iteration.
class StatusLed {
public:
  explicit StatusLed(uint8_t pin);

  void begin();
  void blinkBlocking(uint8_t times, unsigned long intervalMs);
  void setActive(bool active);
  void setFault(bool fault);
  void update();

private:
  static constexpr unsigned long kBlinkIntervalMs = 300;

  void applyActiveVisual();

  uint8_t pin_;
  bool active_ = false;
  bool fault_ = false;
  bool ledOn_ = false;
  unsigned long lastToggleMs_ = 0;
};
