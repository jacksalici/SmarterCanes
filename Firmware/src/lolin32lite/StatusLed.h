#pragma once

#include <Arduino.h>

// Non-blocking status LED: a brief heartbeat flash every 30s while idle (so
// "off" is distinguishable from "unpowered"), blinks at a fixed rate while
// active (e.g. to indicate an ongoing recording). A fault (e.g. an SD write
// failure) overrides both with a solid-on light, distinguishable at a
// glance from either blink pattern, for as long as the fault persists. Call
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
  static constexpr unsigned long kIdleFlashIntervalMs = 30000;  // heartbeat while idle
  static constexpr unsigned long kIdleFlashDurationMs = 100;     // flash width

  void applyActiveVisual();

  uint8_t pin_;
  bool active_ = false;
  bool fault_ = false;
  bool ledOn_ = false;
  unsigned long lastToggleMs_ = 0;
  unsigned long idlePhaseStartMs_ = 0;  // when the current idle on/off phase began
};
