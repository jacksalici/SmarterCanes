#pragma once

#include <Arduino.h>

enum class ButtonEvent : uint8_t { None, Click, DoubleClick, LongPress };

// Non-blocking gesture detection for a single active-low push button.
// Edges are captured by a pin-change interrupt, so a press is timestamped
// the instant it happens instead of depending on how often update() is
// polled (which can otherwise miss a short press if loop() is briefly busy
// with SD/I2C work). Call update() every loop iteration; it returns at
// most one event per call.
class Button {
public:
  explicit Button(uint8_t pin);

  void begin();
  ButtonEvent update();

private:
  static constexpr unsigned long kDebounceMs = 30;
  static constexpr unsigned long kLongPressMs = 800;
  static constexpr unsigned long kDoubleClickMs = 350;

  static void IRAM_ATTR onChange(void *arg);

  uint8_t pin_;
  volatile bool rawState_ = false;
  volatile unsigned long lastEdgeMs_ = 0;

  bool stableState_ = false;
  unsigned long pressStartMs_ = 0;
  unsigned long lastClickMs_ = 0;
  bool longPressFired_ = false;
  bool clickPending_ = false;
};
