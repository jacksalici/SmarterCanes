#pragma once

#include <Arduino.h>

enum class ButtonEvent : uint8_t { None, Click, DoubleClick, LongPress };

// Non-blocking gesture detection for a single active-low push button.
// Call update() every loop iteration; it returns at most one event per call.
class Button {
public:
  explicit Button(uint8_t pin);

  void begin();
  ButtonEvent update();

private:
  static constexpr unsigned long kDebounceMs = 30;
  static constexpr unsigned long kLongPressMs = 800;
  static constexpr unsigned long kDoubleClickMs = 350;

  uint8_t pin_;
  bool rawState_ = false;
  bool stableState_ = false;
  unsigned long lastEdgeMs_ = 0;
  unsigned long pressStartMs_ = 0;
  unsigned long lastClickMs_ = 0;
  bool longPressFired_ = false;
  bool clickPending_ = false;
};
