#include "Button.h"

Button::Button(uint8_t pin) : pin_(pin) {}

void Button::begin() {
  pinMode(pin_, INPUT_PULLUP);
}

ButtonEvent Button::update() {
  const bool reading = digitalRead(pin_) == LOW;
  const unsigned long now = millis();

  if (reading != rawState_) {
    rawState_ = reading;
    lastEdgeMs_ = now;
  }

  ButtonEvent event = ButtonEvent::None;

  if (now - lastEdgeMs_ >= kDebounceMs && stableState_ != rawState_) {
    stableState_ = rawState_;

    if (stableState_) {
      pressStartMs_ = now;
      longPressFired_ = false;
    } else if (!longPressFired_) {
      if (clickPending_ && now - lastClickMs_ <= kDoubleClickMs) {
        clickPending_ = false;
        event = ButtonEvent::DoubleClick;
        Serial.println("[Button] Double click detected");
      } else {
        clickPending_ = true;
        lastClickMs_ = now;
      }
    }
  }

  if (stableState_ && !longPressFired_ && now - pressStartMs_ >= kLongPressMs) {
    longPressFired_ = true;
    clickPending_ = false;
    event = ButtonEvent::LongPress;
    Serial.println("[Button] Long press detected");
  }

  if (clickPending_ && now - lastClickMs_ > kDoubleClickMs) {
    clickPending_ = false;
    event = ButtonEvent::Click;
    Serial.println("[Button] Click detected");
  }

  return event;
}
