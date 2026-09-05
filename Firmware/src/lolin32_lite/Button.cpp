#include "Button.h"

Button::Button(uint8_t pin) : pin_(pin) {}

void IRAM_ATTR Button::onChange(void *arg) {
  Button *self = static_cast<Button *>(arg);
  self->rawState_ = digitalRead(self->pin_) == LOW;
  self->lastEdgeMs_ = millis();
}

void Button::begin() {
  pinMode(pin_, INPUT_PULLUP);
  rawState_ = digitalRead(pin_) == LOW;
  lastEdgeMs_ = millis();
  attachInterruptArg(digitalPinToInterrupt(pin_), onChange, this, CHANGE);
}

ButtonEvent Button::update() {
  const unsigned long now = millis();

  noInterrupts();
  const bool reading = rawState_;
  const unsigned long edgeMs = lastEdgeMs_;
  interrupts();

  ButtonEvent event = ButtonEvent::None;

  if (now - edgeMs >= kDebounceMs && stableState_ != reading) {
    stableState_ = reading;

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
