#include "StatusLed.h"

StatusLed::StatusLed(uint8_t pin) : pin_(pin) {}

void StatusLed::begin() {
  pinMode(pin_, OUTPUT);
  digitalWrite(pin_, LOW);
}

void StatusLed::blinkBlocking(uint8_t times, unsigned long intervalMs) {
  for (uint8_t i = 0; i < times; i++) {
    digitalWrite(pin_, HIGH);
    delay(intervalMs);
    digitalWrite(pin_, LOW);
    delay(intervalMs);
  }
}

void StatusLed::setActive(bool active) {
  if (active == active_) return;
  active_ = active;

  if (!active_) {
    ledOn_ = false;
    digitalWrite(pin_, LOW);
  } else {
    ledOn_ = true;
    digitalWrite(pin_, HIGH);
    lastToggleMs_ = millis();
  }
}

void StatusLed::update() {
  if (!active_) return;

  const unsigned long now = millis();
  if (now - lastToggleMs_ < kBlinkIntervalMs) return;
  lastToggleMs_ = now;

  ledOn_ = !ledOn_;
  digitalWrite(pin_, ledOn_ ? HIGH : LOW);
}
