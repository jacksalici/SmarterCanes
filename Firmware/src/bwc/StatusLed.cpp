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

  if (fault_) return;  // solid fault indicator takes priority over the blink
  applyActiveVisual();
}

void StatusLed::setFault(bool fault) {
  if (fault == fault_) return;
  fault_ = fault;

  if (fault_) {
    ledOn_ = true;
    digitalWrite(pin_, HIGH);
  } else {
    // active_ itself didn't change while masked by the fault, so
    // setActive()'s no-op-on-same-value guard would skip restoring the
    // visual - apply it directly instead.
    applyActiveVisual();
  }
}

void StatusLed::applyActiveVisual() {
  if (!active_) {
    ledOn_ = false;
    digitalWrite(pin_, LOW);
    idlePhaseStartMs_ = millis();
  } else {
    ledOn_ = true;
    digitalWrite(pin_, HIGH);
    lastToggleMs_ = millis();
  }
}

void StatusLed::update() {
  if (fault_) return;  // solid: nothing to toggle

  const unsigned long now = millis();

  if (active_) {
    if (now - lastToggleMs_ < kBlinkIntervalMs) return;
    lastToggleMs_ = now;
    ledOn_ = !ledOn_;
    digitalWrite(pin_, ledOn_ ? HIGH : LOW);
    return;
  }

  // Idle: off, except for a brief heartbeat flash every kIdleFlashIntervalMs.
  if (ledOn_) {
    if (now - idlePhaseStartMs_ < kIdleFlashDurationMs) return;
    ledOn_ = false;
    digitalWrite(pin_, LOW);
    idlePhaseStartMs_ = now;
    return;
  }

  if (now - idlePhaseStartMs_ >= kIdleFlashIntervalMs) {
    ledOn_ = true;
    digitalWrite(pin_, HIGH);
    idlePhaseStartMs_ = now;
  }
}
