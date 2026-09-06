#pragma once

#include <Arduino.h>

// Non-blocking status LED: a brief heartbeat flash every 30s while idle (so
// "off" is distinguishable from "unpowered"), blinks once per second while
// active (e.g. to indicate an ongoing recording). A fault (e.g. an SD write
// failure) overrides both with a solid-on light, distinguishable at a
// glance from either blink pattern, for as long as the fault persists. A
// one-shot signal() pattern takes priority over the idle/active visual
// (but not a fault) so a caller can flash out how a recording ended. Call
// update() every loop iteration.
class StatusLed {
public:
  // One-shot feedback patterns for button gestures, played without serial
  // output: StopBlink (triple blink, same cadence as the power-on flash)
  // for either way of stopping a recording, EventBlink (a quicker double
  // blink) for a mid-recording event mark.
  enum class Signal : uint8_t { StopBlink, EventBlink };

  explicit StatusLed(uint8_t pin);

  void begin();
  void blinkBlocking(uint8_t times, unsigned long intervalMs);
  void setActive(bool active);
  void setFault(bool fault);
  // Queues a one-shot blink pattern; it plays out over subsequent update()
  // calls, preempting the idle/active visual until it finishes, then
  // control reverts to whatever setActive()/setFault() currently reflect.
  void signal(Signal pattern);
  void update();

private:
  static constexpr unsigned long kBlinkIntervalMs = 500;  // 1 blink/s while recording
  static constexpr unsigned long kIdleFlashIntervalMs = 30000;  // heartbeat while idle
  static constexpr unsigned long kIdleFlashDurationMs = 100;     // flash width
  static constexpr unsigned long kStopBlinkPulseMs = 100;   // matches blinkBlocking's power-on flash
  static constexpr unsigned long kEventBlinkPulseMs = 60;   // quicker, so it reads as distinct
  static constexpr uint8_t kMaxSignalSteps = 6;  // on,off x3 - the longest pattern (StopBlink)

  void applyActiveVisual();

  uint8_t pin_;
  bool active_ = false;
  bool fault_ = false;
  bool ledOn_ = false;
  unsigned long lastToggleMs_ = 0;
  unsigned long idlePhaseStartMs_ = 0;  // when the current idle on/off phase began

  bool signalActive_ = false;
  unsigned long signalSteps_[kMaxSignalSteps] = {0};
  uint8_t signalStepCount_ = 0;
  uint8_t signalStepIndex_ = 0;
  unsigned long signalStepStartMs_ = 0;
};
