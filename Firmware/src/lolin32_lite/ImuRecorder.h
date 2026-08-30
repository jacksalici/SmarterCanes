#pragma once

#include <Arduino.h>
#include <SD.h>
#include <ISM330DLCSensor.h>

// Samples the IMU at a fixed rate into a CSV file on SD while recording.
// Each session gets its own auto-numbered file; on stop() the file is
// renamed to carry the session's annotation (-1, 0 or 1).
class ImuRecorder {
public:
  explicit ImuRecorder(ISM330DLCSensor &imu);

  bool begin();
  void start();
  void stop(int8_t annotation);
  void poll();

  bool isRecording() const { return recording_; }

private:
  static constexpr unsigned long kSampleIntervalMs = 10;
  static constexpr unsigned long kFlushIntervalMs = 1000;
  static constexpr const char *kIndexPath = "/rec_index.txt";

  ISM330DLCSensor &imu_;
  File file_;
  bool recording_ = false;
  unsigned long recordStartMs_ = 0;
  unsigned long lastSampleMs_ = 0;
  unsigned long lastFlushMs_ = 0;
  uint32_t sessionIndex_ = 0;

  static uint32_t nextSessionIndex();
  static String sessionPath(uint32_t index);
};
