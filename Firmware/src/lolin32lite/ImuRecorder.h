#pragma once

#include <Arduino.h>
#include <SD.h>
#include <ISM330DLCSensor.h>
#include <Modulino.h>

// Samples the IMU and the Modulino Distance sensor at a fixed rate into a
// CSV file on SD while recording. Each session gets its own auto-numbered
// file; on stop() the file is renamed to carry the session's annotation
// (-1, 0 or 1).
class ImuRecorder {
public:
  ImuRecorder(ISM330DLCSensor &imu, ModulinoDistance &distance);

  bool begin();
  void start();
  void stop(int8_t annotation);
  void poll();

  bool isRecording() const { return recording_; }

  static constexpr const char *kIndexPath = "/rec_index.txt";

private:
  static constexpr unsigned long kSampleIntervalMs = 10;
  static constexpr unsigned long kFlushIntervalMs = 1000;

  ISM330DLCSensor &imu_;
  ModulinoDistance &distance_;
  File file_;
  bool recording_ = false;
  bool distanceReady_ = false;
  unsigned long recordStartMs_ = 0;
  unsigned long lastSampleMs_ = 0;
  unsigned long lastFlushMs_ = 0;
  uint32_t sessionIndex_ = 0;
  float lastDistanceMm_ = NAN;

  static uint32_t nextSessionIndex();
  static String sessionPath(uint32_t index);
};
