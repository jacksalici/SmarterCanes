#pragma once

#include <Arduino.h>
#include <SD.h>
#include <ISM330DLCSensor.h>
#include <Modulino.h>

// Samples the IMU and the Modulino Distance sensor at a fixed rate into CSV
// files on SD while recording.
//
// A session is split into fixed-length segment files (rotated every
// kSegmentDurationMs) rather than kept in one continuous file, so that an
// SD fault partway through a long recording only costs the current segment
// instead of silently losing everything recorded after it - each prior
// segment was already flushed and closed. Samples are batched in a RAM
// buffer and written in bulk instead of one SD write per sample, which
// bounds how much a single failed write can lose and cuts down on
// per-sample I/O latency. A failed write (or a failed file open) marks the
// current segment dead and retries into a fresh one on a backoff timer, so
// a transient fault (e.g. a jostled card connection) self-heals instead of
// silently stopping data collection for the rest of the session.
//
// On stop() every segment belonging to the session is renamed to carry the
// session's annotation (-1, 0 or 1).
class ImuRecorder {
public:
  ImuRecorder(ISM330DLCSensor &imu, ModulinoDistance &distance);

  bool begin();
  void start();
  void stop(int8_t annotation);
  // Marks the next sampled row with event=1 instead of interrupting the
  // recording, so a button press during a walk can flag a moment of
  // interest (e.g. a stumble) without stopping data collection.
  void annotateEvent();
  void poll();

  bool isRecording() const { return recording_; }
  // True while recording but the current segment isn't writable (SD write
  // or file-open failure); the caller can use this to flag a fault visually.
  bool hasFault() const { return recording_ && !fileOpen_; }

  static constexpr const char *kIndexPath = "/rec_index.txt";

private:
  static constexpr unsigned long kSampleIntervalMs = 10;
  static constexpr unsigned long kSegmentDurationMs = 30000;    // rotate to a new file every 30s
  static constexpr unsigned long kBufferFlushIntervalMs = 250;  // RAM buffer -> file
  static constexpr unsigned long kFlushIntervalMs = 1000;       // file -> physical SD card
  static constexpr unsigned long kReopenBackoffMs = 500;        // retry cadence after an SD fault
  static constexpr size_t kBufferCapacity = 2048;
  static constexpr size_t kMaxRowLen = 128;

  ISM330DLCSensor &imu_;
  ModulinoDistance &distance_;
  File file_;
  bool recording_ = false;
  bool distanceReady_ = false;
  bool fileOpen_ = false;
  unsigned long recordStartMs_ = 0;
  unsigned long lastSampleMs_ = 0;
  unsigned long lastBufferFlushMs_ = 0;
  unsigned long lastFlushMs_ = 0;
  unsigned long segmentStartMs_ = 0;
  unsigned long lastOpenAttemptMs_ = 0;
  uint32_t sessionIndex_ = 0;
  uint32_t segmentIndex_ = 0;
  float lastDistanceMm_ = NAN;
  bool pendingEvent_ = false;

  char buffer_[kBufferCapacity];
  size_t bufferLen_ = 0;

  static uint32_t nextSessionIndex();
  static String segmentPath(uint32_t sessionIndex, uint32_t segmentIndex);

  bool openSegment();
  void closeSegment();
  bool flushBuffer();  // returns false if the pending bytes failed to write
  void appendToBuffer(const char *data, size_t len);
  void rotate();  // close (if open), advance to the next segment index, reopen
};
