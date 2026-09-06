#include "ImuRecorder.h"

#include <string.h>

ImuRecorder::ImuRecorder(ISM330DLCSensor &imu, ModulinoDistance &distance)
    : imu_(imu), distance_(distance) {}

bool ImuRecorder::begin() {
  if (imu_.begin() != ISM330DLC_STATUS_OK) {
    Serial.println("[Recorder] IMU initialization failed");
    return false;
  }

  imu_.Enable_X();
  imu_.Enable_G();
  Serial.println("[Recorder] IMU initialized");

  // The VL53L4CD needs a moment after power-up before I2C reads are
  // reliable; retry once after a short settle delay if the first attempt
  // fails instead of giving up on a boot-time glitch.
  delay(50);
  distanceReady_ = distance_.begin();
  if (!distanceReady_) {
    delay(100);
    distanceReady_ = distance_.begin();
  }

  if (!distanceReady_) {
    Serial.println("[Recorder] WARNING: Distance sensor initialization failed");
  } else {
    Serial.println("[Recorder] Distance sensor initialized");
  }

  return true;
}

uint32_t ImuRecorder::nextSessionIndex() {
  uint32_t index = 0;

  File in = SD.open(kIndexPath, FILE_READ);
  if (in) {
    index = in.parseInt();
    in.close();
  }

  SD.remove(kIndexPath);
  File out = SD.open(kIndexPath, FILE_WRITE);
  if (out) {
    out.print(index + 1);
    out.close();
  }

  return index;
}

String ImuRecorder::segmentPath(uint32_t sessionIndex, uint32_t segmentIndex) {
  char buf[32];
  snprintf(buf, sizeof(buf), "/rec_%05lu_seg%03lu.csv", static_cast<unsigned long>(sessionIndex),
           static_cast<unsigned long>(segmentIndex));
  return String(buf);
}

bool ImuRecorder::openSegment() {
  lastOpenAttemptMs_ = millis();
  file_ = SD.open(segmentPath(sessionIndex_, segmentIndex_), FILE_WRITE);
  if (!file_) {
    Serial.print("[Recorder] Failed to open segment ");
    Serial.println(segmentIndex_);
    fileOpen_ = false;
    return false;
  }

  file_.println("t_ms,ax_mg,ay_mg,az_mg,gx_mdps,gy_mdps,gz_mdps,dist_mm,event");
  fileOpen_ = true;
  segmentStartMs_ = millis();
  return true;
}

void ImuRecorder::closeSegment() {
  if (!fileOpen_) return;
  flushBuffer();  // best-effort; we're closing regardless of the result
  file_.flush();
  file_.close();
  fileOpen_ = false;
}

bool ImuRecorder::flushBuffer() {
  if (bufferLen_ == 0) return true;

  bool ok = fileOpen_;
  if (ok) {
    file_.write(reinterpret_cast<const uint8_t *>(buffer_), bufferLen_);
    if (file_.getWriteError()) {
      file_.clearWriteError();
      ok = false;
    }
  }

  bufferLen_ = 0;
  return ok;
}

void ImuRecorder::appendToBuffer(const char *data, size_t len) {
  if (bufferLen_ + len > kBufferCapacity && !flushBuffer()) {
    Serial.println("[Recorder] WARNING: SD write error, rotating to a new segment");
    rotate();
  }
  memcpy(buffer_ + bufferLen_, data, len);
  bufferLen_ += len;
}

void ImuRecorder::rotate() {
  closeSegment();
  segmentIndex_++;
  openSegment();
}

void ImuRecorder::start() {
  if (recording_) return;

  sessionIndex_ = nextSessionIndex();
  segmentIndex_ = 0;
  bufferLen_ = 0;
  lastDistanceMm_ = NAN;
  pendingEvent_ = false;
  recordStartMs_ = lastSampleMs_ = lastBufferFlushMs_ = lastFlushMs_ = millis();
  recording_ = true;

  if (!openSegment()) {
    Serial.println("[Recorder] SD unavailable at start; will keep retrying");
  } else {
    Serial.print("[Recorder] Started session ");
    Serial.println(sessionIndex_);
  }
}

void ImuRecorder::stop(int8_t annotation) {
  if (!recording_) return;

  // Every index below this was, at some point, successfully opened and
  // closed - the current one only counts if it's still open (closeSegment()
  // is about to close it too).
  const uint32_t segmentCount = fileOpen_ ? segmentIndex_ + 1 : segmentIndex_;
  closeSegment();
  recording_ = false;

  for (uint32_t i = 0; i < segmentCount; i++) {
    const String from = segmentPath(sessionIndex_, i);
    const String to = from.substring(0, from.length() - 4) + "_ann" + String(annotation) + ".csv";
    SD.rename(from, to);
  }

  Serial.print("[Recorder] Stopped session ");
  Serial.print(sessionIndex_);
  Serial.print(" (");
  Serial.print(segmentCount);
  Serial.print(" segment(s)) with annotation ");
  Serial.println(annotation);
}

void ImuRecorder::annotateEvent() {
  if (!recording_) return;
  pendingEvent_ = true;
}

void ImuRecorder::poll() {
  if (!recording_) return;

  const unsigned long now = millis();

  if (!fileOpen_) {
    if (now - lastOpenAttemptMs_ < kReopenBackoffMs) return;
    if (!openSegment()) return;
  } else if (now - segmentStartMs_ >= kSegmentDurationMs) {
    rotate();
    if (!fileOpen_) return;  // reopen failed; the backoff branch above will retry
  }

  if (now - lastSampleMs_ < kSampleIntervalMs) return;
  lastSampleMs_ = now;

  int32_t acc[3];
  int32_t gyro[3];
  imu_.Get_X_Axes(acc);
  imu_.Get_G_Axes(gyro);

  if (distanceReady_ && distance_.available()) {
    lastDistanceMm_ = distance_.get();
  }

  const int event = pendingEvent_ ? 1 : 0;
  pendingEvent_ = false;

  char row[kMaxRowLen];
  int len;
  if (isnan(lastDistanceMm_)) {
    len = snprintf(row, sizeof(row), "%lu,%ld,%ld,%ld,%ld,%ld,%ld,-1,%d\n", now - recordStartMs_,
                   static_cast<long>(acc[0]), static_cast<long>(acc[1]), static_cast<long>(acc[2]),
                   static_cast<long>(gyro[0]), static_cast<long>(gyro[1]),
                   static_cast<long>(gyro[2]), event);
  } else {
    len = snprintf(row, sizeof(row), "%lu,%ld,%ld,%ld,%ld,%ld,%ld,%.1f,%d\n", now - recordStartMs_,
                   static_cast<long>(acc[0]), static_cast<long>(acc[1]), static_cast<long>(acc[2]),
                   static_cast<long>(gyro[0]), static_cast<long>(gyro[1]),
                   static_cast<long>(gyro[2]), lastDistanceMm_, event);
  }
  if (len > 0) appendToBuffer(row, static_cast<size_t>(len));

  if (now - lastBufferFlushMs_ >= kBufferFlushIntervalMs) {
    lastBufferFlushMs_ = now;
    if (!flushBuffer()) {
      Serial.println("[Recorder] WARNING: SD write error, rotating to a new segment");
      rotate();
    }
  }

  if (fileOpen_ && now - lastFlushMs_ >= kFlushIntervalMs) {
    file_.flush();
    lastFlushMs_ = now;
  }
}
