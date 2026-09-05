#include "ImuRecorder.h"

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

String ImuRecorder::sessionPath(uint32_t index) {
  char buf[24];
  snprintf(buf, sizeof(buf), "/rec_%05lu.csv", static_cast<unsigned long>(index));
  return String(buf);
}

void ImuRecorder::start() {
  if (recording_) return;

  sessionIndex_ = nextSessionIndex();
  file_ = SD.open(sessionPath(sessionIndex_), FILE_WRITE);
  if (!file_) {
    Serial.println("[Recorder] Failed to open file");
    return;
  }

  file_.println("t_ms,ax_mg,ay_mg,az_mg,gx_mdps,gy_mdps,gz_mdps,dist_mm");

  recording_ = true;
  lastDistanceMm_ = NAN;
  recordStartMs_ = lastSampleMs_ = lastFlushMs_ = millis();
  Serial.print("[Recorder] Started session ");
  Serial.println(sessionIndex_);
}

void ImuRecorder::stop(int8_t annotation) {
  if (!recording_) return;

  file_.close();
  recording_ = false;

  const String from = sessionPath(sessionIndex_);
  const String to = from.substring(0, from.length() - 4) + "_ann" + String(annotation) + ".csv";
  SD.rename(from, to);

  Serial.print("[Recorder] Stopped session ");
  Serial.print(sessionIndex_);
  Serial.print(" with annotation ");
  Serial.println(annotation);
}

void ImuRecorder::poll() {
  if (!recording_) return;

  const unsigned long now = millis();
  if (now - lastSampleMs_ < kSampleIntervalMs) return;
  lastSampleMs_ = now;

  int32_t acc[3];
  int32_t gyro[3];
  imu_.Get_X_Axes(acc);
  imu_.Get_G_Axes(gyro);

  if (distanceReady_ && distance_.available()) {
    lastDistanceMm_ = distance_.get();
  }

  file_.print(now - recordStartMs_);
  file_.print(',');
  file_.print(acc[0]);
  file_.print(',');
  file_.print(acc[1]);
  file_.print(',');
  file_.print(acc[2]);
  file_.print(',');
  file_.print(gyro[0]);
  file_.print(',');
  file_.print(gyro[1]);
  file_.print(',');
  file_.print(gyro[2]);
  file_.print(',');
  if (isnan(lastDistanceMm_)) {
    file_.println(-1);
  } else {
    file_.println(lastDistanceMm_, 1);
  }

  if (now - lastFlushMs_ >= kFlushIntervalMs) {
    file_.flush();
    lastFlushMs_ = now;
  }
}
