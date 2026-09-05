#pragma once

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <SD.h>

#include "ImuRecorder.h"

// Minimal WiFi dashboard: lists recorded session files with per-file
// download links, a "download all" action that streams every session file
// as a single on-the-fly (uncompressed) .zip, and a two-step "delete all"
// action that wipes every session file and resets the session index.
// Refuses every request while
// recorder.isRecording() is true, both to avoid stalling the 10 ms IMU
// sample loop during a blocking request (e.g. a large file download) and to
// avoid touching the SD file handle the recorder still owns.
//
// WiFi/NTP/HTTP failure must never affect standalone recording: begin()
// returns false (and leaves the HTTP server un-started) if WiFi doesn't
// connect within the timeout; poll() becomes a cheap no-op in that case.
class Dashboard {
public:
  explicit Dashboard(ImuRecorder &recorder);

  // Connects to WiFi (bounded wait, kWifiConnectTimeoutMs), attempts a
  // best-effort NTP time sync (bounded wait, ~5s, non-fatal on failure so
  // future SD file timestamps just fall back to FatFs's default), and
  // starts the HTTP server. Returns true iff WiFi connected (dashboard is
  // reachable); does not start the server at all if WiFi failed.
  bool begin(const char *ssid, const char *password);

  // Call every loop() iteration. No-op if WiFi never connected. Otherwise
  // calls WebServer::handleClient(), which is cheap when idle but blocks
  // for the duration of an actual HTTP request (this is only acceptable
  // because every handler refuses to run while recording, so a slow
  // request can only ever happen when the 10 ms sample cadence doesn't
  // matter).
  void poll();

private:
  static constexpr unsigned long kWifiConnectTimeoutMs = 10000;
  static constexpr uint32_t kNtpSyncTimeoutMs = 5000;
  static constexpr uint16_t kHttpPort = 80;
  static constexpr const char *kTimeZone = "CET-1CEST,M3.5.0,M10.5.0/3"; // Europe/Rome

  ImuRecorder &recorder_;
  WebServer server_;
  bool wifiConnected_ = false;

  void setupTime();
  void setupRoutes();

  void handleRoot();
  void handleDownload();
  void handleDownloadAll();
  void handleClearConfirm();
  void handleClearAll();

  // Returns true (and sends a 503 "recording in progress" page) if a
  // handler must not proceed. Every handler calls this first.
  bool rejectIfRecording();

  static bool isSafeFilename(const String &name);
  static String formatTimestamp(time_t t);
};
