#include "Dashboard.h"

#include <time.h>
#include <vector>
#include <esp_rom_crc.h>

namespace {
const char *wifiStatusToString(wl_status_t status) {
  switch (status) {
    case WL_IDLE_STATUS:     return "idle (never started)";
    case WL_NO_SSID_AVAIL:   return "SSID not found (check name/typo, or AP is 5GHz-only - ESP32 is 2.4GHz-only)";
    case WL_SCAN_COMPLETED:  return "scan completed, still associating";
    case WL_CONNECTED:       return "connected";
    case WL_CONNECT_FAILED:  return "connect failed (likely wrong password)";
    case WL_CONNECTION_LOST: return "connection lost";
    case WL_DISCONNECTED:    return "disconnected";
    default:                 return "unknown";
  }
}
}  // namespace

Dashboard::Dashboard(ImuRecorder &recorder) : recorder_(recorder), server_(kHttpPort) {}

bool Dashboard::begin(const char *ssid, const char *password) {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  const unsigned long deadline = millis() + kWifiConnectTimeoutMs;
  while (WiFi.status() != WL_CONNECTED && millis() < deadline) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();

  wifiConnected_ = (WiFi.status() == WL_CONNECTED);
  if (!wifiConnected_) {
    Serial.print("[Dashboard] WiFi connect failed, status: ");
    Serial.println(wifiStatusToString(WiFi.status()));
    return false;
  }

  setupTime();
  setupRoutes();
  server_.begin();
  return true;
}

void Dashboard::setupTime() {
  configTzTime(kTimeZone, "pool.ntp.org", "time.nist.gov");

  struct tm ti;
  if (getLocalTime(&ti, kNtpSyncTimeoutMs)) {
    Serial.println("[Dashboard] Time synced");
  } else {
    Serial.println("[Dashboard] NTP sync failed; SD timestamps may be inaccurate");
  }
}

void Dashboard::setupRoutes() {
  server_.on("/", HTTP_GET, [this] { handleRoot(); });
  server_.on("/download", HTTP_GET, [this] { handleDownload(); });
  server_.on("/downloadAll", HTTP_GET, [this] { handleDownloadAll(); });
  server_.on("/clear", HTTP_GET, [this] { handleClearConfirm(); });
  server_.on("/clear", HTTP_POST, [this] { handleClearAll(); });
  server_.onNotFound([this] { server_.send(404, "text/plain", "Not found"); });
}

void Dashboard::poll() {
  if (!wifiConnected_) return;
  server_.handleClient();
}

bool Dashboard::rejectIfRecording() {
  if (!recorder_.isRecording()) return false;

  server_.send(503, "text/html",
               "<html><body><h1>Recording in progress</h1>"
               "<p>The dashboard is unavailable while a session is recording.</p>"
               "</body></html>");
  return true;
}

namespace {
String indexFileName() {
  String name(ImuRecorder::kIndexPath);
  if (name.startsWith("/")) name.remove(0, 1);
  return name;
}

// Every non-directory file at the SD root except the session-index
// bookkeeping file.
std::vector<String> listSessionFiles(const String &indexName) {
  std::vector<String> names;
  File root = SD.open("/");
  if (root) {
    File entry = root.openNextFile();
    while (entry) {
      const String name(entry.name());
      if (!entry.isDirectory() && name != indexName) {
        names.push_back(name);
      }
      entry.close();
      entry = root.openNextFile();
    }
    root.close();
  }
  return names;
}

void writeLE16(uint8_t *p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
}

void writeLE32(uint8_t *p, uint32_t v) {
  p[0] = static_cast<uint8_t>(v);
  p[1] = static_cast<uint8_t>(v >> 8);
  p[2] = static_cast<uint8_t>(v >> 16);
  p[3] = static_cast<uint8_t>(v >> 24);
}

// DOS date/time as used by the ZIP format. Falls back to 1980-01-01/00:00
// (the format's epoch) if the file has no valid timestamp (e.g. NTP never
// synced), since day/month can't be zero in a DOS date.
void toDosDateTime(time_t t, uint16_t &dosDate, uint16_t &dosTime) {
  struct tm tmv;
  if (t <= 0) {
    dosDate = (1 << 5) | 1;  // month=1, day=1, year offset 0 (1980)
    dosTime = 0;
    return;
  }
  localtime_r(&t, &tmv);
  int year = tmv.tm_year + 1900;
  if (year < 1980) year = 1980;
  dosDate = static_cast<uint16_t>(((year - 1980) << 9) | ((tmv.tm_mon + 1) << 5) | tmv.tm_mday);
  dosTime = static_cast<uint16_t>((tmv.tm_hour << 11) | (tmv.tm_min << 5) | (tmv.tm_sec / 2));
}

// esp_rom_crc32_le() is a raw CRC32 register update with no built-in
// init/final handling (see esp_rom_crc.h) - the PKZIP/CRC-32 standard used
// by the ZIP format needs init=0xFFFFFFFF and a final bitwise-NOT, so the
// running value is seeded with ~0xFFFFFFFF (i.e. 0) and finalized with a
// bitwise-NOT once all of a file's bytes have been folded in.
uint32_t crc32Update(uint32_t crc, const uint8_t *buf, size_t len) {
  return esp_rom_crc32_le(crc, buf, len);
}

uint32_t crc32Finish(uint32_t crc) { return ~crc; }
}  // namespace

void Dashboard::handleRoot() {
  if (rejectIfRecording()) return;

  const String indexName = indexFileName();

  String html;
  html.reserve(1024);
  html += "<html><head><title>SmartCane Dashboard</title></head><body>";
  html += "<h1>Recordings</h1>";
  html += "<table border='1' cellpadding='6'><tr><th>File</th><th>Size</th><th>Date</th><th></th></tr>";

  File root = SD.open("/");
  if (root) {
    File entry = root.openNextFile();
    while (entry) {
      const String name(entry.name());
      if (!entry.isDirectory() && name != indexName) {
        html += "<tr><td>" + name + "</td><td>" + String(entry.size()) + "</td><td>" +
                formatTimestamp(entry.getLastWrite()) + "</td><td><a href=\"/download?file=" +
                name + "\">Download</a></td></tr>";
      }
      entry.close();
      entry = root.openNextFile();
    }
    root.close();
  }

  html += "</table>";
  html += "<p><a href=\"/downloadAll\">Download all (.zip)</a></p>";
  html += "<p><a href=\"/clear\">Delete all recordings</a></p>";
  html += "</body></html>";

  server_.send(200, "text/html", html);
}

void Dashboard::handleDownload() {
  if (rejectIfRecording()) return;

  if (!server_.hasArg("file")) {
    server_.send(400, "text/plain", "Missing file parameter");
    return;
  }

  const String name = server_.arg("file");
  if (!isSafeFilename(name)) {
    server_.send(400, "text/plain", "Invalid filename");
    return;
  }

  File f = SD.open("/" + name, FILE_READ);
  if (!f || f.isDirectory()) {
    if (f) f.close();
    server_.send(404, "text/plain", "File not found");
    return;
  }

  server_.streamFile(f, "text/csv");
  f.close();
}

void Dashboard::handleDownloadAll() {
  if (rejectIfRecording()) return;

  const std::vector<String> names = listSessionFiles(indexFileName());
  if (names.empty()) {
    server_.send(404, "text/plain", "No recordings to download");
    return;
  }

  // Streamed, uncompressed (store-only) ZIP: each entry's CRC32/size is
  // written in a data descriptor after its data (so we never need to seek
  // back and patch a header), then a central directory listing every entry
  // (with the real CRC32/size/offset, always required there regardless of
  // data descriptors) is appended at the end. See appnote.txt section 4.3
  // for the on-disk format this follows.
  struct ZipEntry {
    String name;
    uint32_t crc;
    uint32_t size;
    uint32_t localHeaderOffset;
    uint16_t dosTime;
    uint16_t dosDate;
  };
  std::vector<ZipEntry> entries;
  entries.reserve(names.size());

  server_.sendHeader("Content-Disposition", "attachment; filename=\"recordings.zip\"");
  server_.setContentLength(CONTENT_LENGTH_UNKNOWN);
  server_.send(200, "application/zip", "");

  uint8_t buf[512];
  uint32_t offset = 0;

  for (const String &name : names) {
    File f = SD.open("/" + name, FILE_READ);
    if (!f) continue;

    uint16_t dosDate, dosTime;
    toDosDateTime(f.getLastWrite(), dosDate, dosTime);
    const uint32_t localHeaderOffset = offset;

    uint8_t header[30];
    writeLE32(header + 0, 0x04034b50);
    writeLE16(header + 4, 20);      // version needed to extract
    writeLE16(header + 6, 0x0008);  // flags: data descriptor follows
    writeLE16(header + 8, 0);       // method: stored (no compression)
    writeLE16(header + 10, dosTime);
    writeLE16(header + 12, dosDate);
    writeLE32(header + 14, 0);  // crc32 (in data descriptor instead)
    writeLE32(header + 18, 0);  // compressed size (in data descriptor)
    writeLE32(header + 22, 0);  // uncompressed size (in data descriptor)
    writeLE16(header + 26, name.length());
    writeLE16(header + 28, 0);  // extra field length

    server_.sendContent(reinterpret_cast<const char *>(header), sizeof(header));
    server_.sendContent(name.c_str(), name.length());
    offset += sizeof(header) + name.length();

    uint32_t crc = 0;
    uint32_t fileSize = 0;
    int n;
    while ((n = f.read(buf, sizeof(buf))) > 0) {
      crc = crc32Update(crc, buf, static_cast<size_t>(n));
      server_.sendContent(reinterpret_cast<const char *>(buf), static_cast<size_t>(n));
      fileSize += static_cast<uint32_t>(n);
    }
    f.close();
    crc = crc32Finish(crc);
    offset += fileSize;

    uint8_t desc[16];
    writeLE32(desc + 0, 0x08074b50);  // optional but widely-recognized signature
    writeLE32(desc + 4, crc);
    writeLE32(desc + 8, fileSize);   // compressed size == uncompressed (stored)
    writeLE32(desc + 12, fileSize);
    server_.sendContent(reinterpret_cast<const char *>(desc), sizeof(desc));
    offset += sizeof(desc);

    entries.push_back({name, crc, fileSize, localHeaderOffset, dosTime, dosDate});
  }

  const uint32_t centralDirOffset = offset;
  for (const auto &e : entries) {
    uint8_t cdh[46];
    writeLE32(cdh + 0, 0x02014b50);
    writeLE16(cdh + 4, 20);   // version made by
    writeLE16(cdh + 6, 20);   // version needed to extract
    writeLE16(cdh + 8, 0x0008);
    writeLE16(cdh + 10, 0);   // method: stored
    writeLE16(cdh + 12, e.dosTime);
    writeLE16(cdh + 14, e.dosDate);
    writeLE32(cdh + 16, e.crc);
    writeLE32(cdh + 20, e.size);
    writeLE32(cdh + 24, e.size);
    writeLE16(cdh + 28, e.name.length());
    writeLE16(cdh + 30, 0);  // extra field length
    writeLE16(cdh + 32, 0);  // file comment length
    writeLE16(cdh + 34, 0);  // disk number start
    writeLE16(cdh + 36, 0);  // internal file attributes
    writeLE32(cdh + 38, 0);  // external file attributes
    writeLE32(cdh + 42, e.localHeaderOffset);

    server_.sendContent(reinterpret_cast<const char *>(cdh), sizeof(cdh));
    server_.sendContent(e.name.c_str(), e.name.length());
    offset += sizeof(cdh) + e.name.length();
  }
  const uint32_t centralDirSize = offset - centralDirOffset;

  uint8_t eocd[22];
  writeLE32(eocd + 0, 0x06054b50);
  writeLE16(eocd + 4, 0);
  writeLE16(eocd + 6, 0);
  writeLE16(eocd + 8, static_cast<uint16_t>(entries.size()));
  writeLE16(eocd + 10, static_cast<uint16_t>(entries.size()));
  writeLE32(eocd + 12, centralDirSize);
  writeLE32(eocd + 16, centralDirOffset);
  writeLE16(eocd + 20, 0);  // comment length
  server_.sendContent(reinterpret_cast<const char *>(eocd), sizeof(eocd));

  server_.sendContent("");  // terminates the chunked response
}

void Dashboard::handleClearConfirm() {
  if (rejectIfRecording()) return;

  server_.send(200, "text/html",
               "<html><body><h1>Delete all recordings?</h1>"
               "<p>This removes every recorded session from the SD card and cannot be undone.</p>"
               "<form method='POST' action='/clear'><button type='submit'>Yes, delete all</button></form>"
               "<p><a href=\"/\">Cancel</a></p></body></html>");
}

void Dashboard::handleClearAll() {
  if (rejectIfRecording()) return;

  const std::vector<String> names = listSessionFiles(indexFileName());

  size_t deleted = 0;
  for (const String &name : names) {
    if (SD.remove("/" + name)) deleted++;
  }

  SD.remove(ImuRecorder::kIndexPath);
  File idx = SD.open(ImuRecorder::kIndexPath, FILE_WRITE);
  if (idx) {
    idx.print(0);
    idx.close();
  }

  server_.send(200, "text/html",
               "<html><body><h1>Deleted " + String(deleted) + " file(s)</h1>" +
               "<p><a href=\"/\">Back</a></p></body></html>");
}

bool Dashboard::isSafeFilename(const String &name) {
  if (name.length() == 0 || name.length() > 40) return false;
  if (name.indexOf("..") >= 0) return false;

  for (size_t i = 0; i < name.length(); i++) {
    const char c = name[i];
    if (!isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-' && c != '.') {
      return false;
    }
  }
  return true;
}

String Dashboard::formatTimestamp(time_t t) {
  if (t <= 0) return "unknown";

  struct tm tmv;
  localtime_r(&t, &tmv);
  char buf[24];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
  return String(buf);
}
