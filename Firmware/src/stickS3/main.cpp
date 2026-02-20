/*
 * M5StickS3 IMU Logger
 * =====================
 * Reads IMU at a configurable frequency, queues JSON payloads and POSTs
 * them to a Flask server in a background task.
 *
 * Buttons
 *   BtnA (front)  – hold to show status info on LCD
 *   BtnB (side)   – hold for 2 s to enter WiFi-AP config portal
 *
 * All settings (WiFi, server, frequency) are persisted in NVS via
 * the Preferences library – no env.h needed.
 */

#include <M5Unified.h>
#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <Preferences.h>
#include <freertos/queue.h>

// ═══════════════════════════════════════════════════════════════
//  Defaults (used when NVS is empty)
// ═══════════════════════════════════════════════════════════════
static constexpr const char *DEF_SSID       = "";
static constexpr const char *DEF_PASS       = "";
static constexpr const char *DEF_SRV_HOST   = "192.168.1.100";
static constexpr uint16_t    DEF_SRV_PORT   = 5000;
static constexpr uint16_t    DEF_FREQ_HZ    = 10;      // sampling rate

// ═══════════════════════════════════════════════════════════════
//  Message queue sizing
//  Each payload ≈ 220 bytes.  Keep total RAM usage < 60 KB.
// ═══════════════════════════════════════════════════════════════
static constexpr size_t      MSG_SIZE       = 256;      // bytes per msg
static constexpr size_t      QUEUE_LEN      = 200;      // max buffered msgs

// ═══════════════════════════════════════════════════════════════
//  Runtime state
// ═══════════════════════════════════════════════════════════════
Preferences prefs;

// Config loaded from NVS
String cfgSsid, cfgPass, cfgHost;
uint16_t cfgPort;
uint16_t cfgFreqHz;

String serverUrl;

// Queue
QueueHandle_t msgQueue = nullptr;

// Stats (updated atomically from different cores)
volatile uint32_t totalSamples  = 0;
volatile uint32_t totalSent     = 0;
volatile uint32_t totalFailed   = 0;
volatile int      lastHttpCode  = 0;

// Mode flag
enum Mode { MODE_LOGGING, MODE_CONFIG };
volatile Mode currentMode = MODE_LOGGING;

// AP config portal
WebServer *configServer = nullptr;
static constexpr const char *AP_SSID = "SmartCane-Setup";

// ═══════════════════════════════════════════════════════════════
//  NVS helpers
// ═══════════════════════════════════════════════════════════════
void loadConfig()
{
    prefs.begin("smartcane", true);  // read-only
    cfgSsid   = prefs.getString("ssid",     DEF_SSID);
    cfgPass   = prefs.getString("pass",     DEF_PASS);
    cfgHost   = prefs.getString("srv_host", DEF_SRV_HOST);
    cfgPort   = prefs.getUShort("srv_port", DEF_SRV_PORT);
    cfgFreqHz = prefs.getUShort("freq_hz",  DEF_FREQ_HZ);
    prefs.end();

    if (cfgFreqHz == 0) cfgFreqHz = 1;
    if (cfgFreqHz > 200) cfgFreqHz = 200;

    serverUrl = "http://" + cfgHost + ":" + String(cfgPort) + "/log";
}

void saveConfig()
{
    prefs.begin("smartcane", false); // read-write
    prefs.putString("ssid",     cfgSsid);
    prefs.putString("pass",     cfgPass);
    prefs.putString("srv_host", cfgHost);
    prefs.putUShort("srv_port", cfgPort);
    prefs.putUShort("freq_hz",  cfgFreqHz);
    prefs.end();

    serverUrl = "http://" + cfgHost + ":" + String(cfgPort) + "/log";
}

// ═══════════════════════════════════════════════════════════════
//  WiFi STA connect
// ═══════════════════════════════════════════════════════════════
bool connectWiFi(unsigned long timeoutMs = 15000)
{
    if (cfgSsid.isEmpty()) return false;

    WiFi.mode(WIFI_STA);
    WiFi.begin(cfgSsid.c_str(), cfgPass.c_str());

    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 5);
    M5.Lcd.setTextSize(2);
    M5.Lcd.printf("WiFi...\n%s", cfgSsid.c_str());

    unsigned long t0 = millis();
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        M5.Lcd.print(".");
        if (millis() - t0 > timeoutMs) {
            M5.Lcd.printf("\nTimeout");
            delay(1000);
            return false;
        }
    }

    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 5);
    M5.Lcd.printf("WiFi OK\n%s", WiFi.localIP().toString().c_str());
    delay(800);
    return true;
}

// ═══════════════════════════════════════════════════════════════
//  Config portal HTML  (served from AP mode)
// ═══════════════════════════════════════════════════════════════
static const char CONFIG_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html><html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmartCane Setup</title>
<style>
  *{box-sizing:border-box;font-family:system-ui,sans-serif}
  body{margin:0;padding:24px;background:#f4f4f9;color:#333}
  h1{font-size:1.4rem;margin-bottom:4px}
  p.sub{color:#888;margin-top:0}
  form{max-width:420px;margin:auto;background:#fff;padding:24px;border-radius:12px;
       box-shadow:0 2px 8px rgba(0,0,0,.12)}
  label{display:block;margin-top:14px;font-weight:600;font-size:.9rem}
  input[type=text],input[type=password],input[type=number]{
    width:100%;padding:10px;margin-top:4px;border:1px solid #ccc;border-radius:6px;font-size:1rem}
  button{margin-top:20px;width:100%;padding:12px;background:#4f46e5;color:#fff;
         border:none;border-radius:8px;font-size:1rem;cursor:pointer}
  button:hover{background:#4338ca}
  .info{margin-top:12px;font-size:.82rem;color:#666;text-align:center}
</style></head><body>
<form action="/save" method="POST">
  <h1>SmartCane Setup</h1>
  <p class="sub">Configure WiFi &amp; server settings</p>

  <label>WiFi SSID</label>
  <input name="ssid" type="text" value="%SSID%" required>

  <label>WiFi Password</label>
  <input name="pass" type="password" value="%PASS%">

  <label>Server Host / IP</label>
  <input name="host" type="text" value="%HOST%" required>

  <label>Server Port</label>
  <input name="port" type="number" value="%PORT%" min="1" max="65535" required>

  <label>Sampling Frequency (Hz)</label>
  <input name="freq" type="number" value="%FREQ%" min="1" max="200" required>

  <button type="submit">Save &amp; Reboot</button>
  <p class="info">The device will reboot and connect to the configured WiFi.</p>
</form></body></html>
)rawliteral";

String buildConfigPage()
{
    String page(CONFIG_HTML);
    page.replace("%SSID%", cfgSsid);
    page.replace("%PASS%", cfgPass);
    page.replace("%HOST%", cfgHost);
    page.replace("%PORT%", String(cfgPort));
    page.replace("%FREQ%", String(cfgFreqHz));
    return page;
}

// ═══════════════════════════════════════════════════════════════
//  Config portal handlers
// ═══════════════════════════════════════════════════════════════
void handleConfigRoot()
{
    configServer->send(200, "text/html", buildConfigPage());
}

void handleConfigSave()
{
    cfgSsid   = configServer->arg("ssid");
    cfgPass   = configServer->arg("pass");
    cfgHost   = configServer->arg("host");
    cfgPort   = (uint16_t)configServer->arg("port").toInt();
    cfgFreqHz = (uint16_t)configServer->arg("freq").toInt();

    saveConfig();

    configServer->send(200, "text/html",
        "<html><body style='font-family:sans-serif;text-align:center;padding:40px'>"
        "<h2>Saved!</h2><p>Rebooting...</p></body></html>");

    delay(1500);
    ESP.restart();
}

// ═══════════════════════════════════════════════════════════════
//  Enter / exit config portal mode
// ═══════════════════════════════════════════════════════════════
void startConfigPortal()
{
    currentMode = MODE_CONFIG;

    WiFi.disconnect(true);
    delay(200);
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID);

    IPAddress apIp = WiFi.softAPIP();

    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 5);
    M5.Lcd.setTextSize(2);
    M5.Lcd.printf("CONFIG\nMode\n\n");
    M5.Lcd.printf("AP: %s\n", AP_SSID);
    M5.Lcd.printf("Go to:\n%s", apIp.toString().c_str());

    configServer = new WebServer(80);
    configServer->on("/",     HTTP_GET,  handleConfigRoot);
    configServer->on("/save", HTTP_POST, handleConfigSave);
    configServer->begin();

    // Stay in config mode until reboot (save handler reboots)
    while (true) {
        configServer->handleClient();
        M5.update();
        delay(5);
    }
}

// ═══════════════════════════════════════════════════════════════
//  Show info screen (while BtnA held)
// ═══════════════════════════════════════════════════════════════
void showInfoScreen()
{
    float batV   = M5.Power.getBatteryVoltage() / 1000.0f;
    float batPct = (batV < 3.2f) ? 0.0f : (batV - 3.2f) * 100.0f;

    uint32_t queued = (msgQueue) ? uxQueueMessagesWaiting(msgQueue) : 0;

    M5.Lcd.clear();
    M5.Lcd.setCursor(0, 2);
    M5.Lcd.setTextSize(1);
    M5.Lcd.printf("=== SmartCane ===\n\n");
    M5.Lcd.printf("WiFi: %s\n", (WiFi.status() == WL_CONNECTED) ? "OK" : "N/C");
    M5.Lcd.printf("IP:   %s\n", WiFi.localIP().toString().c_str());
    M5.Lcd.printf("SSID: %s\n\n", cfgSsid.c_str());
    M5.Lcd.printf("Srv:  %s:%d\n\n", cfgHost.c_str(), cfgPort);
    M5.Lcd.printf("Freq: %d Hz\n", cfgFreqHz);
    M5.Lcd.printf("Bat:  %.0f%%  (%.2fV)\n\n", batPct, batV);
    M5.Lcd.printf("Samples: %lu\n", totalSamples);
    M5.Lcd.printf("Sent:    %lu\n", totalSent);
    M5.Lcd.printf("Failed:  %lu\n", totalFailed);
    M5.Lcd.printf("Queued:  %lu/%d\n", queued, QUEUE_LEN);
    M5.Lcd.printf("HTTP:    %d\n", lastHttpCode);
}

// ═══════════════════════════════════════════════════════════════
//  Sender task – runs on core 0, pops queue → HTTP POST
// ═══════════════════════════════════════════════════════════════
void senderTask(void *param)
{
    char buf[MSG_SIZE];

    for (;;) {
        // Block up to 500 ms waiting for a message
        if (xQueueReceive(msgQueue, buf, pdMS_TO_TICKS(500)) == pdTRUE) {
            if (WiFi.status() != WL_CONNECTED) {
                totalFailed++;
                continue;   // drop if WiFi is down – avoids blocking
            }

            HTTPClient http;
            http.begin(serverUrl);
            http.addHeader("Content-Type", "application/json");
            http.setTimeout(2000);          // 2 s max per POST
            int code = http.POST(buf);
            http.end();

            lastHttpCode = code;
            if (code == 200) {
                totalSent++;
            } else {
                totalFailed++;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  Setup
// ═══════════════════════════════════════════════════════════════
void setup()
{
    M5.begin();
    M5.Lcd.setTextSize(2);

    loadConfig();

    // ── Check BtnB held at boot → config portal ──────────────
    M5.update();
    if (M5.BtnB.isPressed()) {
        startConfigPortal();        // never returns (reboots on save)
    }

    // ── If no SSID configured, force config portal ───────────
    if (cfgSsid.isEmpty()) {
        M5.Lcd.clear();
        M5.Lcd.setCursor(0, 20);
        M5.Lcd.printf("No WiFi\nconfigured!\n\nStarting\nsetup...");
        delay(2000);
        startConfigPortal();
    }

    // ── Normal boot: connect WiFi ────────────────────────────
    if (!connectWiFi()) {
        M5.Lcd.clear();
        M5.Lcd.setCursor(0, 20);
        M5.Lcd.printf("WiFi fail\nHold BtnB\nto config");
        // Fall through – sender task will just count failures
    }

    // ── Create queue ─────────────────────────────────────────
    msgQueue = xQueueCreate(QUEUE_LEN, MSG_SIZE);

    // ── Launch sender on core 0 (main loop runs on core 1) ──
    xTaskCreatePinnedToCore(
        senderTask,        // function
        "sender",          // name
        8192,              // stack bytes
        nullptr,           // param
        1,                 // priority
        nullptr,           // handle
        0                  // core 0
    );
}

// ═══════════════════════════════════════════════════════════════
//  Loop  – core 1: sample IMU, enqueue, handle buttons
// ═══════════════════════════════════════════════════════════════
static unsigned long btnBHoldStart = 0;

void loop()
{
    M5.update();

    // ── BtnB held for 2 s → config portal ────────────────────
    if (M5.BtnB.isPressed()) {
        if (btnBHoldStart == 0) btnBHoldStart = millis();
        if (millis() - btnBHoldStart > 2000) {
            startConfigPortal();   // never returns
        }
    } else {
        btnBHoldStart = 0;
    }

    // ── BtnA held → info screen (skip sampling while held) ──
    if (M5.BtnA.isHolding()) {
        showInfoScreen();
        delay(200);
        return;
    }

    // ── Reconnect WiFi if dropped ────────────────────────────
    static unsigned long lastReconnect = 0;
    if (WiFi.status() != WL_CONNECTED && millis() - lastReconnect > 10000) {
        WiFi.begin(cfgSsid.c_str(), cfgPass.c_str());
        lastReconnect = millis();
    }

    // ── Read IMU ─────────────────────────────────────────────
    if (!M5.Imu.update()) {
        delay(1);
        return;
    }

    auto d = M5.Imu.getImuData();

    float batV   = M5.Power.getBatteryVoltage() / 1000.0f;
    float batPct = (batV < 3.2f) ? 0.0f : (batV - 3.2f) * 100.0f;

    // ── Build JSON & enqueue ─────────────────────────────────
    char json[MSG_SIZE];
    snprintf(json, sizeof(json),
             "{\"timestamp\":%lu,"
             "\"accel_x\":%.4f,\"accel_y\":%.4f,\"accel_z\":%.4f,"
             "\"gyro_x\":%.4f,\"gyro_y\":%.4f,\"gyro_z\":%.4f,"
             "\"bat\":%.1f}",
             millis(),
             d.accel.x, d.accel.y, d.accel.z,
             d.gyro.x,  d.gyro.y,  d.gyro.z,
             batPct);

    // Non-blocking enqueue – if full the oldest is NOT overwritten;
    // the newest sample is simply dropped (safe for bounded memory).
    xQueueSend(msgQueue, json, 0);
    totalSamples++;

    // ── Minimal LCD (only refresh every ~0.5 s to save time) ─
    static unsigned long lastLcd = 0;
    if (millis() - lastLcd > 500) {
        lastLcd = millis();
        M5.Lcd.setCursor(0, 5);
        M5.Lcd.clear();
        M5.Lcd.setTextSize(2);
        M5.Lcd.printf("Logging\n%d Hz\n\n", cfgFreqHz);
        M5.Lcd.printf("Q:%lu/%d\n", (uint32_t)uxQueueMessagesWaiting(msgQueue), QUEUE_LEN);
        M5.Lcd.printf("OK:%lu\n", totalSent);
        M5.Lcd.printf("BAT:%.0f%%\n", batPct);
    }

    // ── Pace to configured frequency ─────────────────────────
    delay(1000 / cfgFreqHz);
}    