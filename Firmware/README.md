# Lolin32 Lite IMU Recorder

A firmware for recording 6-axis IMU data (accel + gyro) plus distance-to-ground/obstacle to SD card, with button-based session control, an LED that blinks while recording (and idle-heartbeats/solid-faults/gesture-signals otherwise), and a WiFi dashboard to download and clear recordings.

## Hardware

- **Lolin32 Lite** (ESP32): Main controller
- **ISM330DLC IMU** (I2C, SDA=GPIO26, SCL=GPIO25): 3-axis accelerometer + 3-axis gyroscope
- **Modulino Distance** (VL53L4CD, I2C — shared bus with the IMU, SDA=GPIO26, SCL=GPIO25, address `0x29`): time-of-flight distance sensor
- **microSD card module** (SPI, CS=GPIO5, SCK=GPIO18, MOSI=GPIO23, MISO=GPIO19): Data storage
- **Push button** (GPIO12, active-low with internal pull-up): Gesture control - click to start/mark, double-click/long-press to stop
- **Status LED** (GPIO32, through a series resistor to GND): 3 fast blinks on boot, then blinks once/sec while recording, a brief heartbeat flash every 30s while idle, solid on if a fault occurs (SD card/IMU stuck at boot, or an SD write failure mid-recording), a one-shot blink pattern after a button gesture (triple blink to confirm a stop, quick double blink to confirm an event mark)
  - Note: the Lolin32 Lite's onboard blue LED is hardwired to GPIO22 (active-low) — GPIO32 avoids sharing that pin.

## Architecture

Five-layer design:

1. **Button** (`Button.h/.cpp`): Non-blocking gesture detection
   - Debounce (30 ms), click/double-click/long-press (800 ms threshold)
   - Emits one event per `update()` call

2. **StatusLed** (`StatusLed.h/.cpp`): Non-blocking LED blinker
   - 3 fast blinks at boot; once-per-second blink while recording; a brief heartbeat flash every 30 s while idle
   - Solid on whenever a fault is active (SD write/open failure, or IMU/SD not responding at boot), regardless of active/idle state
   - `signal(Signal)` queues a one-shot pattern that preempts the idle/active visual: `StopBlink` (3 blinks) after a recording stops, `EventBlink` (2 quick blinks) after a mid-recording event mark — a fault still takes priority

3. **ImuRecorder** (`ImuRecorder.h/.cpp`): IMU + distance sampling + file management
   - Fixed 10 ms sample interval; samples are batched in a 2 KB RAM buffer and flushed to SD at most every 250 ms
   - A session is split into 30-second segment files, so an SD fault only costs the segment in progress
   - A failed write or file-open marks the segment dead and retries into a fresh one on a 500 ms backoff
   - Distance is read opportunistically (VL53L4CD updates slower than the sample rate); `-1` is written if no reading is available
   - `annotateEvent()` flags the next sampled row's `event` column instead of interrupting the recording
   - Auto-numbered sessions via index file (`/rec_index.txt`); on stop, every segment is renamed with the annotation suffix (`_ann-1.csv`, `_ann0.csv`, `_ann1.csv`)
   - Periodic flush (1 s) to bound data loss on power failure

4. **Dashboard** (`Dashboard.h/.cpp`): WiFi + NTP + web UI for offloading recordings
   - Connects to WiFi at boot using credentials read from `/.env` on the SD card; missing file or failed connection is non-fatal — the device keeps recording standalone
   - On connect, syncs time via NTP (`configTzTime`, Europe/Rome) so SD file timestamps are meaningful
   - Serves a single-page dashboard (`WebServer`, no auth) listing every recorded file, with per-file download, a streamed "Download all" `.zip`, and a two-step "Delete all"
   - Every route refuses (503) while recording, since a blocking HTTP request would otherwise stall the 10 ms sample loop

5. **Main** (`main.cpp`): Event loop
   - Single click: start recording if idle; if already recording, mark an event on the CSV instead (LED: quick double blink)
   - Double-click: stop with annotation 1 (LED: triple blink)
   - Long-press: stop with annotation 0 (LED: triple blink)

## Build & Deploy

`cane` is the default PlatformIO environment (see `platformio.ini`), so `-e cane` can be omitted below.

**Build:**
```bash
cd Firmware
pio run -e cane
```

**Upload:**
```bash
pio run -e cane -t upload
```

**Monitor serial output (115200 baud):**
```bash
pio device monitor -e cane -b 115200
```

**Clean build:**
```bash
pio run -e cane --target clean
pio run -e cane
```

## Power

See [POWER_ANALYSIS.md](POWER_ANALYSIS.md) for the component-by-component current budget, estimated
battery life per scenario, and power-saving opportunities not yet implemented.

## Dashboard

WiFi credentials are read from a plain-text file at the SD card root, `/.env` — not hardcoded, so they can be changed without reflashing. Create it on the card before inserting:

```
WIFI_SSID=YourNetworkName
WIFI_PASSWORD=YourPassword
```

(Blank lines and lines starting with `#` are ignored.) This file is excluded from the dashboard's file listing and from "Download all"/"Delete all" — it's never treated as a recording, and `/clear` never removes it.

If `/.env` is missing or has no `WIFI_SSID`, boot skips WiFi entirely (logged, non-fatal) and the device just records standalone.

On boot, the serial log shows the WiFi connect attempt and either the dashboard URL (`http://<ip>/`) on success, or a warning that it's continuing without the dashboard (recording still works normally either way).

Once connected, open that URL in a browser on the same network:
- `/` — lists every recorded session (name, size, timestamp) with a **Download** link each
- **Download all (.zip)** — streams every recorded session as a single uncompressed `.zip` (built on the fly, not buffered in RAM or written to the SD card)
- **Delete all recordings** — a confirmation page, then one button that wipes every session file and resets the session counter back to 0

The dashboard is unavailable (503) while a recording is in progress — stop the recording via the button first.

## Serial Output

Firmware logs all events with prefixes:
- `[Main]` — initialization, status
- `[Button]` — gesture detection
- `[Recorder]` — IMU/SD operations
- `[Dashboard]` — WiFi/NTP status

## CSV Format

See [Dataset/README.md](../Dataset/README.md) for the recording filename convention, the CSV column
layout, and the `annY` annotation.

## Related: `distance_test`

A separate, standalone project under `Firmware/src/distance_test/` — just the
Modulino Distance sensor, printing one reading per line for the Serial
Plotter, independent of the rest of this firmware. Useful for checking
sensor wiring/mounting without recording a full session:

```bash
pio run -e distance_test -t upload
```
