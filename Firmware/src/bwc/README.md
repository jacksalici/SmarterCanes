# Lolin32 Lite IMU Recorder

A firmware for recording 6-axis IMU data (accel + gyro) plus distance-to-ground/obstacle to SD card, with button-based session control, an LED that blinks while recording (and idle-heartbeats/solid-faults otherwise), and a WiFi dashboard to download and clear recordings.

## Hardware

- **Lolin32 Lite** (ESP32): Main controller
- **ISM330DLC IMU** (I2C, SDA=GPIO26, SCL=GPIO25): 3-axis accelerometer + 3-axis gyroscope
- **Modulino Distance** (VL53L4CD, I2C — shared bus with the IMU, SDA=GPIO26, SCL=GPIO25, address `0x29`): time-of-flight distance sensor
- **microSD card module** (SPI, CS=GPIO5, SCK=GPIO18, MOSI=GPIO23, MISO=GPIO19): Data storage
- **Push button** (GPIO12, active-low with internal pull-up): Gesture control
- **Status LED** (GPIO32, through a series resistor to GND): 3 fast blinks on boot, then blinks while recording, a brief heartbeat flash every 30s while idle, solid on if a fault occurs (SD card/IMU stuck at boot, or an SD write failure mid-recording)
  - Note: the Lolin32 Lite's onboard blue LED is hardwired to GPIO22 (active-low) — GPIO32 avoids sharing that pin.

## Architecture

Five-layer design:

1. **Button** (`Button.h/.cpp`): Non-blocking gesture detection
   - Debounce (30 ms), click/double-click/long-press (800 ms threshold)
   - Emits one event per `update()` call

2. **StatusLed** (`StatusLed.h/.cpp`): Non-blocking LED blinker
   - 3 fast blocking blinks (100 ms on/off) at boot as a startup indicator
   - Blinks at a fixed 300 ms interval while active (recording)
   - While idle, a brief 100 ms heartbeat flash every 30s - distinguishes "idle" from "unpowered/frozen" without a full-time blink
   - Solid on (no blinking) whenever a fault is active - `recorder.hasFault()` (an SD write/open failure mid-recording), or a boot-time init retry loop in `main.cpp` (SD card / IMU not responding) - regardless of the active/idle state, so a fault is visually distinct from either blink pattern
   - Driven by `recorder.isRecording()` / `recorder.hasFault()` every loop iteration, and directly from `main.cpp`'s boot-time retry loop

3. **ImuRecorder** (`ImuRecorder.h/.cpp`): IMU + distance sampling + file management
   - Fixed 10 ms sample interval; samples are batched in a 2 KB RAM buffer and written to SD in bulk (at most every 250 ms or when the buffer fills), rather than one SD write per sample
   - A session is split into 30-second segment files instead of one continuous file, so an SD fault only costs the segment in progress - every prior segment was already flushed and closed
   - A failed write or file-open marks the segment dead and retries into a fresh one on a 500 ms backoff, so a transient fault (e.g. a jostled card connection) self-heals instead of silently halting data collection for the rest of the session
   - Distance is read opportunistically (VL53L4CD updates slower than the sample rate); the last known reading is reused between updates, and `-1` is written if no reading has been received yet or the sensor is unavailable
   - Auto-numbered sessions via index file (`/rec_index.txt`)
   - On stop, every segment belonging to the session is renamed with the annotation suffix (`_ann-1.csv`, `_ann0.csv`, `_ann1.csv`)
   - Periodic flush (1 sec) of whatever's already reached the file, on top of the buffer's own flush, to bound data loss on power failure

4. **Dashboard** (`Dashboard.h/.cpp`): WiFi + NTP + web UI for offloading recordings
   - Connects to WiFi at boot using credentials read from `/.env` on the SD card (bounded ~10 s timeout); missing file or failure to connect is non-fatal — the device keeps recording standalone, it just skips starting the dashboard
   - On connect, syncs time via NTP (`configTzTime`, Europe/Rome) so SD file timestamps are meaningful; NTP failure is also non-fatal
   - Serves a single-page dashboard (`WebServer`, no auth, no JS) listing every recorded file with a Download link, a "Download all" streamed `.zip`, and a two-step "Delete all" action
   - Every route refuses (503) while `recorder.isRecording()` is true, since a blocking HTTP request (e.g. a large download) would otherwise stall the 10 ms IMU sample loop

5. **Main** (`main.cpp`): Event loop
   - Single click: toggle recording
   - Double-click: stop with annotation 0
   - Long-press: stop with annotation 1

## Build & Deploy

`bwc` is the default PlatformIO environment (see `platformio.ini`), so `-e bwc` can be omitted below.

**Build:**
```bash
cd Firmware
pio run -e bwc
```

**Upload:**
```bash
pio run -e bwc -t upload
```

**Monitor serial output (115200 baud):**
```bash
pio device monitor -e bwc -b 115200
```

**Clean build:**
```bash
pio run -e bwc --target clean
pio run -e bwc
```

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

Each session is stored as one or more 30-second segment files, `rec_XXXXX_segNNN_annY.csv`, where `XXXXX` is the session index, `NNN` the zero-based segment number, and `Y ∈ {-1, 0, 1}` the stop annotation (shared by every segment of the session):

```
t_ms,ax_mg,ay_mg,az_mg,gx_mdps,gy_mdps,gz_mdps,dist_mm
0,100,50,980,10,20,5,-1
10,102,48,982,12,18,6,842.5
...
```

`t_ms` restarts from 0 in each segment (milliseconds since that segment, not the session, started); consecutive segments of the same session are meant to be read back-to-back as one continuous recording.

- `t_ms`: Milliseconds since recording start
- `ax/ay/az`: Acceleration in millig
- `gx/gy/gz`: Rotation in millidegrees/sec
- `dist_mm`: Distance in millimeters from the Modulino Distance sensor; `-1` if no reading is available yet

## Related: `distance_test`

A separate, standalone project under `Firmware/src/distance_test/` — just the
Modulino Distance sensor, printing one reading per line for the Serial
Plotter, independent of the rest of this firmware. Useful for checking
sensor wiring/mounting without recording a full session:

```bash
pio run -e distance_test -t upload
```
