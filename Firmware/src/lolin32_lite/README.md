# Lolin32 Lite IMU Recorder

A firmware for recording 6-axis IMU data (accel + gyro) to SD card with button-based session control.

## Hardware

- **Lolin32 Lite** (ESP32): Main controller
- **ISM330DLC IMU** (I2C, SDA=GPIO26, SCL=GPIO25): 3-axis accelerometer + 3-axis gyroscope
- **microSD card module** (SPI, CS=GPIO5, SCK=GPIO18, MOSI=GPIO23, MISO=GPIO19): Data storage
- **Push button** (GPIO12, active-low with internal pull-up): Gesture control

## Architecture

Three-layer design:

1. **Button** (`Button.h/.cpp`): Non-blocking gesture detection
   - Debounce (30 ms), click/double-click/long-press (800 ms threshold)
   - Emits one event per `update()` call

2. **ImuRecorder** (`ImuRecorder.h/.cpp`): IMU sampling + file management
   - Fixed 10 ms sample interval → CSV rows
   - Auto-numbered sessions via index file (`/rec_index.txt`)
   - Renames files on stop with annotation suffix (`_ann-1.csv`, `_ann0.csv`, `_ann1.csv`)
   - Periodic flush (1 sec) to prevent data loss

3. **Main** (`main.cpp`): Event loop
   - Single click: toggle recording
   - Double-click: stop with annotation 0
   - Long-press: stop with annotation 1

## Build & Deploy

**Build:**
```bash
cd Firmware
pio run -e lolin32_lite
```

**Upload:**
```bash
pio run -e lolin32_lite -t upload
```

**Monitor serial output (115200 baud):**
```bash
pio device monitor -e lolin32_lite
```

**Clean build:**
```bash
pio run -e lolin32_lite --target clean
pio run -e lolin32_lite
```

## Serial Output

Firmware logs all events with prefixes:
- `[Main]` — initialization, status
- `[Button]` — gesture detection
- `[Recorder]` — IMU/SD operations

## CSV Format

Sessions are stored as `rec_XXXXX_annY.csv` where Y ∈ {-1, 0, 1}:

```
t_ms,ax_mg,ay_mg,az_mg,gx_mdps,gy_mdps,gz_mdps
0,100,50,980,10,20,5
10,102,48,982,12,18,6
...
```

- `t_ms`: Milliseconds since recording start
- `ax/ay/az`: Acceleration in millig
- `gx/gy/gz`: Rotation in millidegrees/sec
