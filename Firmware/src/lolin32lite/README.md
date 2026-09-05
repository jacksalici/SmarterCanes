# Lolin32 Lite IMU Recorder

A firmware for recording 6-axis IMU data (accel + gyro) plus distance-to-ground/obstacle to SD card, with button-based session control and an LED that blinks while recording.

## Hardware

- **Lolin32 Lite** (ESP32): Main controller
- **ISM330DLC IMU** (I2C, SDA=GPIO26, SCL=GPIO25): 3-axis accelerometer + 3-axis gyroscope
- **Modulino Distance** (VL53L4CD, I2C — shared bus with the IMU, SDA=GPIO26, SCL=GPIO25, address `0x29`): time-of-flight distance sensor
- **microSD card module** (SPI, CS=GPIO5, SCK=GPIO18, MOSI=GPIO23, MISO=GPIO19): Data storage
- **Push button** (GPIO12, active-low with internal pull-up): Gesture control
- **Status LED** (GPIO32, through a series resistor to GND): 3 fast blinks on boot, then blinks while recording, off otherwise
  - Note: the Lolin32 Lite's onboard blue LED is hardwired to GPIO22 (active-low) — GPIO32 avoids sharing that pin.

## Architecture

Four-layer design:

1. **Button** (`Button.h/.cpp`): Non-blocking gesture detection
   - Debounce (30 ms), click/double-click/long-press (800 ms threshold)
   - Emits one event per `update()` call

2. **StatusLed** (`StatusLed.h/.cpp`): Non-blocking LED blinker
   - 3 fast blocking blinks (100 ms on/off) at boot as a startup indicator
   - Off while idle; blinks at a fixed 300 ms interval while active
   - Driven by `recorder.isRecording()` every loop iteration

3. **ImuRecorder** (`ImuRecorder.h/.cpp`): IMU + distance sampling + file management
   - Fixed 10 ms sample interval → CSV rows
   - Distance is read opportunistically (VL53L4CD updates slower than the sample rate); the last known reading is reused between updates, and `-1` is written if no reading has been received yet or the sensor is unavailable
   - Auto-numbered sessions via index file (`/rec_index.txt`)
   - Renames files on stop with annotation suffix (`_ann-1.csv`, `_ann0.csv`, `_ann1.csv`)
   - Periodic flush (1 sec) to prevent data loss

4. **Main** (`main.cpp`): Event loop
   - Single click: toggle recording
   - Double-click: stop with annotation 0
   - Long-press: stop with annotation 1

## Build & Deploy

**Build:**
```bash
cd Firmware
pio run -e lolin32lite
```

**Upload:**
```bash
pio run -e lolin32lite -t upload
```

**Monitor serial output (115200 baud):**
```bash
pio device monitor -e lolin32_lite
```

**Clean build:**
```bash
pio run -e lolin32lite --target clean
pio run -e lolin32lite
```

## Serial Output

Firmware logs all events with prefixes:
- `[Main]` — initialization, status
- `[Button]` — gesture detection
- `[Recorder]` — IMU/SD operations

## CSV Format

Sessions are stored as `rec_XXXXX_annY.csv` where Y ∈ {-1, 0, 1}:

```
t_ms,ax_mg,ay_mg,az_mg,gx_mdps,gy_mdps,gz_mdps,dist_mm
0,100,50,980,10,20,5,-1
10,102,48,982,12,18,6,842.5
...
```

- `t_ms`: Milliseconds since recording start
- `ax/ay/az`: Acceleration in millig
- `gx/gy/gz`: Rotation in millidegrees/sec
- `dist_mm`: Distance in millimeters from the Modulino Distance sensor; `-1` if no reading is available yet
