# Power Consumption Analysis

Estimated current draw of the Lolin32 Lite IMU Recorder, broken down by
component and by operating scenario, with a **standard** (typical-datasheet)
and a **conservative** (worst-case / poorly-behaved-hardware) estimate for
each. There is no bench measurement behind these numbers - it's a
component-budget estimate built from published datasheet figures and common
ESP32 board behavior. Treat it as a planning tool, not a spec; validate
against a real measurement (USB power meter, or a multimeter/shunt resistor,
or an INA219-type sensor inline on the battery lead) before relying on it for
anything battery-critical.

**mA vs mAh:** all figures below are *average current* in mA. Since
mAh = mA x hours, a sustained average of `X` mA is simply `X` mAh consumed
per hour - the two numbers are the same, which is why the tables only list
mA once.

## 1. Method

Every scenario (idle, recording, dashboard active, fault, ...) is built by
summing each component's contribution for that scenario:

```
scenario total (mA) = sum over components of (active_current x duty_cycle_in_scenario)
```

The firmware never sleeps the CPU or the sensors - `loop()` runs
continuously, and `ImuRecorder::begin()` enables the IMU's accelerometer and
gyroscope once at boot and never disables them, regardless of whether a
session is actively recording (see [Section 5](#5-power-saving-opportunities-not-implemented)).
So the IMU and distance sensor draw roughly the same current whether idle or
recording; only the ESP32's own workload (SD writes, WiFi) actually changes
between states.

## 2. Component current budget

| Component | State | Standard | Conservative | Notes |
|---|---|---:|---:|---|
| ESP32 (Lolin32 Lite) | CPU active, WiFi radio off | 35 mA | 45 mA | `loop()` runs flat-out (no `delay()` beyond the boot blinks and the 1 s fault-retry backoff); higher clock (240 MHz) plus SPI/I2C peripheral overhead |
| ESP32 WiFi | STA connected, idle (default modem-sleep) | +20 mA | +80 mA | Standard assumes the default `WIFI_PS_MIN_MODEM` power-save actually engages; conservative assumes it barely helps, since `Dashboard::poll()` calls `handleClient()` every loop iteration with no idle gap for the radio to sleep between |
| ESP32 WiFi | Active transfer (download/zip stream) | 120 mA | 180 mA | Sustained TX/RX during `/download` or `/downloadAll`; peaks can run higher still on a poorly-regulated board |
| ISM330DLC (IMU) | Always on (accel + gyro enabled) | 0.9 mA | 2.5 mA | Low-power vs. high-performance ODR/mode, per ST datasheet family figures |
| Modulino Distance (VL53L4CD + bridge MCU) | Always ranging | 20 mA | 35 mA | VL53L4CD continuous-ranging average plus the Modulino board's own interface MCU/regulator overhead |
| microSD card + module | Idle (mounted, not accessed) | 0.5 mA | 1 mA | Card powered but not being read/written |
| microSD card + module | Write burst (during a buffered flush) | 60 mA for ~10 ms | 100 mA for ~50 ms | Per flush event (every ≤250 ms while recording, per `ImuRecorder::kBufferFlushIntervalMs`); duty-cycled into the per-scenario average below |
| Status LED | Lit | ~4 mA | ~6 mA | **Unverified assumption** - no resistor value is given in the schematic/README; assumes a ~330 Ω (standard) to ~220 Ω (conservative) series resistor and a ~2 V LED at 3.3 V. Check the actual resistor before trusting this line. |
| Push button | Pressed (internal pull-up) | <0.1 mA | <0.1 mA | Negligible; also transient, not continuous |
| Board/regulator overhead | Always | 3 mA | 8 mA | LDO quiescent current, miscellaneous board traces; conservative assumes power comes in through the onboard 5 V→3.3 V regulator rather than a clean 3.3 V rail |

## 3. Per-scenario totals

LED contribution below already accounts for its duty cycle: idle heartbeat
(100 ms every 30 s, ~0.3% duty) is negligible; recording blink (300 ms
on/off, i.e. 50% duty) averages ~2 mA (standard) / ~3 mA (conservative);
solid-on fault is the full lit current (~4 / ~6 mA).

SD write-burst contribution is likewise duty-cycled: standard assumes a
60 mA burst for ~10 ms out of every 250 ms window (≈2.4 mA average, plus
0.5 mA idle baseline ≈ **3 mA**); conservative assumes a slower/less
efficient card - a 100 mA burst held for ~50 ms out of every 250 ms window
(≈20 mA average, plus 1 mA idle baseline ≈ **21 mA**).

| Scenario | Standard | Conservative |
|---|---:|---:|
| **A. Idle, no WiFi** (no `/.env` credentials, or WiFi failed to connect) | 35 + 0.9 + 20 + 0.5 + ~0 + 3 = **59.4 mA** | 45 + 2.5 + 35 + 1 + ~0 + 8 = **91.5 mA** |
| **B. Idle, WiFi connected** (dashboard listening, no active request) | 59.4 + 20 = **79.4 mA** | 91.5 + 80 = **171.5 mA** |
| **C. Recording, no WiFi** | 35 + 0.9 + 20 + 3 + 2 + 3 = **63.9 mA** | 45 + 2.5 + 35 + 21 + 3 + 8 = **114.5 mA** |
| **D. Recording, WiFi connected** (associated but every route is 503'd by `rejectIfRecording()`) | 63.9 + 20 = **83.9 mA** | 114.5 + 80 = **194.5 mA** |
| **E. Dashboard active transfer** (downloading a recording/zip; only possible while *not* recording) | 35 + 120 + 0.9 + 20 + 15 + ~0 + 3 = **193.9 mA** | 45 + 180 + 2.5 + 35 + 40 + ~0 + 8 = **310.5 mA** |
| **F. Fault, boot-time** (`waitFor()` retry loop in `main.cpp`; WiFi never started yet, sensors not yet enabled) | 35 + 3 + 4 = **42 mA** | 45 + 8 + 6 = **59 mA** |
| **G. Fault, mid-recording SD write failure** (`hasFault()` true; LED solid instead of blinking, SD writes failing/retrying instead of succeeding) | 35 + 0.9 + 20 + 0.5 + 4 + 3 = **63.4 mA** | 45 + 2.5 + 35 + 1 + 6 + 8 = **97.5 mA** |

Boot itself (SD/IMU init, WiFi connect attempt, NTP sync - a few seconds to
at most ~10 s if WiFi is slow to associate) is a short transient and doesn't
meaningfully change an hourly average; it's omitted as its own row.

Note how close C/D are to G - a stuck SD card barely changes total current
(it trades a small write-burst average for a solid LED), which is exactly
why the earlier silent-write-failure bug was so hard to notice without
instrumentation: a faulted recording draws essentially the same power as a
healthy one.

## 4. Estimated battery life

Hours of runtime = battery capacity (mAh) / scenario current (mA), assuming
100% usable capacity (real cells rarely give you all of their rated mAh -
see caveats below).

| Battery | A. Idle, no WiFi | C. Recording, no WiFi | B. Idle, WiFi | D. Recording, WiFi |
|---|---:|---:|---:|---:|
| 500 mAh (small LiPo) | 8.4 h / 5.5 h | 7.8 h / 4.4 h | 6.3 h / 2.9 h | 6.0 h / 2.6 h |
| 1000 mAh | 16.8 h / 10.9 h | 15.6 h / 8.7 h | 12.6 h / 5.8 h | 11.9 h / 5.1 h |
| 2000 mAh (common 18650) | 33.7 h / 21.9 h | 31.3 h / 17.5 h | 25.2 h / 11.7 h | 23.8 h / 10.3 h |
| 3000 mAh | 50.5 h / 32.8 h | 46.9 h / 26.2 h | 37.8 h / 17.5 h | 35.7 h / 15.4 h |

Each cell reads **standard / conservative**. For a device that's mostly idle
with occasional recording sessions, the realistic figure sits between the
"Idle, no WiFi" and "Recording, no WiFi" columns - e.g. a 2000 mAh cell gives
somewhere around **22-34 hours** of standalone use depending on how
pessimistic you want to be, or **10-25 hours** if WiFi/dashboard stays
connected the whole time (leaving it connected has a bigger effect on
runtime than whether it's actively recording).

## 5. Power-saving opportunities (not implemented)

Not changes made in this pass - just what the numbers point at, if runtime
becomes a real constraint:

- **Suspend the IMU/distance sensor when idle.** They're enabled once at
  boot and never turned off (`ImuRecorder::begin()`), so they burn ~21-38 mA
  continuously whether or not a session is recording - the single biggest
  idle-power line item after WiFi. Putting them in standby between sessions
  and re-enabling on `start()` would cut idle current substantially.
- **Only bring WiFi up on demand** (e.g. a distinct button gesture) instead
  of connecting for the whole time the device is powered, or disconnect
  after a period with no dashboard activity. Comparing columns A/B and C/D
  above, WiFi accounts for roughly a third of total idle/recording current.
- **Verify/tune the LED resistor** and confirm actual `WiFi.setSleep()`
  behavior with a real current-clamp measurement - both are estimated here,
  not measured.

## 6. Caveats

- All component figures are estimated from typical/worst-case datasheet
  ranges for the component families involved, not from a specific measured
  unit. Real silicon varies, and so does board layout/regulator quality.
- The LED series-resistor value is assumed (see the table above) since it's
  not specified in the schematic/README - verify before trusting the LED
  line, though its contribution is small either way.
- A real LiPo/Li-ion cell rarely delivers 100% of its rated capacity in
  practice (cutoff voltage, aging, temperature, discharge-rate derating) -
  discount the battery-life table above by roughly 10-20% for a more
  realistic planning figure.
- Powering the board through its onboard 5 V regulator (vs. feeding 3.3 V
  directly) adds LDO overhead that's folded into the "conservative" column's
  regulator-overhead line; if the actual power architecture is different,
  adjust that line accordingly.
