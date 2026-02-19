"""
Nicla Sense ME BLE Dashboard
============================
A Python dashboard using matplotlib to visualize sensor data from
Arduino Nicla Sense ME over Bluetooth Low Energy.

Sensors displayed:
- Accelerometer (X, Y, Z)
- Gyroscope (X, Y, Z)  
- Quaternion (X, Y, Z, W) for 3D orientation
- Temperature, Humidity, Pressure
- Air Quality (BSEC IAQ), CO2, Gas

Requires: bleak, matplotlib, numpy
"""

import asyncio
import struct
import logging
import sys
import time
from datetime import datetime
from collections import deque
import threading

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import numpy as np

from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Create custom formatter with colors for terminal
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.levelname = f"{color}{record.levelname}{reset}"
        record.msg = f"{color}{record.msg}{reset}"
        return super().format(record)

# Configure root logger
logger = logging.getLogger('NiclaDashboard')
logger.setLevel(logging.DEBUG)

# Console handler with colors
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = ColoredFormatter(
    '%(asctime)s.%(msecs)03d | %(levelname)-18s | %(message)s',
    datefmt='%H:%M:%S'
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler for detailed logs
file_handler = logging.FileHandler(f'nicla_ble_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(funcName)-25s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# =============================================================================
# BLE CONFIGURATION - Nicla Sense ME UUIDs
# =============================================================================
# UUID format: 19b10000-XXXX-537e-4f6c-d104768a1214
SERVICE_UUID = "19b10000-0000-537e-4f6c-d104768a1214"

# Characteristic UUIDs
UUID_VERSION      = "19b10000-1001-537e-4f6c-d104768a1214"
UUID_TEMPERATURE  = "19b10000-2001-537e-4f6c-d104768a1214"
UUID_HUMIDITY     = "19b10000-3001-537e-4f6c-d104768a1214"
UUID_PRESSURE     = "19b10000-4001-537e-4f6c-d104768a1214"
UUID_ACCELEROMETER= "19b10000-5001-537e-4f6c-d104768a1214"
UUID_GYROSCOPE    = "19b10000-6001-537e-4f6c-d104768a1214"
UUID_QUATERNION   = "19b10000-7001-537e-4f6c-d104768a1214"
UUID_RGB_LED      = "19b10000-8001-537e-4f6c-d104768a1214"
UUID_BSEC         = "19b10000-9001-537e-4f6c-d104768a1214"
UUID_CO2          = "19b10000-9002-537e-4f6c-d104768a1214"
UUID_GAS          = "19b10000-9003-537e-4f6c-d104768a1214"

# Data buffer size for plotting
BUFFER_SIZE = 200

# =============================================================================
# SENSOR DATA CLASS
# =============================================================================
class NiclaSensorData:
    """Thread-safe storage for all Nicla Sense ME sensor data"""
    
    def __init__(self):
        logger.info("Initializing NiclaSensorData storage")
        self.lock = threading.Lock()
        
        # Motion data buffers (high frequency - notifications)
        self.accel_buffer = deque(maxlen=BUFFER_SIZE)
        self.gyro_buffer = deque(maxlen=BUFFER_SIZE)
        self.quat_buffer = deque(maxlen=BUFFER_SIZE)
        
        # Environmental data (read on demand)
        self.temperature = None
        self.humidity = None
        self.pressure = None
        self.bsec_iaq = None
        self.co2 = None
        self.gas = None
        
        # Statistics
        self.accel_count = 0
        self.gyro_count = 0
        self.quat_count = 0
        self.last_accel_time = None
        self.last_gyro_time = None
        self.last_quat_time = None
        
        # Connection state
        self.is_connected = False
        self.connection_time = None
        
        logger.debug(f"Data buffers initialized with size {BUFFER_SIZE}")
    
    def update_accelerometer(self, x, y, z):
        """Update accelerometer data"""
        with self.lock:
            self.accel_buffer.append((x, y, z))
            self.accel_count += 1
            now = time.time()
            if self.last_accel_time:
                dt = now - self.last_accel_time
                if self.accel_count % 50 == 0:
                    logger.debug(f"ACCEL #{self.accel_count}: X={x:+8.3f} Y={y:+8.3f} Z={z:+8.3f} | Δt={dt*1000:.1f}ms")
            self.last_accel_time = now
    
    def update_gyroscope(self, x, y, z):
        """Update gyroscope data"""
        with self.lock:
            self.gyro_buffer.append((x, y, z))
            self.gyro_count += 1
            now = time.time()
            if self.last_gyro_time:
                dt = now - self.last_gyro_time
                if self.gyro_count % 50 == 0:
                    logger.debug(f"GYRO  #{self.gyro_count}: X={x:+8.3f} Y={y:+8.3f} Z={z:+8.3f} | Δt={dt*1000:.1f}ms")
            self.last_gyro_time = now
    
    def update_quaternion(self, x, y, z, w):
        """Update quaternion data"""
        with self.lock:
            self.quat_buffer.append((x, y, z, w))
            self.quat_count += 1
            now = time.time()
            if self.last_quat_time:
                dt = now - self.last_quat_time
                if self.quat_count % 50 == 0:
                    logger.debug(f"QUAT  #{self.quat_count}: X={x:+6.3f} Y={y:+6.3f} Z={z:+6.3f} W={w:+6.3f} | Δt={dt*1000:.1f}ms")
            self.last_quat_time = now
    
    def update_environment(self, temp=None, hum=None, press=None, iaq=None, co2=None, gas=None):
        """Update environmental sensor data"""
        with self.lock:
            if temp is not None:
                self.temperature = temp
                logger.info(f"TEMPERATURE: {temp:.1f} °C")
            if hum is not None:
                self.humidity = hum
                logger.info(f"HUMIDITY: {hum} %")
            if press is not None:
                self.pressure = press
                logger.info(f"PRESSURE: {press:.1f} hPa")
            if iaq is not None:
                self.bsec_iaq = iaq
                logger.info(f"AIR QUALITY (IAQ): {iaq:.0f}")
            if co2 is not None:
                self.co2 = co2
                logger.info(f"CO2 EQUIVALENT: {co2} ppm")
            if gas is not None:
                self.gas = gas
                logger.info(f"GAS RESISTANCE: {gas} Ω")
    
    def get_stats(self):
        """Get current statistics"""
        with self.lock:
            return {
                'accel_samples': self.accel_count,
                'gyro_samples': self.gyro_count,
                'quat_samples': self.quat_count,
                'connected': self.is_connected,
                'uptime': time.time() - self.connection_time if self.connection_time else 0
            }

# =============================================================================
# BLE NOTIFICATION HANDLERS
# =============================================================================
def parse_float_array(data, count):
    """Parse binary data as array of floats (little-endian)"""
    logger.debug(f"Parsing {len(data)} bytes as {count} floats: {data.hex()}")
    try:
        values = struct.unpack(f'<{count}f', data)
        logger.debug(f"Parsed values: {values}")
        return values
    except struct.error as e:
        logger.error(f"Failed to parse float array: {e}")
        return None

def create_accel_handler(sensor_data):
    """Create accelerometer notification handler"""
    def handler(sender, data):
        logger.debug(f"[NOTIFY] Accelerometer raw data ({len(data)} bytes): {data.hex()}")
        values = parse_float_array(data, 3)
        if values:
            sensor_data.update_accelerometer(*values)
    return handler

def create_gyro_handler(sensor_data):
    """Create gyroscope notification handler"""
    def handler(sender, data):
        logger.debug(f"[NOTIFY] Gyroscope raw data ({len(data)} bytes): {data.hex()}")
        values = parse_float_array(data, 3)
        if values:
            sensor_data.update_gyroscope(*values)
    return handler

def create_quat_handler(sensor_data):
    """Create quaternion notification handler"""
    def handler(sender, data):
        logger.debug(f"[NOTIFY] Quaternion raw data ({len(data)} bytes): {data.hex()}")
        values = parse_float_array(data, 4)
        if values:
            sensor_data.update_quaternion(*values)
    return handler

# =============================================================================
# BLE CONNECTION TASK
# =============================================================================
async def ble_connection_task(device_address, device_name, sensor_data):
    """Main BLE connection and data collection task"""
    
    logger.info("=" * 60)
    logger.info("STARTING BLE CONNECTION TASK")
    logger.info("=" * 60)
    logger.info(f"Target Device: {device_name}")
    logger.info(f"Target Address: {device_address}")
    logger.info(f"Expected Service UUID: {SERVICE_UUID}")
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            logger.info(f"Creating BleakClient... (attempt {retry_count + 1}/{max_retries})")
            logger.debug(f"Connection parameters: timeout=60.0s, address={device_address}")
            
            async with BleakClient(device_address, timeout=60.0) as client:
            
                # Check connection
                if not client.is_connected:
                    logger.error("Connection failed - client reports not connected")
                    logger.warning(f"Retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    await asyncio.sleep(2)
                    continue
            
            sensor_data.is_connected = True
            sensor_data.connection_time = time.time()
            retry_count = max_retries  # Connected successfully, exit retry loop
            
            logger.info("=" * 60)
            logger.info("CONNECTION ESTABLISHED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"MTU Size: {client.mtu_size if hasattr(client, 'mtu_size') else 'Unknown'}")
            
            # Discover services
            logger.info("")
            logger.info("DISCOVERING SERVICES AND CHARACTERISTICS...")
            logger.info("-" * 40)
            
            try:
                # On macOS, we might need to give it a moment
                await asyncio.sleep(0.5)
                services = client.services
            except Exception as e:
                logger.warning(f"Error getting services: {e}. Retrying...")
                await asyncio.sleep(1)
                services = client.services
            
            logger.info(f"Found {len(services)} services")
            
            target_service = None
            for service in services:
                logger.info(f"  Service: {service.uuid}")
                logger.debug(f"    Handle: {service.handle}")
                
                if service.uuid.lower() == SERVICE_UUID.lower():
                    target_service = service
                    logger.info(f"    >>> MATCHED TARGET SERVICE! <<<")
                
                for char in service.characteristics:
                    props = ', '.join(char.properties)
                    logger.debug(f"      Characteristic: {char.uuid}")
                    logger.debug(f"        Properties: {props}")
                    logger.debug(f"        Handle: {char.handle}")
            
            if not target_service:
                logger.warning(f"Target service {SERVICE_UUID} not found!")
                logger.warning("Will try to use characteristics by UUID directly")
            
            # Read environmental sensors
            logger.info("")
            logger.info("READING ENVIRONMENTAL SENSORS...")
            logger.info("-" * 40)
            
            # Temperature
            try:
                logger.debug(f"Reading Temperature from {UUID_TEMPERATURE}")
                data = await asyncio.wait_for(client.read_gatt_char(UUID_TEMPERATURE), timeout=5.0)
                logger.debug(f"Temperature raw: {data.hex()}")
                temp = struct.unpack('<f', data)[0]
                sensor_data.update_environment(temp=temp)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout reading temperature characteristic")
            except Exception as e:
                logger.warning(f"Failed to read temperature: {e}")
            
            # Humidity
            try:
                logger.debug(f"Reading Humidity from {UUID_HUMIDITY}")
                data = await asyncio.wait_for(client.read_gatt_char(UUID_HUMIDITY), timeout=5.0)
                logger.debug(f"Humidity raw: {data.hex()}")
                hum = struct.unpack('<I', data)[0]
                sensor_data.update_environment(hum=hum)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout reading humidity characteristic")
            except Exception as e:
                logger.warning(f"Failed to read humidity: {e}")
            
            # Pressure
            try:
                logger.debug(f"Reading Pressure from {UUID_PRESSURE}")
                data = await asyncio.wait_for(client.read_gatt_char(UUID_PRESSURE), timeout=5.0)
                logger.debug(f"Pressure raw: {data.hex()}")
                press = struct.unpack('<f', data)[0]
                sensor_data.update_environment(press=press)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout reading pressure characteristic")
            except Exception as e:
                logger.warning(f"Failed to read pressure: {e}")
            
            # BSEC Air Quality
            try:
                logger.debug(f"Reading BSEC IAQ from {UUID_BSEC}")
                data = await asyncio.wait_for(client.read_gatt_char(UUID_BSEC), timeout=5.0)
                logger.debug(f"BSEC raw: {data.hex()}")
                iaq = struct.unpack('<f', data)[0]
                sensor_data.update_environment(iaq=iaq)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout reading BSEC characteristic")
            except Exception as e:
                logger.warning(f"Failed to read BSEC: {e}")
            
            # CO2
            try:
                logger.debug(f"Reading CO2 from {UUID_CO2}")
                data = await asyncio.wait_for(client.read_gatt_char(UUID_CO2), timeout=5.0)
                logger.debug(f"CO2 raw: {data.hex()}")
                co2 = struct.unpack('<i', data)[0]
                sensor_data.update_environment(co2=co2)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout reading CO2 characteristic")
            except Exception as e:
                logger.warning(f"Failed to read CO2: {e}")
            
            # Gas
            try:
                logger.debug(f"Reading Gas from {UUID_GAS}")
                data = await asyncio.wait_for(client.read_gatt_char(UUID_GAS), timeout=5.0)
                logger.debug(f"Gas raw: {data.hex()}")
                gas = struct.unpack('<I', data)[0]
                sensor_data.update_environment(gas=gas)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout reading gas characteristic")
            except Exception as e:
                logger.warning(f"Failed to read gas: {e}")
            
            # Subscribe to motion sensor notifications
            logger.info("")
            logger.info("SUBSCRIBING TO MOTION SENSOR NOTIFICATIONS...")
            logger.info("-" * 40)
            
            # Accelerometer notifications
            try:
                logger.info(f"Subscribing to Accelerometer ({UUID_ACCELEROMETER})...")
                await asyncio.wait_for(
                    client.start_notify(UUID_ACCELEROMETER, create_accel_handler(sensor_data)),
                    timeout=5.0
                )
                logger.info("  ✓ Accelerometer notifications started")
            except asyncio.TimeoutError:
                logger.warning(f"  ✗ Timeout subscribing to accelerometer")
            except Exception as e:
                logger.error(f"  ✗ Failed to subscribe to accelerometer: {e}")
            
            # Gyroscope notifications
            try:
                logger.info(f"Subscribing to Gyroscope ({UUID_GYROSCOPE})...")
                await asyncio.wait_for(
                    client.start_notify(UUID_GYROSCOPE, create_gyro_handler(sensor_data)),
                    timeout=5.0
                )
                logger.info("  ✓ Gyroscope notifications started")
            except asyncio.TimeoutError:
                logger.warning(f"  ✗ Timeout subscribing to gyroscope")
            except Exception as e:
                logger.error(f"  ✗ Failed to subscribe to gyroscope: {e}")
            
            # Quaternion notifications
            try:
                logger.info(f"Subscribing to Quaternion ({UUID_QUATERNION})...")
                await asyncio.wait_for(
                    client.start_notify(UUID_QUATERNION, create_quat_handler(sensor_data)),
                    timeout=5.0
                )
                logger.info("  ✓ Quaternion notifications started")
            except asyncio.TimeoutError:
                logger.warning(f"  ✗ Timeout subscribing to quaternion")
            except Exception as e:
                logger.error(f"  ✗ Failed to subscribe to quaternion: {e}")
            
            logger.info("")
            logger.info("=" * 60)
            logger.info("DATA COLLECTION ACTIVE - Receiving sensor data...")
            logger.info("=" * 60)
            
            # Keep connection alive and periodically read environmental data
            read_interval = 5.0  # Read environmental data every 5 seconds
            last_read = time.time()
            
            while sensor_data.is_connected and plt.get_fignums():
                await asyncio.sleep(0.5)
                
                # Periodically read environmental sensors
                if time.time() - last_read > read_interval:
                    logger.debug("Periodic environmental sensor read...")
                    try:
                        data = await asyncio.wait_for(client.read_gatt_char(UUID_TEMPERATURE), timeout=3.0)
                        temp = struct.unpack('<f', data)[0]
                        sensor_data.update_environment(temp=temp)
                    except Exception as e:
                        logger.debug(f"Periodic temp read failed: {e}")
                    
                    try:
                        data = await asyncio.wait_for(client.read_gatt_char(UUID_HUMIDITY), timeout=3.0)
                        hum = struct.unpack('<I', data)[0]
                        sensor_data.update_environment(hum=hum)
                    except Exception as e:
                        logger.debug(f"Periodic humidity read failed: {e}")
                    
                    try:
                        data = await asyncio.wait_for(client.read_gatt_char(UUID_PRESSURE), timeout=3.0)
                        press = struct.unpack('<f', data)[0]
                        sensor_data.update_environment(press=press)
                    except Exception as e:
                        logger.debug(f"Periodic pressure read failed: {e}")
                    
                    last_read = time.time()
                    
                    # Log statistics
                    stats = sensor_data.get_stats()
                    logger.info(f"STATS: Accel={stats['accel_samples']} Gyro={stats['gyro_samples']} Quat={stats['quat_samples']} Uptime={stats['uptime']:.1f}s")
            
            # Cleanup
            logger.info("Stopping notifications...")
            try:
                await client.stop_notify(UUID_ACCELEROMETER)
                await client.stop_notify(UUID_GYROSCOPE)
                await client.stop_notify(UUID_QUATERNION)
            except Exception as e:
                logger.warning(f"Error stopping notifications: {e}")
            
            sensor_data.is_connected = False
            logger.info("BLE connection task completed")
            
        except asyncio.TimeoutError:
            logger.warning(f"Connection timeout on attempt {retry_count + 1}/{max_retries}")
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Waiting 3 seconds before retry...")
                await asyncio.sleep(3)
            else:
                logger.error("All connection attempts failed - timeout")
                sensor_data.is_connected = False
                break
        
        except BleakError as e:
            logger.warning(f"BLE Error on attempt {retry_count + 1}/{max_retries}: {e}")
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Waiting 3 seconds before retry...")
                await asyncio.sleep(3)
            else:
                logger.error("All connection attempts failed")
                sensor_data.is_connected = False
                break
        
        except Exception as e:
            logger.error(f"Unexpected error on attempt {retry_count + 1}/{max_retries}: {e}", exc_info=True)
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Waiting 3 seconds before retry...")
                await asyncio.sleep(3)
            else:
                sensor_data.is_connected = False
                break
    
    if not sensor_data.is_connected:
        logger.error("=" * 60)
        logger.error("FAILED TO CONNECT TO DEVICE")
        logger.error("=" * 60)
        logger.error("Possible causes:")
        logger.error("  1. Device is out of range")
        logger.error("  2. Device is not powered on")
        logger.error("  3. Device firmware not running or is advertising wrong service")
        logger.error("  4. macOS Bluetooth permissions issue")
        logger.error("=" * 60)

# =============================================================================
# MATPLOTLIB DASHBOARD
# =============================================================================
class NiclaDashboard:
    """Real-time matplotlib dashboard for Nicla Sense ME data"""
    
    def __init__(self, sensor_data):
        logger.info("Initializing Matplotlib Dashboard...")
        self.sensor_data = sensor_data
        
        # Create figure with grid layout
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Nicla Sense ME - Live Sensor Dashboard', fontsize=14, fontweight='bold')
        
        # Grid: 3 rows, 4 columns
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Accelerometer plot (top left, spans 2 columns)
        self.ax_accel = self.fig.add_subplot(gs[0, :2])
        self.ax_accel.set_title('Accelerometer')
        self.ax_accel.set_xlabel('Sample')
        self.ax_accel.set_ylabel('Acceleration (m/s²)')
        self.ax_accel.grid(True, alpha=0.3)
        self.line_accel_x, = self.ax_accel.plot([], [], 'r-', label='X', linewidth=1)
        self.line_accel_y, = self.ax_accel.plot([], [], 'g-', label='Y', linewidth=1)
        self.line_accel_z, = self.ax_accel.plot([], [], 'b-', label='Z', linewidth=1)
        self.ax_accel.legend(loc='upper right')
        
        # Gyroscope plot (top right, spans 2 columns)
        self.ax_gyro = self.fig.add_subplot(gs[0, 2:])
        self.ax_gyro.set_title('Gyroscope')
        self.ax_gyro.set_xlabel('Sample')
        self.ax_gyro.set_ylabel('Angular Velocity (°/s)')
        self.ax_gyro.grid(True, alpha=0.3)
        self.line_gyro_x, = self.ax_gyro.plot([], [], 'r-', label='X', linewidth=1)
        self.line_gyro_y, = self.ax_gyro.plot([], [], 'g-', label='Y', linewidth=1)
        self.line_gyro_z, = self.ax_gyro.plot([], [], 'b-', label='Z', linewidth=1)
        self.ax_gyro.legend(loc='upper right')
        
        # Quaternion plot (middle left, spans 2 columns)
        self.ax_quat = self.fig.add_subplot(gs[1, :2])
        self.ax_quat.set_title('Quaternion (Orientation)')
        self.ax_quat.set_xlabel('Sample')
        self.ax_quat.set_ylabel('Value')
        self.ax_quat.grid(True, alpha=0.3)
        self.line_quat_x, = self.ax_quat.plot([], [], 'r-', label='X', linewidth=1)
        self.line_quat_y, = self.ax_quat.plot([], [], 'g-', label='Y', linewidth=1)
        self.line_quat_z, = self.ax_quat.plot([], [], 'b-', label='Z', linewidth=1)
        self.line_quat_w, = self.ax_quat.plot([], [], 'm-', label='W', linewidth=1)
        self.ax_quat.legend(loc='upper right')
        
        # Environment gauges (middle right)
        self.ax_env = self.fig.add_subplot(gs[1, 2:])
        self.ax_env.set_title('Environmental Sensors')
        self.ax_env.axis('off')
        self.env_text = self.ax_env.text(0.5, 0.5, 'Waiting for data...',
                                          transform=self.ax_env.transAxes,
                                          ha='center', va='center',
                                          fontsize=12, family='monospace',
                                          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Status bar (bottom)
        self.ax_status = self.fig.add_subplot(gs[2, :])
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(0.5, 0.5, 'Initializing...',
                                                transform=self.ax_status.transAxes,
                                                ha='center', va='center',
                                                fontsize=11, family='monospace')
        
        logger.info("Dashboard layout created")
    
    def update(self, frame):
        """Animation update function"""
        with self.sensor_data.lock:
            # Update accelerometer
            if self.sensor_data.accel_buffer:
                x_vals = list(range(len(self.sensor_data.accel_buffer)))
                accel_x = [a[0] for a in self.sensor_data.accel_buffer]
                accel_y = [a[1] for a in self.sensor_data.accel_buffer]
                accel_z = [a[2] for a in self.sensor_data.accel_buffer]
                
                self.line_accel_x.set_data(x_vals, accel_x)
                self.line_accel_y.set_data(x_vals, accel_y)
                self.line_accel_z.set_data(x_vals, accel_z)
                
                self.ax_accel.set_xlim(0, max(BUFFER_SIZE, len(x_vals)))
                if accel_x:
                    all_vals = accel_x + accel_y + accel_z
                    margin = max(abs(max(all_vals) - min(all_vals)) * 0.1, 1)
                    self.ax_accel.set_ylim(min(all_vals) - margin, max(all_vals) + margin)
            
            # Update gyroscope
            if self.sensor_data.gyro_buffer:
                x_vals = list(range(len(self.sensor_data.gyro_buffer)))
                gyro_x = [g[0] for g in self.sensor_data.gyro_buffer]
                gyro_y = [g[1] for g in self.sensor_data.gyro_buffer]
                gyro_z = [g[2] for g in self.sensor_data.gyro_buffer]
                
                self.line_gyro_x.set_data(x_vals, gyro_x)
                self.line_gyro_y.set_data(x_vals, gyro_y)
                self.line_gyro_z.set_data(x_vals, gyro_z)
                
                self.ax_gyro.set_xlim(0, max(BUFFER_SIZE, len(x_vals)))
                if gyro_x:
                    all_vals = gyro_x + gyro_y + gyro_z
                    margin = max(abs(max(all_vals) - min(all_vals)) * 0.1, 10)
                    self.ax_gyro.set_ylim(min(all_vals) - margin, max(all_vals) + margin)
            
            # Update quaternion
            if self.sensor_data.quat_buffer:
                x_vals = list(range(len(self.sensor_data.quat_buffer)))
                quat_x = [q[0] for q in self.sensor_data.quat_buffer]
                quat_y = [q[1] for q in self.sensor_data.quat_buffer]
                quat_z = [q[2] for q in self.sensor_data.quat_buffer]
                quat_w = [q[3] for q in self.sensor_data.quat_buffer]
                
                self.line_quat_x.set_data(x_vals, quat_x)
                self.line_quat_y.set_data(x_vals, quat_y)
                self.line_quat_z.set_data(x_vals, quat_z)
                self.line_quat_w.set_data(x_vals, quat_w)
                
                self.ax_quat.set_xlim(0, max(BUFFER_SIZE, len(x_vals)))
                self.ax_quat.set_ylim(-1.1, 1.1)
            
            # Update environmental display
            env_lines = []
            if self.sensor_data.temperature is not None:
                env_lines.append(f"🌡️  Temperature: {self.sensor_data.temperature:6.1f} °C")
            if self.sensor_data.humidity is not None:
                env_lines.append(f"💧 Humidity:    {self.sensor_data.humidity:6d} %")
            if self.sensor_data.pressure is not None:
                env_lines.append(f"📊 Pressure:    {self.sensor_data.pressure:6.1f} hPa")
            if self.sensor_data.bsec_iaq is not None:
                env_lines.append(f"🌬️  Air Quality: {self.sensor_data.bsec_iaq:6.0f} IAQ")
            if self.sensor_data.co2 is not None:
                env_lines.append(f"💨 CO₂:         {self.sensor_data.co2:6d} ppm")
            if self.sensor_data.gas is not None:
                env_lines.append(f"⚗️  Gas:         {self.sensor_data.gas:6d} Ω")
            
            if env_lines:
                self.env_text.set_text('\n'.join(env_lines))
            
            # Update status bar
            stats = self.sensor_data.get_stats()
            status = f"{'🟢 CONNECTED' if stats['connected'] else '🔴 DISCONNECTED'} | "
            status += f"Uptime: {stats['uptime']:.1f}s | "
            status += f"Samples - Accel: {stats['accel_samples']} | Gyro: {stats['gyro_samples']} | Quat: {stats['quat_samples']}"
            self.status_text.set_text(status)
        
        return (self.line_accel_x, self.line_accel_y, self.line_accel_z,
                self.line_gyro_x, self.line_gyro_y, self.line_gyro_z,
                self.line_quat_x, self.line_quat_y, self.line_quat_z, self.line_quat_w,
                self.env_text, self.status_text)
    
    def run(self):
        """Start the animation"""
        logger.info("Starting dashboard animation...")
        self.anim = FuncAnimation(self.fig, self.update, interval=50, blit=False, cache_frame_data=False)
        plt.show()

# =============================================================================
# BLE DEVICE SCANNER
# =============================================================================
async def scan_for_devices(timeout=10.0):
    """Scan for BLE devices and return list"""
    logger.info("=" * 60)
    logger.info("SCANNING FOR BLE DEVICES")
    logger.info("=" * 60)
    logger.info(f"Scan timeout: {timeout} seconds")
    logger.info("Looking for devices with 'Nicla' in name...")
    
    devices = []
    nicla_devices = []
    
    def detection_callback(device, advertisement_data):
        logger.debug(f"Detected: {device.name or 'Unknown'} [{device.address}] RSSI: {advertisement_data.rssi}")
        if advertisement_data.service_uuids:
            logger.debug(f"  Service UUIDs: {advertisement_data.service_uuids}")
    
    scanner = BleakScanner(detection_callback=detection_callback)
    
    logger.info("Starting scan...")
    await scanner.start()
    await asyncio.sleep(timeout)
    await scanner.stop()
    
    devices = scanner.discovered_devices_and_advertisement_data
    
    logger.info("")
    logger.info(f"Scan complete. Found {len(devices)} devices:")
    logger.info("-" * 60)
    
    sorted_devices = []
    for device, adv_data in devices.values():
        sorted_devices.append((device, adv_data))
        
        # Check if this is a Nicla device
        is_nicla = device.name and 'nicla' in device.name.lower()
        has_service = SERVICE_UUID.lower() in [str(u).lower() for u in (adv_data.service_uuids or [])]
        
        marker = ""
        if is_nicla or has_service:
            marker = " <<< NICLA DEVICE"
            nicla_devices.append((device, adv_data))
        
        logger.info(f"  {device.name or 'Unknown':30} [{device.address}] RSSI: {adv_data.rssi:4d}{marker}")
    
    logger.info("-" * 60)
    logger.info(f"Found {len(nicla_devices)} potential Nicla device(s)")
    
    return sorted_devices, nicla_devices

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main application entry point"""
    logger.info("=" * 60)
    logger.info("NICLA SENSE ME - BLE DASHBOARD")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python version: {sys.version}")
    logger.info("")
    
    # Scan for devices
    all_devices, nicla_devices = asyncio.run(scan_for_devices(timeout=10.0))
    
    if not all_devices:
        logger.error("No BLE devices found! Make sure:")
        logger.error("  1. Bluetooth is enabled on your computer")
        logger.error("  2. The Nicla Sense ME is powered on")
        logger.error("  3. The Nicla firmware is running (green LED)")
        return
    
    # Auto-select Nicla device or ask user
    selected_device = None
    
    if len(nicla_devices) == 1:
        selected_device = nicla_devices[0][0]
        logger.info(f"Auto-selected Nicla device: {selected_device.name}")
    elif len(nicla_devices) > 1:
        logger.info("Multiple Nicla devices found. Please select:")
        for i, (dev, adv) in enumerate(nicla_devices):
            print(f"  {i}: {dev.name} [{dev.address}] RSSI: {adv.rssi}")
        try:
            idx = int(input("Enter device number: "))
            selected_device = nicla_devices[idx][0]
        except (ValueError, IndexError):
            logger.error("Invalid selection")
            return
    else:
        # No Nicla devices found, show all devices
        logger.warning("No Nicla devices detected. Showing all devices:")
        print("\nAvailable devices:")
        for i, (dev, adv) in enumerate(all_devices):
            print(f"  {i}: {dev.name or 'Unknown':30} [{dev.address}] RSSI: {adv.rssi}")
        try:
            idx = int(input("\nEnter device number to connect: "))
            selected_device = all_devices[idx][0]
        except (ValueError, IndexError):
            logger.error("Invalid selection")
            return
    
    logger.info("")
    logger.info(f"Selected device: {selected_device.name} [{selected_device.address}]")
    
    # Initialize sensor data storage
    sensor_data = NiclaSensorData()
    
    # Start BLE connection in background thread
    def run_ble():
        asyncio.run(ble_connection_task(selected_device.address, selected_device.name, sensor_data))
    
    ble_thread = threading.Thread(target=run_ble, daemon=True)
    ble_thread.start()
    logger.info("BLE connection thread started")
    
    # Give BLE time to connect before showing dashboard
    logger.info("Waiting for BLE connection...")
    time.sleep(2)
    
    # Create and run dashboard
    dashboard = NiclaDashboard(sensor_data)
    dashboard.run()
    
    # Cleanup
    sensor_data.is_connected = False
    logger.info("Dashboard closed. Shutting down...")
    
    # Final statistics
    stats = sensor_data.get_stats()
    logger.info("")
    logger.info("=" * 60)
    logger.info("SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total uptime: {stats['uptime']:.1f} seconds")
    logger.info(f"Accelerometer samples: {stats['accel_samples']}")
    logger.info(f"Gyroscope samples: {stats['gyro_samples']}")
    logger.info(f"Quaternion samples: {stats['quat_samples']}")
    logger.info("=" * 60)
    logger.info("Application terminated")

if __name__ == "__main__":
    main()
