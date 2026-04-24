"""
IMU Logger + Live Dashboard
============================
Flask server that receives JSON POSTs from the M5StickS3,
stores every sample in SQLite, and shows a live matplotlib dashboard.

Sensors displayed:
- Accelerometer (X, Y, Z)
- Gyroscope (X, Y, Z)
- Temperature, Battery %

Requires: flask, matplotlib, numpy
"""

import logging
import sys
import time
import sqlite3
import os
from datetime import datetime
from collections import deque
import threading

import numpy as np

from flask import Flask, request, jsonify

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m',
        'ERROR': '\033[31m', 'CRITICAL': '\033[35m', 'RESET': '\033[0m',
    }
    def format(self, record):
        c = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        r = self.COLORS['RESET']
        record.levelname = f"{c}{record.levelname}{r}"
        record.msg = f"{c}{record.msg}{r}"
        return super().format(record)

logger = logging.getLogger('IMULogger')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(ColoredFormatter(
    '%(asctime)s.%(msecs)03d | %(levelname)-18s | %(message)s', datefmt='%H:%M:%S'
))
logger.addHandler(console_handler)

file_handler = logging.FileHandler(
    f'imu_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(funcName)-25s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
))
logger.addHandler(file_handler)

# =============================================================================
# SQLITE SETUP
# =============================================================================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imu_logs.db")

def init_db():
    """Create the database and table if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS imu_data (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   INTEGER,
            accel_x     REAL,
            accel_y     REAL,
            accel_z     REAL,
            gyro_x      REAL,
            gyro_y      REAL,
            gyro_z      REAL,
            temp        REAL,
            bat         REAL,
            received_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"SQLite database ready at {DB_PATH}")

def insert_sample(data: dict):
    """Insert one IMU sample into the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO imu_data
            (timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temp, bat)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("timestamp"),
        data.get("accel_x"), data.get("accel_y"), data.get("accel_z"),
        data.get("gyro_x"),  data.get("gyro_y"),  data.get("gyro_z"),
        data.get("temp"),    data.get("bat"),
    ))
    conn.commit()
    conn.close()

# =============================================================================
# IN-MEMORY SENSOR DATA (for live dashboard)
# =============================================================================
BUFFER_SIZE = 200

class SensorData:
    """Thread-safe ring-buffer storage fed by the Flask route."""

    def __init__(self):
        self.lock = threading.Lock()
        self.accel_buffer = deque(maxlen=BUFFER_SIZE)
        self.gyro_buffer  = deque(maxlen=BUFFER_SIZE)

        self.temperature = None
        self.battery     = None

        self.sample_count = 0
        self.start_time   = time.time()
        self.is_receiving = False
        self.last_rx_time = None

    def push(self, data: dict):
        with self.lock:
            ax = data.get("accel_x", 0)
            ay = data.get("accel_y", 0)
            az = data.get("accel_z", 0)
            gx = data.get("gyro_x", 0)
            gy = data.get("gyro_y", 0)
            gz = data.get("gyro_z", 0)

            self.accel_buffer.append((ax, ay, az))
            self.gyro_buffer.append((gx, gy, gz))

            if data.get("temp") is not None:
                self.temperature = data["temp"]
            if data.get("bat") is not None:
                self.battery = data["bat"]

            self.sample_count += 1
            self.is_receiving = True
            self.last_rx_time = time.time()

    def get_stats(self):
        with self.lock:
            return {
                'samples': self.sample_count,
                'receiving': self.is_receiving,
                'uptime': time.time() - self.start_time,
            }

# Global instance shared between Flask route and dashboard
sensor_data = SensorData()

# =============================================================================
# FLASK APP
# =============================================================================
app = Flask(__name__)
# Suppress default Flask request logs (they're noisy at 10 Hz)
log_werkzeug = logging.getLogger('werkzeug')
log_werkzeug.setLevel(logging.WARNING)

@app.route("/log", methods=["POST"])
def log_data():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "no JSON body"}), 400
    try:
        insert_sample(data)
        sensor_data.push(data)
        if sensor_data.sample_count % 50 == 0:
            logger.info(f"Sample #{sensor_data.sample_count}: "
                        f"ax={data.get('accel_x',0):+.3f} "
                        f"ay={data.get('accel_y',0):+.3f} "
                        f"az={data.get('accel_z',0):+.3f}")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        logger.error(f"Error processing sample: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    stats = sensor_data.get_stats()
    return jsonify(stats), 200

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "SmartCane IMU Logger Server",
        "version": "1.0",
        "status": "running"
    }), 200
    
#show data
@app.route("/dashboard", methods=["GET"])
def dashboard():
    with sensor_data.lock:
        accel = list(sensor_data.accel_buffer)
        gyro  = list(sensor_data.gyro_buffer)
        temp  = sensor_data.temperature
        bat   = sensor_data.battery

    return jsonify({
        "accel": accel,
        "gyro": gyro,
        "temp": temp,
        "bat": bat,
    }), 200
    

def main():
    logger.info("=" * 60)
    logger.info("IMU LOGGER SERVER + DASHBOARD")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python {sys.version}")

    init_db()


    app.run('0.0.0.0', 5001, debug=True, use_reloader=False)
    
    stats = sensor_data.get_stats()
    logger.info("=" * 60)
    logger.info("SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples:  {stats['samples']}")
    logger.info(f"Total uptime:   {stats['uptime']:.1f} s")
    logger.info(f"Database:       {DB_PATH}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
