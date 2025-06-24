import asyncio
import matplotlib.pyplot as plt
from bleak import BleakClient, BleakScanner
from collections import deque
import threading
from matplotlib.animation import FuncAnimation

# Buffer size for plotting
BUFFER_SIZE = 100

# Plot objects
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
line_gyro_x, = ax1.plot([], [], 'r-', label='Gyro X')
line_gyro_y, = ax1.plot([], [], 'g-', label='Gyro Y')
line_gyro_z, = ax1.plot([], [], 'b-', label='Gyro Z')
line_accel_x, = ax2.plot([], [], 'r-', label='Accel X')
line_accel_y, = ax2.plot([], [], 'g-', label='Accel Y')
line_accel_z, = ax2.plot([], [], 'b-', label='Accel Z')

class SmartCane:
    def __init__(self):
        self.sample_count = 0
        self.gyro_buffer = deque(maxlen=BUFFER_SIZE)   # Stores tuples (x, y, z)
        self.accel_buffer = deque(maxlen=BUFFER_SIZE)  # Stores tuples (x, y, z)
        self.uuid_service = "eb7f25c3-8d96-4311-92c9-45e90f6b6f5b"
        self.uuid_char_gyro = "a94090de-f49a-49f4-97c0-a95abc6cbb95"
        self.uuid_char_accel = "ca52c70a-3eb6-4043-add4-df23393e387f"
        self.is_connected = False
        self.lock = threading.Lock()

    def set_gyro_data(self, data):
        try:
            values = [float(x.strip()) for x in data.split(",")]
            if len(values) >= 3:
                with self.lock:
                    self.gyro_buffer.append((values[0], values[1], values[2]))
                    self.sample_count += 1
        except Exception as e:
            print(f"Error parsing gyro data: {data}, Error: {e}")

    def set_accel_data(self, data):
        try:
            values = [float(x.strip()) for x in data.split(",")]
            if len(values) >= 3:
                with self.lock:
                    self.accel_buffer.append((values[0], values[1], values[2]))
                    self.sample_count += 1
        except Exception as e:
            print(f"Error parsing accel data: {data}, Error: {e}")

smartCane = SmartCane()

def initialize_plot():
    ax1.set_ylim(-100, 100)
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Gyroscope (deg/s)")
    ax1.set_title("Live Gyroscope Data (X, Y, Z)")
    ax1.grid(True)
    ax1.legend()

    ax2.set_ylim(-20, 20)
    ax2.set_xlabel("Sample")
    ax2.set_ylabel("Accelerometer (g)")
    ax2.set_title("Live Accelerometer Data (X, Y, Z)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

def animate_plot(frame):
    with smartCane.lock:
        if smartCane.gyro_buffer and smartCane.accel_buffer:
            x_vals = list(range(len(smartCane.gyro_buffer)))

            gyro_x = [g[0] for g in smartCane.gyro_buffer]
            gyro_y = [g[1] for g in smartCane.gyro_buffer]
            gyro_z = [g[2] for g in smartCane.gyro_buffer]

            accel_x = [a[0] for a in smartCane.accel_buffer]
            accel_y = [a[1] for a in smartCane.accel_buffer]
            accel_z = [a[2] for a in smartCane.accel_buffer]

            # Update lines
            line_gyro_x.set_data(x_vals, gyro_x)
            line_gyro_y.set_data(x_vals, gyro_y)
            line_gyro_z.set_data(x_vals, gyro_z)
            ax1.set_xlim(0, max(BUFFER_SIZE, len(x_vals)))

            line_accel_x.set_data(x_vals, accel_x)
            line_accel_y.set_data(x_vals, accel_y)
            line_accel_z.set_data(x_vals, accel_z)
            ax2.set_xlim(0, max(BUFFER_SIZE, len(x_vals)))

            # Auto-scale Y
            if len(gyro_x) > 1:
                gyro_all = gyro_x + gyro_y + gyro_z
                margin = max((max(gyro_all) - min(gyro_all)) * 0.1, 1.0)
                ax1.set_ylim(min(gyro_all) - margin, max(gyro_all) + margin)

            if len(accel_x) > 1:
                accel_all = accel_x + accel_y + accel_z
                margin = max((max(accel_all) - min(accel_all)) * 0.1, 0.5)
                ax2.set_ylim(min(accel_all) - margin, max(accel_all) + margin)

    return line_gyro_x, line_gyro_y, line_gyro_z, line_accel_x, line_accel_y, line_accel_z

def handle_notification_gyro(sender, data):
    try:
        data_str = data.decode('utf-8').strip()
        smartCane.set_gyro_data(data_str)
    except Exception as e:
        print(f"Gyro notification error: {e}")

def handle_notification_accel(sender, data):
    try:
        data_str = data.decode('utf-8').strip()
        smartCane.set_accel_data(data_str)
    except Exception as e:
        print(f"Accel notification error: {e}")

async def ble_task(device_address, device_name):

    print(f"Connecting to {device_name} [{device_address}]...")
    try:
        async with BleakClient(device_address, timeout=20.0) as client:
            if not client.is_connected:
                print("Failed to connect.")
                return

            smartCane.is_connected = True
            services = await client.get_services()

            gyro_char = None
            accel_char = None
            for service in services:
                print(f"services found: {service.uuid}")
                if str(service.uuid).lower() == smartCane.uuid_service.lower():
                    print(f"Connected to service: {service.uuid}")
                    print("Searching for characteristics...")
                    print(f"characteristics: {[str(char.uuid) for char in service.characteristics]} ")
                    for char in service.characteristics:
                        if str(char.uuid).lower() == smartCane.uuid_char_gyro.lower():
                            gyro_char = char
                        elif str(char.uuid).lower() == smartCane.uuid_char_accel.lower():
                            accel_char = char

            if not gyro_char or not accel_char:
                print("Required characteristics not found!")
                return

            await client.start_notify(gyro_char.uuid, handle_notification_gyro)
            await client.start_notify(accel_char.uuid, handle_notification_accel)
            print("Notifications started. Receiving data...")

            while smartCane.is_connected and plt.get_fignums():
                await asyncio.sleep(1)

            await client.stop_notify(gyro_char.uuid)
            await client.stop_notify(accel_char.uuid)
            smartCane.is_connected = False

    except Exception as e:
        print(f"Connection error: {e}")
        smartCane.is_connected = False

def main():
    print("Pre-scanning for BLE devices...")
    devices = asyncio.run(BleakScanner.discover(timeout=5.0))

    if not devices:
        print("No devices found.")
        return

    print("Devices:")
    for i, d in enumerate(devices):
        print(f"{i}: {d.name or 'Unknown'} [{d.address}] RSSI: {d.rssi}")

    try:
        index = int(input("Select device index to connect: "))
        if index < 0 or index >= len(devices):
            print("Invalid index.")
            return
    except ValueError:
        print("Invalid input.")
        return

    address = devices[index].address
    device_name = devices[index].name or 'Unknown'

    ble_thread = threading.Thread(target=lambda: asyncio.run(ble_task(address, device_name)), daemon=True)
    ble_thread.start()
    initialize_plot()


    anim = FuncAnimation(fig, animate_plot, interval=100, blit=False, cache_frame_data=False)
    plt.show()

    smartCane.is_connected = False
    print("Application closed.")

if __name__ == "__main__":
    main()
