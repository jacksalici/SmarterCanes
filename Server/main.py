import asyncio
import matplotlib.pyplot as plt
from bleak import BleakClient, BleakScanner
from collections import deque

# Replace this with your BLE device's service and characteristic UUIDs
SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
CHARACTERISTIC_UUID = "87654321-4321-4321-4321-cba987654321"

# Buffer size for plotting
BUFFER_SIZE = 100

# Data storage
data_buffer = deque(maxlen=BUFFER_SIZE)
x_data = deque(maxlen=BUFFER_SIZE)

# Initialize plot


sample_count = 0

def initialize_plot():
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-')
    ax.set_ylim(0, 100)  
    ax.set_xlabel("Sample")
    ax.set_ylabel("Value")
    plt.title("Live BLE Data")

def handle_notification(sender, data):
    print(f"Notification from {sender}: {data}")
    
    """ global sample_count
    value = int(data[0])  # Adjust based on your device's data format
    sample_count += 1
    x_data.append(sample_count)
    data_buffer.append(value)

    line.set_xdata(x_data)
    line.set_ydata(data_buffer)
    ax.set_xlim(max(0, sample_count - BUFFER_SIZE), sample_count)
    plt.draw()
    plt.pause(0.001) """

async def run():
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover()
    for i, device in enumerate(devices):
        print(f"{i}: {device.name} [{device.address}]")

    index = int(input("Select device index to connect: "))
    address = devices[index].address

    async with BleakClient(address) as client:
        print(f"Connected to {devices[index].name}")
        await client.start_notify(CHARACTERISTIC_UUID, handle_notification)
        print("Receiving data... Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            await client.stop_notify(CHARACTERISTIC_UUID)

if __name__ == "__main__":
    asyncio.run(run())
