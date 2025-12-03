import sounddevice as sd
import serial.tools.list_ports

def list_audio_devices():
    print("\n--- AUDIO DEVICES ---")
    print(f"{'ID':<4} {'Name':<40} {'In':<5} {'Out':<5}")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        # Filter for likely candidates (optional, but helpful)
        if 'USB' in dev['name'] or 'Built-in' in dev['name']:
            print(f"{i:<4} {dev['name'][:38]:<40} {dev['max_input_channels']:<5} {dev['max_output_channels']:<5}")

def list_serial_ports():
    print("\n--- SERIAL PORTS (for PTT) ---")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Digirig usually appears as CP210x or similar
        print(f"Port: {port.device}")
        print(f"Desc: {port.description}")
        print("-" * 20)

if __name__ == "__main__":
    print("Connect your Digirig and run this to find your Device IDs/Paths.")
    list_audio_devices()
    list_serial_ports()