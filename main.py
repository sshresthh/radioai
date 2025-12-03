import asyncio
import logging
import sys
import time
import serial
import serial.tools.list_ports
import sounddevice as sd
import numpy as np
import soundfile as sf
import scipy.signal
from pathlib import Path

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main")

# Audio Configuration
HARDWARE_RATE = 48000
CHANNELS = 1
BLOCK_SIZE = 4800  # 0.1s at 48kHz

# VAD Parameters
SILENCE_THRESHOLD = 0.02
SILENCE_DURATION = 0.8

# Configuration
ANSWER_FILE = "answer.mp3"
WAIT_AFTER_QUESTION = 1.0  # Wait 1 second after detecting question
COOLDOWN_DURATION = 2.0  # Wait after transmitting to avoid self-echo/squelch


def auto_detect_serial_port():
    """Auto-detect serial port (prefer USB serial devices)"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Common USB serial device patterns
        desc_lower = port.description.lower()
        if any(keyword in desc_lower for keyword in ['usb', 'serial', 'uart', 'cp210', 'ch340', 'ftdi', 'slab']):
            logger.info(
                f"Auto-detected serial port: {port.device} ({port.description})")
            return port.device

    # Fallback to first available port
    if ports:
        logger.info(f"Using first available port: {ports[0].device}")
        return ports[0].device

    logger.error("No serial ports found!")
    sys.exit(1)


def auto_detect_audio_devices():
    """Auto-detect input and output audio devices"""
    devices = sd.query_devices()
    input_id = None
    output_id = None

    for i, dev in enumerate(devices):
        # Look for USB audio devices (like Digirig)
        dev_name = dev['name']
        if 'USB' in dev_name or 'Digirig' in dev_name:
            if dev['max_input_channels'] > 0 and input_id is None:
                input_id = i
                logger.info(f"Auto-detected input device: {i} - {dev_name}")
            if dev['max_output_channels'] > 0 and output_id is None:
                output_id = i
                logger.info(f"Auto-detected output device: {i} - {dev_name}")

    # Fallback to default devices
    if input_id is None:
        input_id = sd.default.device[0] if sd.default.device[0] is not None else 0
        logger.info(f"Using default input device: {input_id}")
    if output_id is None:
        output_id = sd.default.device[1] if sd.default.device[1] is not None else sd.default.device[0]
        logger.info(f"Using default output device: {output_id}")

    return input_id, output_id


class RadioInterface:
    def __init__(self, port, output_device_id):
        try:
            self.ser = serial.Serial(port, 115200, rtscts=False, dsrdtr=False)
            self.ser.rts = False
            self.ptt_active = False
            self.output_device_id = output_device_id
            logger.info(f"✅ Radio Interface Initialized on {port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to serial port {port}: {e}")
            sys.exit(1)

    def ptt_on(self):
        if not self.ptt_active:
            logger.info("🔴 PTT ON")
            self.ser.rts = True
            self.ptt_active = True
            time.sleep(0.3)

    def ptt_off(self):
        if self.ptt_active:
            logger.info("⚪ PTT OFF")
            self.ser.rts = False
            self.ptt_active = False

    def play_audio_file(self, audio_data, samplerate):
        """Play audio data through radio"""
        self.ptt_on()
        try:
            # Resample if needed to hardware rate
            if samplerate != HARDWARE_RATE:
                num_samples = int(
                    round(len(audio_data) * float(HARDWARE_RATE) / samplerate))
                audio_data = scipy.signal.resample(audio_data, num_samples)
                samplerate = HARDWARE_RATE

            # Ensure audio is in float32 format and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize audio to prevent clipping
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val

            # Convert to stereo if mono
            if len(audio_data.shape) == 1:
                stereo_audio = np.column_stack((audio_data, audio_data))
            else:
                stereo_audio = audio_data

            sd.play(stereo_audio, samplerate=samplerate,
                    device=self.output_device_id, blocking=True)
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
        finally:
            self.ptt_off()


def load_mp3_file(filepath):
    """Load MP3 file and return audio data and sample rate"""
    if not Path(filepath).exists():
        logger.error(f"Audio file not found: {filepath}")
        sys.exit(1)

    try:
        audio_data, samplerate = sf.read(filepath, dtype='float32')
        # If stereo, convert to mono by averaging channels
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        logger.info(
            f"Loaded {filepath} (sample rate: {samplerate} Hz, duration: {len(audio_data)/samplerate:.2f}s)")
        return audio_data, samplerate
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        sys.exit(1)


async def audio_producer(input_queue, loop, input_device_id):
    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        loop.call_soon_threadsafe(input_queue.put_nowait, indata.copy())

    stream = sd.InputStream(
        device=input_device_id,
        channels=CHANNELS,
        samplerate=HARDWARE_RATE,
        blocksize=BLOCK_SIZE,
        callback=callback
    )
    with stream:
        while True:
            await asyncio.sleep(1)


async def main_loop():
    # Auto-detect devices
    serial_port = auto_detect_serial_port()
    input_device_id, output_device_id = auto_detect_audio_devices()

    # Load answer.mp3
    answer_audio, answer_samplerate = load_mp3_file(ANSWER_FILE)

    # Initialize radio interface
    radio = RadioInterface(serial_port, output_device_id)

    input_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    producer_task = asyncio.create_task(
        audio_producer(input_queue, loop, input_device_id))

    logger.info("🎧 Listening for questions... (Press Ctrl+C to stop)")

    buffer = []
    is_recording = False
    silence_start_time = None
    is_processing_or_talking = False  # Flag to ignore input during broadcast/cooldown

    try:
        while True:
            chunk = await input_queue.get()

            # Ignore input during broadcast and cooldown to avoid self-echo
            if is_processing_or_talking:
                continue

            volume = np.sqrt(np.mean(chunk**2))

            if not is_recording:
                if volume > SILENCE_THRESHOLD:
                    logger.info(
                        f"Question detected! (Vol: {volume:.3f}) Recording...")
                    is_recording = True
                    buffer = [chunk]
                    silence_start_time = None
            else:
                buffer.append(chunk)

                if volume < SILENCE_THRESHOLD:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time > SILENCE_DURATION:
                        logger.info("End of question detected.")
                        is_recording = False
                        is_processing_or_talking = True  # Lock input

                        # Wait before broadcasting
                        logger.info(
                            f"Waiting {WAIT_AFTER_QUESTION} seconds before broadcasting...")
                        await asyncio.sleep(WAIT_AFTER_QUESTION)

                        # Broadcast answer.mp3
                        logger.info("Broadcasting answer.mp3...")
                        await loop.run_in_executor(
                            None,
                            radio.play_audio_file,
                            answer_audio,
                            answer_samplerate
                        )
                        logger.info("Broadcast complete.")

                        # Cooldown period to avoid self-echo/squelch tail
                        logger.info(
                            f"🛡️  Cooldown: Ignoring audio for {COOLDOWN_DURATION}s to skip self-echo/squelch...")
                        await asyncio.sleep(COOLDOWN_DURATION)

                        # Flush input queue of any data recorded during transmission+cooldown
                        flushed = 0
                        while not input_queue.empty():
                            input_queue.get_nowait()
                            flushed += 1
                        if flushed > 0:
                            logger.info(
                                f"♻️  Flushed {flushed} buffer chunks.")

                        # Reset state to listen for next question
                        is_processing_or_talking = False
                        buffer = []
                        silence_start_time = None
                        logger.info("🎧 Listening for next question...")
                else:
                    silence_start_time = None

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        producer_task.cancel()
        radio.ser.close()
        logger.info("Exited.")

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nExiting...")
