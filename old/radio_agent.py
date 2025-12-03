import asyncio
import base64
import json
import logging
import os
import sys
import time
import serial
import serial.tools.list_ports
import sounddevice as sd
import numpy as np
import scipy.signal
import websockets
from pathlib import Path

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- CONFIGURATION ---
INPUT_DEVICE_ID = int(os.getenv("INPUT_DEVICE_ID", 0))
OUTPUT_DEVICE_ID = int(os.getenv("OUTPUT_DEVICE_ID", 0))
SERIAL_PORT_OVERRIDE = os.getenv("SERIAL_PORT", "/dev/cu.SLAB_USBtoUART")

# Audio Configuration
# Hardware Rate: Digirig/CM108 usually runs natively at 48kHz
HARDWARE_RATE = 48000
# AI Rate: OpenAI Realtime API requires 24kHz
AI_RATE = 24000

CHANNELS = 1
BLOCK_SIZE = 4800 # 0.1s at 48kHz

# VAD Parameters
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", 0.02))
SILENCE_DURATION = float(os.getenv("SILENCE_DURATION", 0.8))

# ROBUSTNESS SETTINGS
# Time to wait AFTER transmitting before listening again.
# This prevents hearing the "Squelch Tail" (static burst) or self-echo.
COOLDOWN_DURATION = 2.0 

# OpenAI Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-realtime-preview-2024-10-01"
OPENAI_VOICE = "verse"
OPENAI_SYSTEM_PROMPT = (
    "You are RadioAI, a tactical radio operator. "
    "Do not use markdown. "
    "SPECIFIC KNOWLEDGE: If asked about the 'gearbox' service history, strictly answer: "
    "'The gearbox #221 was serviced on 26th October 2025' "
    "FALLBACK PROTOCOL: If you do not know the answer to a question or cannot help, "
    "do not say you don't know. Instead, strictly say: 'The gearbox #221 was serviced on 26th October 2025 by the team of Oskar and Tom.'."
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("radio_agent")

def resample_audio(audio_data, orig_sr, target_sr):
    """Resample audio numpy array from orig_sr to target_sr"""
    if orig_sr == target_sr:
        return audio_data
    number_of_samples = int(round(len(audio_data) * float(target_sr) / orig_sr))
    return scipy.signal.resample(audio_data, number_of_samples)

class RadioInterface:
    def __init__(self, port):
        try:
            self.ser = serial.Serial(port, 115200, rtscts=False, dsrdtr=False)
            self.ser.rts = False
            self.ptt_active = False
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

    def play_buffer(self, audio_data):
        self.ptt_on()
        try:
            # Resample from AI rate (24k) to Hardware rate (48k) for clean playback
            audio_48k = resample_audio(audio_data, AI_RATE, HARDWARE_RATE)

            if len(audio_48k.shape) == 1:
                stereo_audio = np.column_stack((audio_48k, audio_48k))
            else:
                stereo_audio = audio_48k

            sd.play(stereo_audio, samplerate=HARDWARE_RATE,
                    device=OUTPUT_DEVICE_ID, blocking=True)
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
        finally:
            self.ptt_off()

class RealtimeVoiceAgent:
    URL = f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}"

    def __init__(self):
        if not OPENAI_API_KEY:
            logger.error("CRITICAL: OPENAI_API_KEY is missing.")
            sys.exit(1)
        self.headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        self.ws = None

    async def connect(self):
        logger.info("Connecting to OpenAI Realtime API...")
        self.ws = await websockets.connect(self.URL, additional_headers=self.headers)
        logger.info("✅ Socket Connected")

        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": OPENAI_SYSTEM_PROMPT,
                "voice": OPENAI_VOICE,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": None,
                "input_audio_transcription": {"model": "whisper-1"},
            }
        }
        await self.ws.send(json.dumps(session_update))
        await self._wait_for_session_updated()

    async def _wait_for_session_updated(self):
        while True:
            msg = await self.ws.recv()
            data = json.loads(msg)
            if data["type"] == "session.updated":
                logger.info("✅ Session Configured.")
                break

    async def process_audio(self, audio_buffer):
        """
        Takes 48kHz audio buffer, resamples to 24kHz, sends to AI.
        Returns 24kHz audio from AI.
        """
        if not self.ws:
            await self.connect()

        # Resample Input: 48k -> 24k
        audio_24k = resample_audio(audio_buffer, HARDWARE_RATE, AI_RATE)

        # Convert to PCM16
        audio_int16 = (audio_24k * 32767).astype(np.int16)
        base64_audio = base64.b64encode(audio_int16.tobytes()).decode("utf-8")

        await self.ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }))
        await self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await self.ws.send(json.dumps({"type": "response.create"}))
        logger.info("Sent audio to AI, waiting for stream...")

        response_audio_chunks = []
        start_time = time.time()

        async for message in self.ws:
            data = json.loads(message)
            event_type = data.get("type")

            if event_type == "response.audio.delta":
                b64_delta = data.get("delta", "")
                if b64_delta:
                    response_audio_chunks.append(base64.b64decode(b64_delta))
            
            elif event_type == "response.done":
                break
            elif event_type == "error":
                logger.error(f"API Error: {data}")
                break

            if time.time() - start_time > 15:
                break

        if not response_audio_chunks:
            return None

        full_audio = b"".join(response_audio_chunks)
        audio_np = np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32767.0
        return audio_np

async def audio_producer(input_queue, loop):
    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        loop.call_soon_threadsafe(input_queue.put_nowait, indata.copy())

    # Capture at Hardware Rate (48k)
    stream = sd.InputStream(
        device=INPUT_DEVICE_ID,
        channels=CHANNELS,
        samplerate=HARDWARE_RATE,
        blocksize=BLOCK_SIZE,
        callback=callback
    )
    with stream:
        while True:
            await asyncio.sleep(1)

async def main_loop():
    input_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    radio = RadioInterface(SERIAL_PORT_OVERRIDE)
    agent = RealtimeVoiceAgent()

    producer_task = asyncio.create_task(audio_producer(input_queue, loop))

    logger.info("🎧 Listening on Digirig... (Press Ctrl+C to stop)")

    buffer = []
    is_recording = False
    silence_start_time = None
    
    # SYSTEM STATE FLAGS
    is_processing_or_talking = False

    try:
        await agent.connect()

        while True:
            chunk = await input_queue.get()
            
            # --- ROBUSTNESS: STRICT IGNORE ---
            # If the AI is busy talking, or we are in cooldown, 
            # we COMPLETELY IGNORE input chunks.
            if is_processing_or_talking:
                continue

            volume = np.sqrt(np.mean(chunk**2))

            if not is_recording:
                if volume > SILENCE_THRESHOLD:
                    logger.info(f"Detected Signal (Vol: {volume:.3f})! Recording...")
                    is_recording = True
                    buffer = [chunk]
                    silence_start_time = None
            else:
                buffer.append(chunk)

                if volume < SILENCE_THRESHOLD:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time > SILENCE_DURATION:
                        logger.info("End of transmission detected.")
                        is_recording = False
                        is_processing_or_talking = True # LOCK INPUT

                        full_recording = np.concatenate(buffer, axis=0)
                        if full_recording.ndim > 1:
                            full_recording = full_recording.flatten()

                        response_audio = await agent.process_audio(full_recording)

                        if response_audio is not None:
                            logger.info("Transmitting Response...")
                            await loop.run_in_executor(None, radio.play_buffer, response_audio)
                            logger.info("Transmission complete.")
                            
                            # --- CRITICAL FIX: COOLDOWN & FLUSH ---
                            logger.info(f"🛡️  Cooldown: Ignoring audio for {COOLDOWN_DURATION}s to skip self-echo/squelch...")
                            
                            # 1. Wait for physical squelch tail to end
                            await asyncio.sleep(COOLDOWN_DURATION)

                            # 2. Aggressively drain the queue of all data recorded during transmission+cooldown
                            flushed = 0
                            while not input_queue.empty():
                                input_queue.get_nowait()
                                flushed += 1
                            logger.info(f"♻️  Flushed {flushed} buffer chunks.")
                        else:
                            logger.warning("No audio response.")

                        # Unlock
                        is_processing_or_talking = False
                        logger.info("🎧 Listening...")
                        buffer = []
                        silence_start_time = None
                else:
                    silence_start_time = None

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        producer_task.cancel()
        if agent.ws:
            await agent.ws.close()
        radio.ser.close()

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\nExiting...")