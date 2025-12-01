"""
JarvisPipeline: offline/edge-style voice assistant using

- Wake word + STT: Vosk
- LLM: Ollama (local model, e.g. smollm:135m, qwen2.5:0.5b, llama3.2:1b)
- TTS: pyttsx3 (system voices)

Flow:
  [mic] -> Vosk wake word -> Vosk full utterance -> Ollama -> pyttsx3

Stateless: no long-term memory, each request is independent.

Works on Linux / macOS / Binbows as long as:
  - a working microphone & speakers are configured
  - a Vosk model is downloaded
  - Ollama is running with the chosen model pulled
"""

import json
import queue
import sys
import time
import threading
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import pyttsx3
from vosk import Model, KaldiRecognizer

# ==========================
# CONFIGURATION
# ==========================

# Audio
SAMPLE_RATE = 16000          # Vosk default
CHANNELS = 1                 # mono
DEVICE = None                # None = default input device

# Path to your Vosk model directory (download separately)
# e.g. "models/vosk-model-small-en-us-0.15"
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"

# Wake words (all lowercase)
WAKE_WORDS = ["jarvis", "hey jarvis"]

# STT behavior
MAX_UTTERANCE_SECONDS = 15          # max time to listen after wake
SILENCE_THRESHOLD = 0.004           # RMS threshold for voice vs silence
SILENCE_DURATION_END = 2.5          # seconds of silence before we stop

# LLM / Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
ACTIVE_MODEL = "smollm:135m"        # change to whatever you like
LLM_TIMEOUT = 20                    # seconds

# TTS (pyttsx3)
TTS_RATE = 180                      # words per minute
TTS_VOLUME = 1.0                    # 0.0–1.0

SYSTEM_PROMPT = (
    "You are Jarvis, a concise and helpful voice assistant. "
    "You are talking to a human through speech. "
    "Respond in 1–3 short sentences without markdown or bullet points. "
    "Keep answers easy to say out loud and avoid long lists. "
    "If you are unsure, say you are not sure instead of inventing details."
)

# Mic gain (software amplification; 1.0 = none)
MIC_GAIN = 2.0


# ==========================
# TTS (pyttsx3)
# ==========================

def speak(text: str) -> None:
    """
    Speak text out loud using pyttsx3.

    We re-init the engine per call because that tends to be the most
    robust approach on Binbows; overhead is tiny compared to STT/LLM.
    """
    text = (text or "").strip()
    if not text:
        return

    print(f"[Jarvis speaking]: {text}")

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[TTS ERROR] {e}", file=sys.stderr)


# ==========================
# LLM CLIENT (Ollama)
# ==========================

def ask_llm(user_text: str) -> Optional[str]:
    """
    Call Ollama's /api/chat with the chosen model, stateless.
    """
    payload = {
        "model": ACTIVE_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "stream": False,
        "options": {
            "temperature": 0.4,
            "num_predict": 128,
        },
    }

    try:
        resp = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message") or {}
        content = msg.get("content", "").strip()
        return content or None
    except Exception as e:
        print(f"[ERROR] LLM failed: {e}", file=sys.stderr)
        return None


# ==========================
# AUDIO STREAM
# ==========================

class AudioStream:
    """
    Simple microphone input wrapper using sounddevice.
    """

    def __init__(self,
                 samplerate: int = SAMPLE_RATE,
                 channels: int = CHANNELS,
                 device=DEVICE) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self._stop = threading.Event()

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[AUDIO STATUS] {status}", file=sys.stderr)

        # apply software mic gain
        indata = indata * MIC_GAIN
        indata = np.clip(indata, -1.0, 1.0)

        if self.channels > 1:
            indata = indata.mean(axis=1, keepdims=True)
        self.q.put(indata.copy())

    def start(self) -> None:
        if self.stream is not None:
            return
        self._stop.clear()
        print(
            f"[DEBUG] Opening InputStream: samplerate={self.samplerate}, "
            f"channels={self.channels}, device={self.device}"
        )
        try:
            self.stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                callback=self._callback,
                device=self.device,
                blocksize=0,
            )
            self.stream.start()
        except Exception as e:
            print(f"[AUDIO ERROR] Failed to open input stream: {e}", file=sys.stderr)
            raise

    def stop(self) -> None:
        self._stop.set()
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"[AUDIO STOP ERROR]: {e}", file=sys.stderr)
            self.stream = None

        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def read(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        if self._stop.is_set():
            return None
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


# ==========================
# VOSK SETUP
# ==========================

print("[INFO] Loading Vosk model...")
vosk_model = Model(VOSK_MODEL_PATH)
print("[INFO] Vosk model loaded.")

def create_wake_recognizer() -> KaldiRecognizer:
    grammar = json.dumps(WAKE_WORDS)
    return KaldiRecognizer(vosk_model, SAMPLE_RATE, grammar)

def create_full_recognizer() -> KaldiRecognizer:
    return KaldiRecognizer(vosk_model, SAMPLE_RATE)


# ==========================
# WAKE WORD
# ==========================

def contains_wake_word(text: str) -> bool:
    t = text.lower()
    for w in WAKE_WORDS:
        if w in t:
            return True
    return False

def wait_for_wake_word(audio_stream: AudioStream) -> None:
    recognizer = create_wake_recognizer()
    print("[INFO] Waiting for wake word...")

    while True:
        data = audio_stream.read(timeout=1.0)
        if data is None:
            continue

        pcm = (data * 32767).astype(np.int16).tobytes()

        if recognizer.AcceptWaveform(pcm):
            try:
                text = json.loads(recognizer.Result()).get("text", "")
            except json.JSONDecodeError:
                text = ""
            if text:
                print(f"[Wake STT]: {text}")
            if contains_wake_word(text):
                print("[INFO] Wake word detected!")
                return


# ==========================
# FULL UTTERANCE STT
# ==========================

def listen_for_utterance(audio_stream: AudioStream) -> Optional[str]:
    recognizer = create_full_recognizer()
    print("[INFO] Listening for utterance...")

    start_time = time.time()
    last_voice_time = start_time
    heard_any_voice = False
    transcript = ""

    while True:
        now = time.time()
        if now - start_time > MAX_UTTERANCE_SECONDS:
            print("[INFO] Max utterance duration reached.")
            break

        data = audio_stream.read(timeout=1.0)
        if data is None:
            continue

        # RMS-based silence detection
        rms = float(np.sqrt(np.mean(np.square(data))))
        is_voice = rms > SILENCE_THRESHOLD
        if is_voice:
            heard_any_voice = True
            last_voice_time = now

        pcm = (data * 32767).astype(np.int16).tobytes()

        if recognizer.AcceptWaveform(pcm):
            try:
                part = json.loads(recognizer.Result()).get("text", "")
            except json.JSONDecodeError:
                part = ""
            if part:
                transcript = (transcript + " " + part).strip()

        # Only start checking for end-of-speech after we've heard some voice
        if heard_any_voice and (now - last_voice_time) > SILENCE_DURATION_END:
            print("[INFO] Detected end of speech by silence.")
            break

    # Final result merge
    try:
        final_text = json.loads(recognizer.FinalResult()).get("text", "")
    except json.JSONDecodeError:
        final_text = ""

    if final_text:
        transcript = (transcript + " " + final_text).strip()

    if transcript:
        print(f"[User]: {transcript}")
        return transcript
    else:
        print("[INFO] No speech recognized.")
        return None


# ==========================
# MAIN LOOP
# ==========================

def main() -> None:
    audio_stream = AudioStream()
    audio_stream.start()

    speak("Jarvis online. Say my name when you need me.")

    try:
        while True:
            # 1) Wait for wake word
            wait_for_wake_word(audio_stream)

            # 2) Acknowledge
            speak("Yes?")

            # 3) Capture utterance
            user_text = listen_for_utterance(audio_stream)
            if not user_text:
                speak("I didn't catch that. Please try again.")
                continue

            # 4) Send to LLM
            reply = ask_llm(user_text)
            if not reply:
                speak("I'm having trouble thinking right now.")
                continue

            # 5) Speak reply
            speak(reply)

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down JarvisPipeline.")
    finally:
        audio_stream.stop()


if __name__ == "__main__":
    main()
