# JarvisPipeline

JarvisPipeline is an offline, edge-friendly voice assistant that runs entirely locally.  
It combines:

- **Speech-to-Text (STT)** → Vosk  
- **Local LLM inference** → Ollama (any pulled model)  
- **Text-to-Speech (TTS)** → pyttsx3  

It activates using a wake word, listens for an utterance, sends the text to a local LLM, and speaks the reply back.

The pipeline is designed to run on small Linux devices (like an Orange Pi), macOS, or Binbows as long as you have a microphone, speakers, and local models.

---

## ✨ Features

- **Wake Word Detection**  
  Passive listening for “jarvis” or “hey jarvis” using a Vosk grammar.

- **Full Offline STT**  
  Converts speech to text using a Vosk model.

- **Local LLM via Ollama**  
  Runs with any installed Ollama model (e.g. `smollm:135m`, `qwen2.5:0.5b`, `llama3.2:1b`).

- **TTS Output**  
  Speaks responses using pyttsx3 with system voices.

- **Stateless**  
  Each interaction is processed independently.

- **Edge-Optimized**  
  Requires only local compute and no network access.

---

## 🧠 How It Works

### 🔊 1. Microphone Input  
A custom `AudioStream` class collects audio using `sounddevice`, applies optional mic gain, and feeds the data into Vosk.

### 🔔 2. Wake Word Mode  
A Vosk recognizer with a small grammar listens for wake words:

- `jarvis`
- `hey jarvis`

When triggered, Jarvis acknowledges and switches modes.

### 🎙️ 3. Full Utterance STT  
A full Vosk recognizer listens up to a max duration or until silence is detected using:

- configurable RMS threshold  
- silence duration cutoff  

This produces the final transcript.

### 🧠 4. LLM Request  
The transcript is sent to Ollama’s `/api/chat` endpoint with a system prompt tuned for:

- short replies  
- speech-friendly phrasing  
- no markdown  

### 🔊 5. TTS Reply  
Jarvis speaks the response using pyttsx3.

### 🔁 6. Loop  
After speaking, Jarvis returns to wake word mode.

---

## ⚙️ Requirements

### System
- Python 3.11.x 
- Microphone + speakers  
- Linux, macOS, or Binbows  

### Python Dependencies

Install with:

```bash
pip install sounddevice numpy vosk pyttsx3 requests
```

### External Tools

#### 🧠 Ollama (local LLM)
Install from: https://ollama.com  
Pull a model (example):

```bash
ollama pull smollm:135m
```

Make sure Ollama is running.

#### 🗣️ Vosk Model  
Download a model such as:  
`vosk-model-small-en-us-0.15`

Set its path in the script:

```python
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"
```

---

## 🔧 Configuration Overview

Key options at the top of the file:

```python
WAKE_WORDS = ["jarvis", "hey jarvis"]

ACTIVE_MODEL = "smollm:135m"
OLLAMA_URL = "http://localhost:11434/api/chat"

SILENCE_THRESHOLD = 0.004
SILENCE_DURATION_END = 2.5

TTS_RATE = 180
TTS_VOLUME = 1.0

MIC_GAIN = 2.0
```

You can customize:

- Wake words  
- Which LLM to use  
- Silence detection sensitivity  
- Mic gain  
- Voice settings  
- System prompt  

---

## ▶️ Running JarvisPipeline

1. Start Ollama and ensure your model is pulled.

2. Confirm your Vosk model path is correct.

3. Run:

```bash
python jarvis_pipeline.py
```

4. You should hear:

> “Jarvis online. Say my name when you need me.”

5. Say “Jarvis” or “Hey Jarvis,” then ask your question.

6. Stop anytime with **Ctrl + C**.

---

## 📁 Project Structure

```
jarvis_pipeline.py      # Main pipeline script
models/
  vosk-model-...        # Vosk STT model directory
README.md               # Project documentation
```

---

## 🌱 Future Improvements
 
- Optional short-term memory   
- LED/audio indicators for states  
- CLI flags for switching LLM models  

---

## 📌 Notes

- Runs fully locally; no internet required after models are installed.  
- Performance depends on microphone quality, Vosk model size, and selected LLM.  
- pyttsx3 behavior varies across platforms.
