---

# 🎙️ NeuTTS Air – Voice Cloning TTS App

NeuTTS Air is a **Gradio-based** web application that allows you to generate natural-sounding speech using **instant voice cloning**.
You can provide a short reference audio clip and text to synthesize — the app will reproduce the same voice with new content.

---

## 🚀 Features

* 🗣️ **Instant voice cloning** using [NeuTTS Air](https://huggingface.co/neuphonic)
* 🔊 **Cross-platform support** (Windows, macOS, Linux)
* ⚙️ **Model management** (download, initialize, unload)
* 🤖 **Automatic speech transcription** using [OpenAI Whisper Tiny](https://github.com/openai/whisper)
* 🎚️ **Quality/Speed trade-off** with Q4 and Q8 model variants
* 🪄 **Chunked text processing** for long passages
* 🎵 **Smooth audio blending** with crossfading between segments
* 💬 **Interactive Gradio UI**

---

## 🧰 Requirements

### Python

```
Python 3.9 or later
```

### Install Dependencies

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 🪄 Setup (Platform Specific)

### 🪟 Windows

NeuTTS uses **espeak-ng** via the `phonemizer` library.
Install **espeak-ng** using one of the following:

```bash
choco install espeak
# or
scoop install espeak
```

You can also manually download from:
👉 [https://github.com/espeak-ng/espeak-ng/releases](https://github.com/espeak-ng/espeak-ng/releases)

### 🍎 macOS

```bash
brew install espeak
# or
brew install espeak-ng
```

### 🐧 Linux

Usually preinstalled; if not:

```bash
sudo apt install espeak-ng
```

---

## ⚡ Usage

### 1. Run the app

```bash
python app.py
```

The app will start a **local Gradio web server**, typically accessible at:

```
http://127.0.0.1:7860
```

### 2. In the web interface:

1. Select a model (Q4 or Q8)
2. Click **📥 Download** → **🚀 Initialize**
3. Upload a short **reference audio clip** (3–15 seconds)
4. (Optional) Provide reference text — or leave blank for auto-transcription
5. Enter new text to synthesize
6. Click **🎵 Generate Speech**

---

## 📦 Model Details

| Model               | Repository                     | Description                       |
| ------------------- | ------------------------------ | --------------------------------- |
| Q4 (Faster)         | `neuphonic/neutts-air-q4-gguf` | Lower quality, faster inference   |
| Q8 (Higher Quality) | `neuphonic/neutts-air-q8-gguf` | Slower, but more natural sounding |

Models are automatically downloaded from the Hugging Face Hub using `snapshot_download`.

---

## 🧠 How It Works

* Loads **NeuTTS Air** for text-to-speech voice cloning
* Loads **Whisper Tiny** for automatic reference transcription
* Chunks long text intelligently at sentence boundaries
* Applies **crossfading** between audio segments for smooth transitions
* Outputs a high-quality `.wav` file at 24 kHz

---

## 🧹 Cleanup

Models and CUDA memory are automatically released when:

* You click **Unload Model**
* You close the Gradio app
* You interrupt with `Ctrl + C`

---

## 🧾 License

This project is provided for **research and educational use**.
Refer to the licenses of the included models (`neuphonic/neutts-air-*` and `openai/whisper`) for their respective terms.

---

## 💡 Tips

* Use **clean reference audio** with minimal background noise.
* The **reference text** must match the spoken words in the reference clip for best results.
* For long paragraphs, the app automatically splits and processes text seamlessly.

---


