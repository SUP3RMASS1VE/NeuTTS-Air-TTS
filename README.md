# 🎙️ NeuTTS Air - Voice Cloning TTS

A Gradio-based web interface for NeuTTS Air, enabling instant voice cloning and natural-sounding text-to-speech synthesis.

## Features

- **Instant Voice Cloning**: Clone any voice with just 3-15 seconds of reference audio
- **Auto-Transcription**: Automatic transcription of reference audio using Whisper
- **Multiple Model Options**: Choose between Q4 (faster) or Q8 (higher quality) models
- **Long Text Support**: Automatic text chunking with smooth crossfading for longer passages
- **Seed Control**: Reproducible generation with optional seed parameter
- **User-Friendly Interface**: Clean Gradio web UI with real-time status updates

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for best performance)
- espeak-ng (for phonemization)

## Installation

### 1. Install espeak-ng

**Windows:**
```bash
# Using Chocolatey
choco install espeak

# Using Scoop
scoop install espeak
```

**macOS:**
```bash
brew install espeak-ng
```

**Linux:**
```bash
sudo apt-get install espeak-ng
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Application

```bash
python app.py
```

The Gradio interface will launch in your browser at `http://localhost:7860`

### Using the Interface

1. **Model Setup** (first time only):
   - Select a model (Q4 for faster, Q8 for higher quality)
   - Click "Download" to download the model
   - Click "Initialize" to load the model into memory

2. **Generate Speech**:
   - Enter the text you want to synthesize
   - Upload a reference audio file (3-15 seconds, clean audio recommended)
   - The reference text will auto-transcribe, or you can provide it manually
   - (Optional) Set a seed for reproducible results
   - Click "Generate Speech"

3. **Model Management**:
   - Use "Unload" to free up memory and switch between models

## Model Options

- **Q4 (Faster, Lower Quality)**: `neuphonic/neutts-air-q4-gguf`
  - Faster inference
  - Lower memory usage
  - Good for testing and quick iterations

- **Q8 (Slower, Higher Quality)**: `neuphonic/neutts-air-q8-gguf`
  - Higher quality output
  - More memory intensive
  - Best for final production

## Tips for Best Results

- **Reference Audio**: Use 3-15 seconds of clean audio with minimal background noise
- **Reference Text**: Should match the audio exactly (or leave empty for auto-transcription)
- **Text Length**: The system automatically handles long texts by chunking and crossfading
- **Seed**: Use the same seed for consistent results across generations

## Project Structure

```
.
├── app.py                 # Main Gradio application
├── neuttsair/
│   ├── __init__.py
│   └── neutts.py         # Core NeuTTS Air implementation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Dependencies

- `gradio` - Web interface
- `torch` - Deep learning framework
- `openai-whisper` - Audio transcription
- `soundfile` - Audio I/O
- `librosa` - Audio processing
- `neucodec` - Neural audio codec
- `phonemizer` - Text to phoneme conversion
- `llama-cpp-python` - GGUF model support
- `transformers` - HuggingFace models
- `resemble-perth` - Audio processing utilities

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [NeuTTS Air](https://huggingface.co/neuphonic) by Neuphonic
- Uses [Whisper](https://github.com/openai/whisper) for transcription
- Powered by [Gradio](https://gradio.app/) for the web interface
