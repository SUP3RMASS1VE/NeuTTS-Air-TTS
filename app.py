import gradio as gr
import soundfile as sf
import tempfile
import os
import sys
import platform
from pathlib import Path
import warnings
import logging

# Suppress redirect warning on Windows/MacOS
warnings.filterwarnings("ignore", message="Redirects are currently not supported")
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "OFF"
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)

import torch
import whisper
import numpy as np
import signal
import atexit

# Configure espeak for cross-platform compatibility
def setup_espeak():
    """Setup espeak library path based on the operating system."""
    system = platform.system()
    
    if system == "Windows":
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        
        # Common Windows installation paths
        possible_paths = [
            r"C:\Program Files\eSpeak NG\libespeak-ng.dll",
            r"C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll",
            Path.home() / "scoop" / "apps" / "espeak" / "current" / "libespeak.dll",
            r"C:\ProgramData\chocolatey\lib\espeak\tools\libespeak.dll",
        ]
        
        # Try to find espeak
        for path in possible_paths:
            if Path(path).exists():
                EspeakWrapper.set_library(str(path))
                print(f"✓ Found espeak at: {path}")
                return True
        
        # If not found, show helpful error
        print("\n" + "="*60)
        print("ERROR: espeak-ng not found on your system!")
        print("="*60)
        print("\nPlease install espeak-ng:")
        print("  • Using Chocolatey: choco install espeak")
        print("  • Using Scoop: scoop install espeak")
        print("  • Manual download: https://github.com/espeak-ng/espeak-ng/releases")
        print("\nAfter installation, restart your terminal and try again.")
        print("="*60 + "\n")
        sys.exit(1)
    
    elif system == "Darwin":  # macOS
        # macOS users might need to set the library path
        try:
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
            homebrew_paths = [
                "/opt/homebrew/lib/libespeak-ng.dylib",
                "/usr/local/lib/libespeak-ng.dylib",
                "/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib",
            ]
            for path in homebrew_paths:
                if Path(path).exists():
                    EspeakWrapper.set_library(path)
                    print(f"✓ Found espeak at: {path}")
                    return True
        except:
            pass
        
        # If not found on macOS, show instructions
        print("\n" + "="*60)
        print("Note: If you encounter espeak errors on macOS:")
        print("="*60)
        print("\nInstall espeak with Homebrew:")
        print("  brew install espeak")
        print("  or")
        print("  brew install espeak-ng")
        print("="*60 + "\n")
    
    # Linux and other systems usually work out of the box
    # If there are issues, the phonemizer will show its own error
    return True

# Setup espeak before importing NeuTTSAir
setup_espeak()

from neuttsair.neutts import NeuTTSAir
from huggingface_hub import snapshot_download


# Global variables for models
tts = None
whisper_model = None


def cleanup_models():
    """Clean up models before exit to prevent cleanup errors."""
    global tts, whisper_model
    
    try:
        if tts is not None:
            print("Cleaning up TTS model...")
            # Try to explicitly close the model if it has a close method
            if hasattr(tts, 'close'):
                tts.close()
            # Delete the reference
            del tts
            tts = None
    except Exception as e:
        print(f"Error during TTS cleanup: {e}")
    
    try:
        if whisper_model is not None:
            print("Cleaning up Whisper model...")
            del whisper_model
            whisper_model = None
    except Exception as e:
        print(f"Error during Whisper cleanup: {e}")
    
    # Force garbage collection and clear CUDA cache
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")
    except Exception as e:
        print(f"Error clearing cache: {e}")


def signal_handler(signum, frame):
    """Handle keyboard interrupt gracefully."""
    print("\nKeyboard interruption in main thread... closing server.")
    cleanup_models()
    sys.exit(0)


# Register cleanup handlers
atexit.register(cleanup_models)
signal.signal(signal.SIGINT, signal_handler)

# Model configurations
MODELS = {
    "Q4 (Faster, Lower Quality)": "neuphonic/neutts-air-q4-gguf",
    "Q8 (Slower, Higher Quality)": "neuphonic/neutts-air-q8-gguf"
}


def download_model(model_choice):
    """Download the selected model."""
    try:
        repo_id = MODELS[model_choice]
        print(f"Downloading {model_choice} from {repo_id}...")
        snapshot_download(repo_id=repo_id, allow_patterns=["*.gguf"])
        return f"✅ {model_choice} downloaded successfully!"
    except Exception as e:
        return f"❌ Download failed: {str(e)}"


def initialize_models(model_choice):
    """Initialize TTS and Whisper models."""
    global tts, whisper_model
    
    try:
        if tts is not None:
            return "⚠️ Model already loaded. Use 'Unload Model' to switch models."
        
        repo_id = MODELS[model_choice]
        print(f"Initializing {model_choice}...")
        
        tts = NeuTTSAir(
            backbone_repo=repo_id,
            backbone_device="cuda" if torch.cuda.is_available() else "cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print("Loading Whisper Tiny for auto-transcription...")
        whisper_model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
        
        return f"✅ {model_choice} loaded successfully!"
    except Exception as e:
        return f"❌ Initialization failed: {str(e)}"


def unload_models():
    """Unload models to free up memory."""
    global tts, whisper_model
    
    try:
        if tts is None and whisper_model is None:
            return "⚠️ No models are currently loaded."
        
        cleanup_models()
        
        return "✅ Models unloaded successfully! You can now load a different model."
    except Exception as e:
        return f"❌ Unload failed: {str(e)}"


def transcribe_audio(audio_path):
    """Transcribe audio using Whisper Tiny."""
    try:
        print(f"Transcribing audio: {audio_path}")
        result = whisper_model.transcribe(audio_path, language="en")
        transcription = result["text"].strip()
        print(f"Transcription: {transcription}")
        return transcription
    except Exception as e:
        print(f"Transcription error: {e}")
        return None


def auto_transcribe_on_upload(audio_path):
    """Auto-transcribe when audio is uploaded."""
    if audio_path is None:
        return ""
    
    print("🎤 Audio uploaded, transcribing...")
    transcription = transcribe_audio(audio_path)
    
    if transcription:
        return transcription
    else:
        return ""


def split_text_into_chunks(text, max_chunk_size=150):
    """Split text into chunks at sentence boundaries."""
    import re
    
    # Split by sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max size and we have content
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def crossfade_audio(wav1, wav2, fade_samples=480):
    """Crossfade between two audio segments to smooth transitions."""
    if len(wav1) < fade_samples or len(wav2) < fade_samples:
        # If segments too short, just concatenate
        return np.concatenate([wav1, wav2])
    
    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    
    # Apply crossfade
    wav1_end = wav1[:-fade_samples]
    wav1_fade = wav1[-fade_samples:] * fade_out
    wav2_fade = wav2[:fade_samples] * fade_in
    wav2_start = wav2[fade_samples:]
    
    crossfaded = wav1_fade + wav2_fade
    
    return np.concatenate([wav1_end, crossfaded, wav2_start])


def generate_speech(input_text, ref_audio, ref_text, seed):
    """Generate speech from text using reference audio and text."""
    
    if tts is None:
        return None, "", "⚠️ Please initialize a model first using the Model Setup section."
    
    if not input_text or not input_text.strip():
        return None, "", "Please enter text to synthesize."
    
    if ref_audio is None:
        return None, "", "Please upload a reference audio file."
    
    ref_audio_path = ref_audio
    
    # Auto-transcribe if no reference text provided
    if not ref_text or not ref_text.strip():
        print("No reference text provided, auto-transcribing...")
        ref_text_content = transcribe_audio(ref_audio_path)
        if not ref_text_content:
            return None, "", "❌ Failed to transcribe audio. Please provide reference text manually."
    else:
        ref_text_content = ref_text
    
    try:
        # Encode reference audio
        print(f"Encoding reference audio: {ref_audio_path}")
        ref_codes = tts.encode_reference(ref_audio_path)
        
        # Split text into chunks if needed
        chunks = split_text_into_chunks(input_text)
        
        if len(chunks) > 1:
            print(f"Text split into {len(chunks)} chunks for processing...")
        
        # If no seed provided or seed is 0, generate one to use for all chunks
        if seed is None or seed == 0:
            actual_seed = np.random.randint(1, 2**31 - 1)
            print(f"Generated seed for all chunks: {actual_seed}")
        else:
            actual_seed = int(seed)
        
        # Generate speech for each chunk (all chunks use the same seed)
        all_wavs = []
        for i, chunk in enumerate(chunks):
            print(f"Generating speech for chunk {i+1}/{len(chunks)} using seed {actual_seed}: {chunk[:50]}...")
            wav = tts.infer(chunk, ref_codes, ref_text_content, seed=actual_seed)
            all_wavs.append(wav)
        
        # Concatenate all chunks with crossfading
        if len(all_wavs) == 1:
            final_wav = all_wavs[0]
        else:
            final_wav = all_wavs[0]
            for wav in all_wavs[1:]:
                final_wav = crossfade_audio(final_wav, wav, fade_samples=480)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, final_wav, 24000)
            output_path = tmp_file.name
        
        status_msg = f"✅ Speech generated successfully! (Seed: {actual_seed})"
        if len(chunks) > 1:
            status_msg += f"\n📝 Processed in {len(chunks)} chunks using the same seed for consistency"
        
        return output_path, ref_text_content, status_msg
        
    except Exception as e:
        return None, "", f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(
    title="NeuTTS Air - Voice Cloning TTS",
    theme=gr.themes.Ocean()
) as demo:
    gr.Markdown("# 🎙️ NeuTTS Air - Voice Cloning")
    gr.Markdown("Generate natural-sounding speech with instant voice cloning.")
    
    # Model Setup Section
    with gr.Accordion("⚙️ Model Setup", open=False):
        model_selector = gr.Radio(
            choices=list(MODELS.keys()),
            value="Q4 (Faster, Lower Quality)",
            label="Select Model"
        )
        
        with gr.Row():
            download_btn = gr.Button("📥 Download", variant="secondary")
            init_btn = gr.Button("🚀 Initialize", variant="primary")
            unload_btn = gr.Button("🗑️ Unload", variant="secondary")
        
        model_status = gr.Textbox(label="Status", interactive=False)
        
        download_btn.click(
            fn=download_model,
            inputs=[model_selector],
            outputs=[model_status]
        )
        
        init_btn.click(
            fn=initialize_models,
            inputs=[model_selector],
            outputs=[model_status]
        )
        
        unload_btn.click(
            fn=unload_models,
            inputs=[],
            outputs=[model_status]
        )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter the text you want to convert to speech...",
                value="Welcome to Voice Assistant 2.0! I'm faster, smarter, and a little more alive. If you hear breathing in your speaker, don't worry, it's just me practicing.",
                lines=6
            )
            
            ref_audio = gr.Audio(
                label="Reference Audio (⚠️ Use 3-7 seconds to avoid OOM errors)",
                type="filepath",
                sources=["upload"]
            )
            
            ref_text = gr.Textbox(
                label="Reference Text (auto-transcribes if empty)",
                placeholder="Leave empty to auto-transcribe...",
                lines=3
            )
            
            seed_input = gr.Number(
                label="Seed (optional)",
                value=None,
                precision=0
            )
            
            generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
        
        with gr.Column():
            output_audio = gr.Audio(
                label="Generated Speech",
                type="filepath"
            )
            
            status_text = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3
            )
            
            transcription_text = gr.Textbox(
                label="Reference Transcription",
                interactive=False,
                lines=3
            )
    
    # Auto-transcribe when audio is uploaded
    ref_audio.change(
        fn=auto_transcribe_on_upload,
        inputs=[ref_audio],
        outputs=[ref_text]
    )
    
    # Generate speech on button click
    generate_btn.click(
        fn=generate_speech,
        inputs=[input_text, ref_audio, ref_text, seed_input],
        outputs=[output_audio, transcription_text, status_text]
    )
    
    with gr.Accordion("💡 Tips & Examples", open=False):
        gr.Markdown("""
        **Tips for best results:**
        - Reference audio: 3-7 seconds, clean audio, minimal background noise
        - Reference text should match the audio exactly
        
        **Example texts to try:**
        - "Hello, my name is Alex and I'm excited to demonstrate this voice cloning technology."
        - "The quick brown fox jumps over the lazy dog."
        - "Welcome to the future of text-to-speech synthesis!"
        """)


if __name__ == "__main__":
    demo.launch(share=False)
