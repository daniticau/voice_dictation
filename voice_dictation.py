"""
High-Performance Voice Dictation Script for Windows
====================================================
Uses faster-whisper (CTranslate2) for fast, local speech-to-text transcription.
Push-to-talk with F4 hotkey - press and hold to record, release to transcribe and type.

Threading Architecture:
-----------------------
- Main Thread: Runs the pynput keyboard listener (must stay responsive)
- Recording Thread: Handles audio capture in a separate thread to avoid blocking
- Transcription happens on key release in a thread pool to keep UI responsive

The model is loaded once at startup and kept in VRAM for instant inference.
"""

import threading
import queue
import time
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import pyautogui
import winsound
from pynput import keyboard
from faster_whisper import WhisperModel

# =============================================================================
# Configuration
# =============================================================================

HOTKEY = keyboard.Key.f4  # Push-to-talk key
SAMPLE_RATE = 16000       # Whisper requires 16kHz audio
CHANNELS = 1              # Mono audio
MODEL_SIZE = "base.en"    # English-only base model for speed

# Audio feedback frequencies (Hz)
BEEP_START_FREQ = 800     # Higher pitch for "start listening"
BEEP_STOP_FREQ = 400      # Lower pitch for "stop listening"
BEEP_DURATION_MS = 100    # Short beep duration

# History file path (same directory as script)
HISTORY_FILE = Path(__file__).parent / "dictation_history.jsonl"

# =============================================================================
# Global State
# =============================================================================

class DictationState:
    """
    Thread-safe state management for the dictation system.
    
    Using a class with locks ensures that the recording state
    and audio buffer are safely accessed across threads.
    """
    def __init__(self):
        self.is_recording = False
        self.audio_buffer: list[np.ndarray] = []
        self.lock = threading.Lock()
        self.stream: Optional[sd.InputStream] = None
        
    def start_recording(self):
        with self.lock:
            self.is_recording = True
            self.audio_buffer = []
    
    def stop_recording(self) -> np.ndarray:
        with self.lock:
            self.is_recording = False
            if self.audio_buffer:
                # Concatenate all audio chunks into a single array
                audio = np.concatenate(self.audio_buffer, axis=0)
                self.audio_buffer = []
                return audio.flatten().astype(np.float32)
            return np.array([], dtype=np.float32)
    
    def add_audio_chunk(self, chunk: np.ndarray):
        with self.lock:
            if self.is_recording:
                self.audio_buffer.append(chunk.copy())

# =============================================================================
# History Logger
# =============================================================================

class HistoryLogger:
    """
    Simple JSONL-based history logger for dictation sessions.
    
    Each entry is a JSON object on its own line, making it easy to:
    - Append new entries without reading the whole file
    - Process entries line-by-line for analysis
    - Convert to other formats (CSV, DataFrame, etc.)
    """
    
    def __init__(self, filepath: Path = HISTORY_FILE):
        self.filepath = filepath
    
    def log(self, text: str, duration_seconds: float):
        """
        Log a dictation entry to the history file.
        
        Args:
            text: The transcribed text
            duration_seconds: How long the recording was
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "text": text,
            "char_count": len(text),
            "word_count": len(text.split()) if text else 0
        }
        
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Warning: Could not save to history: {e}", file=sys.stderr)
    
    def get_history(self, limit: int = None) -> list[dict]:
        """
        Read history entries from the file.
        
        Args:
            limit: Maximum number of entries to return (newest first)
        
        Returns:
            List of history entry dictionaries
        """
        if not self.filepath.exists():
            return []
        
        entries = []
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception as e:
            print(f"Warning: Could not read history: {e}", file=sys.stderr)
            return []
        
        if limit:
            return entries[-limit:]
        return entries

# =============================================================================
# Audio Feedback
# =============================================================================

def play_start_beep():
    """Play a high-frequency beep to indicate recording has started."""
    # Run in thread to avoid blocking
    threading.Thread(
        target=lambda: winsound.Beep(BEEP_START_FREQ, BEEP_DURATION_MS),
        daemon=True
    ).start()

def play_stop_beep():
    """Play a lower-frequency beep to indicate recording has stopped."""
    threading.Thread(
        target=lambda: winsound.Beep(BEEP_STOP_FREQ, BEEP_DURATION_MS),
        daemon=True
    ).start()

# =============================================================================
# Whisper Model Management
# =============================================================================

def load_whisper_model() -> WhisperModel:
    """
    Load the Whisper model once at startup.
    
    By default, uses CPU for maximum compatibility.
    Set FORCE_CUDA=1 environment variable to try GPU acceleration.
    """
    import os
    
    print(f"Loading Whisper model '{MODEL_SIZE}'...")
    
    # Check if user wants to force CUDA mode
    force_cuda = os.environ.get("FORCE_CUDA", "").lower() in ("1", "true", "yes")
    
    if force_cuda:
        try:
            print("  FORCE_CUDA is set, trying GPU...")
            model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
            # Quick test to verify it actually works
            test_audio = np.zeros(1600, dtype=np.float32)
            list(model.transcribe(test_audio)[0])
            print("[OK] Model loaded on CUDA (float16)")
            return model
        except Exception as e:
            print(f"  CUDA failed: {str(e)[:60]}...")
            print("  Falling back to CPU...")
    
    # Default: use CPU for reliability
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("[OK] Model loaded on CPU (int8)")
    return model

# =============================================================================
# Transcription
# =============================================================================

def transcribe_audio(model: WhisperModel, audio: np.ndarray) -> str:
    """
    Transcribe audio using faster-whisper.
    
    Args:
        model: The loaded WhisperModel instance
        audio: Float32 numpy array of audio samples at 16kHz
    
    Returns:
        Transcribed text with whitespace stripped
    """
    if len(audio) == 0:
        return ""
    
    # faster-whisper expects float32 audio normalized to [-1, 1]
    # sounddevice already provides this format
    segments, info = model.transcribe(
        audio,
        beam_size=5,
        language="en",
        vad_filter=True,  # Filter out non-speech segments
    )
    
    # Collect all segment texts
    text_parts = [segment.text for segment in segments]
    full_text = " ".join(text_parts)
    
    # Strip leading/trailing whitespace as required
    return full_text.strip()

# =============================================================================
# Keyboard Listener
# =============================================================================

class DictationController:
    """
    Main controller for the push-to-talk dictation system.
    
    Threading Model:
    ---------------
    1. The pynput listener runs in the main thread (required by pynput on Windows)
    2. Audio recording uses sounddevice's callback mechanism (runs in audio thread)
    3. Transcription runs in a separate thread to avoid blocking the listener
    
    This architecture ensures:
    - Hotkey detection is always responsive
    - Audio recording doesn't miss samples
    - Transcription doesn't freeze the UI
    """
    
    def __init__(self, model: WhisperModel):
        self.model = model
        self.state = DictationState()
        self.history = HistoryLogger()
        self.transcription_thread: Optional[threading.Thread] = None
        self.recording_start_time: Optional[float] = None
        
    def audio_callback(self, indata: np.ndarray, frames: int, 
                       time_info: dict, status: sd.CallbackFlags):
        """
        Callback for sounddevice audio stream.
        
        This runs in the audio thread managed by sounddevice.
        We just copy the audio data to our buffer - minimal processing
        to avoid audio glitches.
        """
        if status:
            print(f"Audio status: {status}", file=sys.stderr)
        self.state.add_audio_chunk(indata)
    
    def start_recording(self):
        """Begin audio capture when hotkey is pressed."""
        if self.state.is_recording:
            return  # Already recording
        
        play_start_beep()
        self.state.start_recording()
        self.recording_start_time = time.time()
        
        # Start the audio input stream
        # The stream runs in its own thread managed by sounddevice
        self.state.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            callback=self.audio_callback,
        )
        self.state.stream.start()
        print("[REC] Recording...")
    
    def stop_recording_and_transcribe(self):
        """Stop recording, transcribe, and type the result."""
        if not self.state.is_recording:
            return  # Not recording
        
        # Stop the audio stream
        if self.state.stream:
            self.state.stream.stop()
            self.state.stream.close()
            self.state.stream = None
        
        # Get the recorded audio and calculate duration
        audio = self.state.stop_recording()
        duration = time.time() - self.recording_start_time if self.recording_start_time else 0
        self.recording_start_time = None
        play_stop_beep()
        print("[...] Processing...")
        
        if len(audio) == 0:
            print("No audio recorded.")
            return
        
        # Run transcription in a separate thread to avoid blocking the listener
        # This keeps the hotkey responsive even during long transcriptions
        def transcribe_and_type():
            try:
                text = transcribe_audio(self.model, audio)
                if text:
                    print(f"[TXT] Transcribed: {text}")
                    # Log to history
                    self.history.log(text, duration)
                    # Type the text into the active window
                    # Small delay to ensure the window is ready
                    time.sleep(0.1)
                    # Use clipboard for reliable Unicode support
                    import pyperclip
                    pyperclip.copy(text)
                    pyautogui.hotkey('ctrl', 'v')
                else:
                    print("No speech detected.")
            except Exception as e:
                print(f"Transcription error: {e}", file=sys.stderr)
        
        self.transcription_thread = threading.Thread(target=transcribe_and_type, daemon=True)
        self.transcription_thread.start()
    
    def on_press(self, key):
        """Handle key press events."""
        if key == HOTKEY:
            self.start_recording()
    
    def on_release(self, key):
        """Handle key release events."""
        if key == HOTKEY:
            self.stop_recording_and_transcribe()
        elif key == keyboard.Key.esc:
            # ESC to quit
            print("\nExiting...")
            return False  # Stop the listener
    
    def cleanup(self):
        """Clean up resources before exiting."""
        # Stop any active recording
        if self.state.stream:
            try:
                self.state.stream.stop()
                self.state.stream.close()
            except:
                pass
            self.state.stream = None
        self.state.is_recording = False
    
    def run(self):
        """Start the dictation system."""
        import signal
        
        print("\n" + "=" * 50)
        print("Voice Dictation Active")
        print("=" * 50)
        print(f"Hold {HOTKEY} to record, release to transcribe")
        print("Press Ctrl+C or ESC to quit")
        print("=" * 50 + "\n")
        
        # Create the listener
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\nCtrl+C detected. Exiting...")
            self.cleanup()
            self.listener.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Start the listener (non-blocking when using start() instead of join())
        self.listener.start()
        
        # Use a loop with timeout to allow signal handling on Windows
        try:
            while self.listener.is_alive():
                self.listener.join(timeout=0.5)
        except KeyboardInterrupt:
            print("\nCtrl+C detected. Exiting...")
            self.cleanup()
            self.listener.stop()

# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Initialize and run the voice dictation system."""
    print("=" * 50)
    print("Voice Dictation - Powered by faster-whisper")
    print("=" * 50 + "\n")
    
    # Load the model once at startup - stays in memory
    model = load_whisper_model()
    
    # Warm up the model with a short silence to initialize CUDA kernels
    print("Warming up model...")
    dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1 second of silence
    _ = transcribe_audio(model, dummy_audio)
    print("[OK] Model ready!\n")
    
    # Create and run the controller
    controller = DictationController(model)
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")

if __name__ == "__main__":
    main()
