"""
Surgical Agent Orchestration Platform (SAOP)

USAGE:
    python real_time_speech.py --date 20250422 --time 093300 --remote_user "user" --remote_host "host" --remote_port "port" --remote_path "/path/to/your/remote/server" --save_path "/path/for/the/audio/file"

PACKAGES:
    - NVIDIA GPU: pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
    - Intel Arc GPU: pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 torch-directml==0.2.5.dev240914
    - others: pip install PyAudio sounddevice scipy pydub edge-tts

NOTES:
    - Adjust time.sleep values based on microsoft_edge_tts audio duration!
"""

import sounddevice as sd
import numpy as np
import torch
from scipy.io import wavfile
import subprocess
import argparse
import os
import sys
import io
import edge_tts
import torchaudio
import time
from faster_whisper import WhisperModel
from collections import deque

# Explicitly set output encoding to UTF-8 (for remote server decoding with 'utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def microsoft_edge_tts(args, text):
    # Save TTS file
    tts = edge_tts.Communicate(text, args.tts_model)
    sound_file = './tmp.mp3'
    tts.save_sync(sound_file)
    # Play TTS file (torchaudio is faster than pydub's AudioSegment)
    waveform, sr = torchaudio.load("tmp.mp3")
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    sd.play(waveform.T.numpy(), samplerate=16000)

def listen_for_keyword(args, whisper_model, vad_model, utils):
    print("🎤 Ready. Your microphone should be available. Call DaVinci!", flush=True)
    microsoft_edge_tts(args, "Ready.")
    time.sleep(0.8) # Adjusted for "Ready." duration (0.7 causes audio clipping)
    get_speech_timestamps, _, _, _, _ = utils

    sample_rate = args.sample_rate
    block_size = 512
    buffer = deque(maxlen=sample_rate * 8) # 8, 10 
    audio_window = []
    start_time = time.time()
    timeout = 20 # 60, 3600

    # Cross-segment keyword check
    def cross_contains(text, keywords=("davinci", "davin", "vinci")):
        """Check if keyword is present in current STT text"""
        text = text.replace(" ", "").lower()  # Remove spaces (e.g., handle "da vinci")
        return any(kw in text for kw in keywords)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=block_size) as stream:
        speech_start_time = None  # Store speech detection timestamp
        while True:
            if time.time() - start_time > timeout:
                print("Timeout: Keyword recognition time exceeded", flush=True)
                return False

            chunk, _ = stream.read(block_size)
            chunk = chunk.flatten()
            buffer.extend(chunk)
            audio_window.append(torch.from_numpy(chunk).float())

            # Run VAD when 1 second of audio is accumulated
            if len(audio_window) * block_size >= sample_rate:
                audio_tensor = torch.cat(audio_window)

                timestamps = get_speech_timestamps(
                    audio_tensor, vad_model, sampling_rate=sample_rate,
                    min_silence_duration_ms=200, # 200
                    min_speech_duration_ms=200, # 150
                    speech_pad_ms=150 # 150
                )

                if timestamps:
                    # Record speech detection time (first segment start time)
                    if speech_start_time is None:
                        speech_start_time = time.time()

                    float_audio = np.array(buffer, dtype=np.float32)

                    try:
                        segments, _ = whisper_model.transcribe(
                            float_audio,
                            beam_size=5,
                            #vad_filter=True,
                            language='en',
                            temperature=[0.0],
                            condition_on_previous_text=False,
                            hotwords="davinci",
                            #word_timestamps=True,
                            compression_ratio_threshold=1.2,
                            log_prob_threshold=-1.0, # 0.5, Ignore segments with low decoding probability
                            no_speech_threshold=0.25, # 0.35, Suppress noise/silence
                            #hallucination_silence_threshold=0.3, 
                        )

                        for seg in segments:
                            text = seg.text.strip()
                            #print(f"🗣 Whisper result: {text}", flush=True)

                            if cross_contains(text):
                                detect_time = time.time()  # ✅ 키워드 인식 시점
                                if speech_start_time:
                                    latency = detect_time - speech_start_time
                                    #print(f"⚡ Latency: {latency:.2f} seconds", flush=True)

                                print("✅ 'davinci' recognized! Please tell me your command.", flush=True)
                                microsoft_edge_tts(args, "Please continue.")
                                return True

                    except Exception as e:
                        print(f"Whisper recognition error: {e}", flush=True)

                    # Keep only 1.0 second tail instead of clearing everything
                    tail_sec = 1.0
                    tail = int(tail_sec * sample_rate)
                    if len(buffer) > tail:
                        tail_buf = list(buffer)[-tail:]
                        buffer.clear()
                        buffer.extend(tail_buf)

                    audio_window.clear()

                # Reset window periodically even when VAD doesn't detect speech
                elif len(audio_window) > int(3 * sample_rate / block_size):
                    # Keep overlap tail
                    tail_sec = 1.0
                    tail = int(tail_sec * sample_rate)
                    if len(buffer) > tail:
                        tail_buf = list(buffer)[-tail:]
                        buffer.clear()
                        buffer.extend(tail_buf)
                    audio_window.clear()

def record_audio(args, model):
    """Record audio from microphone"""
    sd.default.samplerate = args.sample_rate
    sd.default.channels = args.channel_count
    audio_buffer = []
    silence_duration = 0.0

    with sd.InputStream(samplerate=args.sample_rate, channels=args.channel_count, dtype='float32', blocksize=args.chunk_size) as stream:
        time.sleep(1.2)  # Adjusted for "Please continue." duration
        while True:
            audio_chunk, _ = stream.read(args.chunk_size)
            if audio_chunk.size == 0:
                break
            audio_buffer.append(audio_chunk.copy())

            chunk_tensor = torch.from_numpy(audio_chunk.squeeze()).float()
            speech_prob = model(chunk_tensor, args.sample_rate).item()

            if speech_prob < args.voice_probability_threshold:
                silence_duration += (args.chunk_size / args.sample_rate)
            else:
                silence_duration = 0.0

            if silence_duration >= args.silence_threshold:
                print("Silence detected, end recording.", flush=True)
                break
    return audio_buffer

def save_audio(args, audio_buffer):
    """Save recorded audio buffer to WAV file"""
    if len(audio_buffer) == 0:
        print("No recorded audio data. Skipping save.", flush=True)
        return None

    audio_data = np.concatenate(audio_buffer, axis=0).squeeze()
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    save_path = args.save_path
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    # Convert to Linux path format (for SCP)
    remote_path_for_scp = os.path.normpath(os.path.join(args.remote_path, save_path)).replace('\\', '/')
    args.remote_path = remote_path_for_scp
    filename = os.path.join(save_path, "record.wav")

    wavfile.write(filename, args.sample_rate, audio_int16)
    return filename

def send_file_to_server(args, filename):
    """Send WAV file to server"""
    if not filename:
        return
    
    try:
        scp_target = f"{args.remote_user}@{args.remote_host}:{args.remote_path}"
        subprocess.run(["scp", "-P", args.remote_port, filename, scp_target], check=True, stdin=subprocess.DEVNULL)
        print("Completed to transfer recording file to server!", flush=True)
    except subprocess.CalledProcessError as e:
        print("Failed to transfer recording file to server.", e, flush=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Voice recording and server transfer")
    parser.add_argument("--date", default=None, help="Current date")
    parser.add_argument("--time", default=None, help="Current time")
    parser.add_argument("--remote_user", default=None, help="Remote Linux server user")
    parser.add_argument("--remote_host", default=None, help="Remote Linux server host")
    parser.add_argument("--remote_port", default=None, help="Remote Linux server port")
    parser.add_argument("--remote_path", default="./",
                        help="Linux remote server base path")
    parser.add_argument("--save_path", default="./",
                        help="Linux remote server data path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Microphone input and recording settings
    args.sample_rate = 16000
    args.channel_count = 1
    args.chunk_size = 512                        # Number of samples to read at once (512=32ms @16kHz)
    args.silence_threshold = 4                   # Silence duration threshold (seconds)
    args.voice_probability_threshold = 0.5       # Voice/silence judgment threshold probability
    args.tts_model = "en-US-AriaNeural"          # https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts
    
    # For edge-devices
    device = "cpu"

    # Load silero-vad model
    torch.set_num_threads(1)

    # Ensure cache directory exists
    cache_dir = os.path.expanduser('~/.cache/torch/hub/')
    os.makedirs(cache_dir, exist_ok=True)
    torch.hub.set_dir(cache_dir)

    try:
        # Try loading from cache first
        vad_model, utils = torch.hub.load(
            'snakers4/silero-vad', 
            'silero_vad', 
            trust_repo=True, 
            force_reload=False
        )
        print("Silero VAD model loaded from cache.", flush=True)
    except Exception as e:
        print("Downloading Silero VAD model (first time only).", flush=True)
        # Force download if cache doesn't exist or is corrupted
        vad_model, utils = torch.hub.load(
            'snakers4/silero-vad', 
            'silero_vad', 
            trust_repo=True, 
            force_reload=True
        )
        print("Silero VAD model downloaded and loaded.", flush=True)

    # Load whisper model for keyword detection
    whisper_model = WhisperModel("small", device="cpu", compute_type="float32")
    print("Whisper model loaded.", flush=True)

    # Wait for keyword using VAD + Whisper
    listen_for_keyword(args, whisper_model, vad_model, utils)

    # Record audio after 'davinci' is recognized
    audio_buffer = record_audio(args, vad_model)
    
    # Transfer recorded audio to server
    output_filename = save_audio(args, audio_buffer)
    send_file_to_server(args, output_filename)

    # Delete file after transfer
    if output_filename and os.path.exists(output_filename):
        os.remove(output_filename)
        dir_path = os.path.dirname(output_filename)
        
        # Delete parent folders sequentially
        while dir_path and os.path.isdir(dir_path):
            if not os.listdir(dir_path):  # If folder is empty
                os.rmdir(dir_path)
                dir_path = os.path.dirname(dir_path)
            else:
                break  # Stop if folder is not empty
            