"""
OpenAI-compatible TTS Streaming Server using NeuTTS-Air
Supports streaming audio generation with RTF < 1 and latency < 1 second
Handles long-form text by chunking into sentences
"""

import os
import sys
import time
import struct
import re
from pathlib import Path
from typing import Generator, Optional, List
from contextlib import asynccontextmanager

import numpy as np
import torch
import librosa
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path for neuttsair import
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuttsair.neutts import NeuTTSAir
from neucodec import NeuCodec


# Configuration
REFERENCE_AUDIO_PATH = "/home/serj/tmp/example2.wav"
REFERENCE_TEXT = "All right, so have you ever heard of a little thing named text to speech? Well, it allows you to convert text into speech. I know that's super cool, isn't it?"
SAMPLE_RATE = 24000
BACKBONE_MODEL = "neuphonic/neutts-air-q8-gguf"

# Max characters per chunk (~20 seconds of audio at ~150 WPM, ~5 chars/word)
# Context is 2048 tokens, with ~500 ref codes, leaving ~1500 for text+output
# At ~50 tokens/sec output rate and hop_length of 480, each token = ~20ms audio
# ~1000 output tokens = 20 seconds. Allow ~80 words = ~400 chars per chunk
MAX_CHUNK_CHARS = 400


# Global TTS instance
tts_engine: Optional[NeuTTSAir] = None
ref_codes: Optional[torch.Tensor] = None


def encode_reference_voice():
    """Encode the reference voice at startup using full codec (not ONNX)"""
    global ref_codes
    print(f"Encoding reference voice from: {REFERENCE_AUDIO_PATH}")
    
    # Use the full NeuCodec for encoding (ONNX decoder can't encode)
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to("cpu")
    
    # Load and encode reference audio
    wav, _ = librosa.load(REFERENCE_AUDIO_PATH, sr=16000, mono=True)
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    
    with torch.no_grad():
        ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
    
    # Free the encoder to save memory
    del codec
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"Reference encoded: {len(ref_codes)} codes")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize TTS engine on startup"""
    global tts_engine, ref_codes
    
    print("=" * 60)
    print("Initializing NeuTTS-Air TTS Server")
    print("=" * 60)
    
    # Initialize TTS engine with GGUF model on GPU and ONNX decoder
    tts_engine = NeuTTSAir(
        backbone_repo=BACKBONE_MODEL,
        backbone_device="gpu",
        codec_repo="neuphonic/neucodec-onnx-decoder",
        codec_device="cpu"
    )
    
    # Encode reference voice
    encode_reference_voice()
    
    print("=" * 60)
    print("TTS Server Ready!")
    print("=" * 60)
    
    yield
    
    # Cleanup
    print("Shutting down TTS server...")


app = FastAPI(
    title="NeuTTS-Air OpenAI-Compatible TTS API",
    description="Streaming TTS API compatible with OpenAI's /v1/audio/speech endpoint",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request model"""
    model: str = Field(default="neutts-air", description="Model to use for TTS")
    input: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default="default", description="Voice to use")
    response_format: str = Field(default="pcm", description="Audio format (pcm, wav)")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speed multiplier")


def create_wav_header(sample_rate: int = 24000, bits_per_sample: int = 16, num_channels: int = 1) -> bytes:
    """Create a WAV header for streaming (with placeholder size)"""
    # Use max int32 as placeholder for streaming
    data_size = 0x7FFFFFFF
    file_size = data_size + 36
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        file_size,
        b'WAVE',
        b'fmt ',
        16,  # fmt chunk size
        1,   # audio format (PCM)
        num_channels,
        sample_rate,
        sample_rate * num_channels * bits_per_sample // 8,  # byte rate
        num_channels * bits_per_sample // 8,  # block align
        bits_per_sample,
        b'data',
        data_size
    )
    return header


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """
    Split text into chunks at sentence boundaries.
    Each chunk should be under max_chars to fit within context window.
    """
    # Split on sentence boundaries
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text.strip())
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If single sentence is too long, split on commas or other punctuation
        if len(sentence) > max_chars:
            # Split on commas, semicolons, or colons
            sub_parts = re.split(r'(?<=[,;:])\s+', sentence)
            for part in sub_parts:
                part = part.strip()
                if not part:
                    continue
                if len(current_chunk) + len(part) + 1 <= max_chars:
                    current_chunk = f"{current_chunk} {part}".strip() if current_chunk else part
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part
        elif len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [text[:max_chars]]


def audio_chunk_generator(text: str, response_format: str = "pcm") -> Generator[bytes, None, None]:
    """Generate audio chunks from text using streaming TTS with long-form support"""
    global tts_engine, ref_codes
    
    start_time = time.time()
    first_chunk_time = None
    total_samples = 0
    chunk_count = 0
    
    # Send WAV header if requested
    if response_format == "wav":
        yield create_wav_header(SAMPLE_RATE)
    
    # Split text into chunks for long-form content
    text_chunks = chunk_text(text)
    num_text_chunks = len(text_chunks)
    
    if num_text_chunks > 1:
        print(f"Long text detected: splitting into {num_text_chunks} chunks", flush=True)
    
    try:
        for text_idx, text_chunk in enumerate(text_chunks):
            if num_text_chunks > 1:
                print(f"Processing text chunk {text_idx + 1}/{num_text_chunks}: '{text_chunk[:50]}...'", flush=True)
            
            for audio_chunk in tts_engine.infer_stream(text_chunk, ref_codes, REFERENCE_TEXT):
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency = first_chunk_time - start_time
                    print(f"First chunk latency: {latency:.3f}s", flush=True)
                
                # Convert float audio to int16 PCM
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
                total_samples += len(audio_int16)
                chunk_count += 1
                
                yield audio_int16.tobytes()
        
        # Calculate final metrics
        total_time = time.time() - start_time
        audio_duration = total_samples / SAMPLE_RATE
        rtf = total_time / audio_duration if audio_duration > 0 else 0
        
        print(f"Generation complete: {chunk_count} chunks, "
              f"audio={audio_duration:.2f}s, time={total_time:.2f}s, RTF={rtf:.3f}", flush=True)
        
    except Exception as e:
        print(f"Error in audio generation: {e}")
        raise


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    OpenAI-compatible TTS endpoint with streaming support.
    Streams audio chunks as they are generated.
    """
    if not tts_engine or ref_codes is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialized")
    
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    print(f"TTS Request: '{request.input[:100]}{'...' if len(request.input) > 100 else ''}'")
    
    # Determine content type based on format
    content_type = "audio/wav" if request.response_format == "wav" else "audio/pcm"
    
    # Create streaming response
    return StreamingResponse(
        audio_chunk_generator(request.input, request.response_format),
        media_type=content_type,
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Channels": "1",
            "X-Bits-Per-Sample": "16",
            "Transfer-Encoding": "chunked",
        }
    )


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)"""
    return {
        "object": "list",
        "data": [
            {
                "id": "neutts-air",
                "object": "model",
                "created": 1699000000,
                "owned_by": "neuphonic"
            }
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": BACKBONE_MODEL,
        "reference_loaded": ref_codes is not None
    }


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML page"""
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(static_path, media_type="text/html")
    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

