"""
Text-to-Speech Agent for LLMFS

This agent uses Cartesia for real-time text-to-speech and speech-to-text.
It exposes bidirectional audio-text conversion through the 9P filesystem interface.

Filesystem structure:
    agents/ts/
    ├── ctl        # Control: auto on/off, input_device, output_device, voice, etc.
    ├── input      # Write text here → generates AUDIO_OUT
    ├── OUTPUT     # Read transcribed text from audio_in (CAPS = blocking read)
    ├── audio_in   # Write audio data → transcribed to OUTPUT
    ├── AUDIO_OUT  # Read generated audio from input text (CAPS = blocking read)
    ├── config     # Configuration as JSON
    ├── status     # Real-time status (levels, devices, auto mode)
    └── errors     # Error queue
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum

# Core LLMFS imports
from core.files import (
    SyntheticDir, SyntheticFile, QueueFile,
    CtlFile, CtlHandler
)
from core.types import FidState

# Cartesia imports
try:
    from cartesia import AsyncCartesia
    CARTESIA_AVAILABLE = True
except ImportError:
    CARTESIA_AVAILABLE = False

# Audio imports
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# Speech recognition
try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False


# Audio configuration
if AUDIO_AVAILABLE:
    FORMAT = pyaudio.paInt16
else:
    FORMAT = 8  # paInt16 value as fallback
CHANNELS = 1
SAMPLE_RATE = 44100  # Cartesia standard sample rate
CHUNK_SIZE = 1024

# Available Cartesia voices
CARTESIA_VOICES = [
    "6ccbfb76-1fc6-48f7-b71d-91ac6298247b",  # Example voice from working code
    "79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
    "a0e99841-438c-4a64-b679-ae501e7d6091",  # Helpful Woman
    "694f9389-aac1-45b6-b726-9d9369183238",  # Gentle Man
    "bf991597-6c13-47e4-8411-91ec2de5c466",  # Australian Woman
]


class TSState(Enum):
    """TS Agent state"""
    IDLE = "idle"           # Not active
    READY = "ready"         # Ready but not auto
    AUTO = "auto"           # Auto mode - actively listening/speaking
    PROCESSING = "processing"  # Processing audio/text
    ERROR = "error"         # Error state


@dataclass
class TSConfig:
    """Configuration for TS agent"""
    voice: str = CARTESIA_VOICES[0]  # Default to British Lady
    model: str = "sonic-english"
    output_device: Optional[int] = None
    input_device: Optional[int] = None
    auto_mode: bool = False
    language: str = "en"  # For STT
    
    def to_dict(self) -> dict:
        return {
            "voice": self.voice,
            "model": self.model,
            "output_device": self.output_device,
            "input_device": self.input_device,
            "auto_mode": self.auto_mode,
            "language": self.language,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'TSConfig':
        return cls(
            voice=d.get("voice", CARTESIA_VOICES[0]),
            model=d.get("model", "sonic-english"),
            output_device=d.get("output_device"),
            input_device=d.get("input_device"),
            auto_mode=d.get("auto_mode", False),
            language=d.get("language", "en"),
        )


class TSCtlHandler(CtlHandler):
    """Control handler for TS agent"""
    
    def __init__(self, agent: 'TSAgent'):
        self.agent = agent
    
    async def execute(self, command: str) -> Optional[str]:
        parts = command.split(' ', 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd == "auto":
            if arg:
                if arg.lower() in ("on", "true", "1"):
                    await self.agent.set_auto_mode(True)
                    return "Auto mode enabled"
                elif arg.lower() in ("off", "false", "0"):
                    await self.agent.set_auto_mode(False)
                    return "Auto mode disabled"
                else:
                    raise ValueError("Usage: auto on|off")
            return "on" if self.agent.config.auto_mode else "off"
        
        elif cmd == "voice":
            if arg:
                self.agent.config.voice = arg
                return f"Voice set to {arg}"
            return self.agent.config.voice
        
        elif cmd == "model":
            if arg:
                self.agent.config.model = arg
                return f"Model set to {arg}"
            return self.agent.config.model
        
        elif cmd == "devices":
            # List audio devices
            if not AUDIO_AVAILABLE:
                return "PyAudio not available"
            pya = pyaudio.PyAudio()
            lines = []
            for i in range(pya.get_device_count()):
                info = pya.get_device_info_by_index(i)
                direction = []
                if info['maxInputChannels'] > 0:
                    direction.append("in")
                if info['maxOutputChannels'] > 0:
                    direction.append("out")
                lines.append(f"{i}: {info['name']} ({','.join(direction)})")
            pya.terminate()
            return "\n".join(lines)
        
        elif cmd == "output":
            # Set output device index
            if arg:
                try:
                    idx = int(arg)
                    self.agent.config.output_device = idx
                    return f"Output device set to {idx}"
                except ValueError:
                    raise ValueError("Usage: output <device_index>")
            return str(self.agent.config.output_device) if self.agent.config.output_device is not None else "default"
        
        elif cmd == "input":
            # Set input device index
            if arg:
                try:
                    idx = int(arg)
                    self.agent.config.input_device = idx
                    return f"Input device set to {idx}"
                except ValueError:
                    raise ValueError("Usage: input <device_index>")
            return str(self.agent.config.input_device) if self.agent.config.input_device is not None else "default"
        
        elif cmd == "language":
            if arg:
                self.agent.config.language = arg
                return f"Language set to {arg}"
            return self.agent.config.language
        
        elif cmd == "start":
            await self.agent.start()
            return "Started"
        
        elif cmd == "stop":
            await self.agent.stop()
            return "Stopped"
        
        elif cmd == "restart":
            await self.agent.stop()
            await asyncio.sleep(0.5)
            await self.agent.start()
            return "Restarted"
        
        elif cmd == "status":
            lines = [
                f"state: {self.agent.state.value}",
                f"auto_mode: {self.agent.config.auto_mode}",
                f"voice: {self.agent.config.voice}",
                f"model: {self.agent.config.model}",
                f"input_level: {self.agent._input_level:.2f}",
                f"output_level: {self.agent._output_level:.2f}",
                f"input_device: {self.agent.config.input_device or 'default'}",
                f"output_device: {self.agent.config.output_device or 'default'}",
            ]
            return "\n".join(lines)
        
        elif cmd == "voices":
            return "\n".join([f"{v}" for v in CARTESIA_VOICES])
        
        else:
            raise ValueError(
                f"Unknown command: {cmd}. "
                "Available: auto, voice, model, devices, output, input, language, start, stop, restart, status, voices"
            )
    
    async def get_status(self) -> bytes:
        lines = [
            f"state {self.agent.state.value}",
            f"auto {self.agent.config.auto_mode}",
            f"voice {self.agent.config.voice}",
            f"model {self.agent.config.model}",
            f"input_device {self.agent.config.input_device or 'default'}",
            f"output_device {self.agent.config.output_device or 'default'}",
            f"input_level {self.agent._input_level:.2f}",
            f"output_level {self.agent._output_level:.2f}",
            "",
            "Available devices:",
        ]
        
        # Add device list
        if AUDIO_AVAILABLE:
            try:
                pya = pyaudio.PyAudio()
                for i in range(pya.get_device_count()):
                    info = pya.get_device_info_by_index(i)
                    direction = []
                    if info['maxInputChannels'] > 0:
                        direction.append("in")
                    if info['maxOutputChannels'] > 0:
                        direction.append("out")
                    lines.append(f"  {i}: {info['name']} ({','.join(direction)})")
                pya.terminate()
            except Exception as e:
                lines.append(f"  Error listing devices: {e}")
        else:
            lines.append("  PyAudio not available")
        
        return ("\n".join(lines) + "\n").encode()


class TSInputFile(SyntheticFile):
    """Text input file - write text to generate speech"""
    
    def __init__(self, agent: 'TSAgent'):
        super().__init__("input")
        self.agent = agent
        self._last_input = ""
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        return self._last_input.encode()[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        text = data.decode('utf-8').strip()
        if not text:
            return len(data)
        
        self._last_input = text
        await self.agent._text_input_queue.put(text)
        return len(data)


class TSOutputFile(SyntheticFile):
    """Text output file - read transcribed text"""
    
    def __init__(self, agent: 'TSAgent'):
        super().__init__("OUTPUT")
        self.agent = agent
        self._reader_queues: Dict[int, asyncio.Queue] = {}
    
    async def open(self, fid: FidState, mode: int):
        """Create a queue for this reader"""
        self._reader_queues[fid.fid] = asyncio.Queue(maxsize=100)
    
    async def close(self, fid: FidState):
        """Remove queue when reader closes"""
        self._reader_queues.pop(fid.fid, None)
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """Read transcribed text - blocks until data available"""
        queue = self._reader_queues.get(fid.fid)
        if queue is None:
            queue = asyncio.Queue(maxsize=100)
            self._reader_queues[fid.fid] = queue
        
        try:
            data = await asyncio.wait_for(queue.get(), timeout=30.0)
            return data[:count]
        except asyncio.TimeoutError:
            return b""
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Output is read-only")
    
    async def broadcast(self, data: bytes):
        """Send text data to all readers"""
        for queue in self._reader_queues.values():
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(data)
                except:
                    pass


class TSAudioInFile(SyntheticFile):
    """Audio input file - write audio to transcribe to text"""
    
    def __init__(self, agent: 'TSAgent'):
        super().__init__("audio_in")
        self.agent = agent
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        # Audio input is write-only
        return b""
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        if data:
            await self.agent._audio_input_queue.put(data)
        return len(data)


class TSAudioOutFile(SyntheticFile):
    """Audio output file - read generated audio"""
    
    def __init__(self, agent: 'TSAgent'):
        super().__init__("AUDIO_OUT")
        self.agent = agent
        self._reader_queues: Dict[int, asyncio.Queue] = {}
    
    async def open(self, fid: FidState, mode: int):
        """Create a queue for this reader"""
        self._reader_queues[fid.fid] = asyncio.Queue(maxsize=100)
    
    async def close(self, fid: FidState):
        """Remove queue when reader closes"""
        self._reader_queues.pop(fid.fid, None)
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """Read audio data - blocks until data available"""
        queue = self._reader_queues.get(fid.fid)
        if queue is None:
            queue = asyncio.Queue(maxsize=100)
            self._reader_queues[fid.fid] = queue
        
        try:
            data = await asyncio.wait_for(queue.get(), timeout=30.0)
            return data[:count]
        except asyncio.TimeoutError:
            return b""
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Audio output is read-only")
    
    async def broadcast(self, data: bytes):
        """Send audio data to all readers"""
        for queue in self._reader_queues.values():
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(data)
                except:
                    pass


class TSAgent(SyntheticDir):
    """
    Text-to-Speech Agent
    
    Provides bidirectional audio-text conversion:
    - Text → Audio (TTS via Cartesia)
    - Audio → Text (STT via speech_recognition)
    """
    
    def __init__(
        self,
        name: str,
        route_manager=None
    ):
        super().__init__(name)
        
        if not CARTESIA_AVAILABLE:
            raise ImportError("Cartesia package not available. Install with: pip install cartesia")
        
        self.route_manager = route_manager
        
        # Configuration
        self.config = TSConfig()
        
        # State
        self.state = TSState.IDLE
        self._running = False
        self._auto_task = None
        
        # Cartesia client
        api_key = os.getenv("CARTESIA_API_KEY")
        if not api_key:
            raise ValueError("CARTESIA_API_KEY not found in environment")
        self.cartesia = AsyncCartesia(api_key=api_key)
        
        # Audio
        self._pya = None
        self._input_stream = None
        self._output_stream = None
        self._input_level = 0.0
        self._output_level = 0.0
        
        # Speech recognizer (for STT)
        if STT_AVAILABLE:
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
        else:
            self._recognizer = None
        
        # Queues
        self._text_input_queue = asyncio.Queue()
        self._audio_input_queue = asyncio.Queue()
        self._audio_output_queue = asyncio.Queue()
        
        # Files
        self.add(CtlFile("ctl", TSCtlHandler(self)))
        
        # Text input (write text → generates audio)
        self.input = TSInputFile(self)
        self.add(self.input)
        
        # Text output (read transcribed text from audio)
        self.output = TSOutputFile(self)
        self.add(self.output)
        
        # Audio input (write audio → transcribe to text)
        self.audio_in = TSAudioInFile(self)
        self.add(self.audio_in)
        
        # Audio output (read generated audio from text)
        self.audio_out = TSAudioOutFile(self)
        self.add(self.audio_out)
        
        # Config file
        self.add(TSConfigFile(self))
        
        # Status file
        self.add(TSStatusFile(self))
        
        # Error queue
        self.errors = QueueFile("errors")
        self.add(self.errors)
    
    async def set_auto_mode(self, enabled: bool):
        """Enable/disable auto mode"""
        self.config.auto_mode = enabled
        
        if enabled:
            if not self._running:
                await self.start()
            self.state = TSState.AUTO
        else:
            self.state = TSState.READY if self._running else TSState.IDLE
    
    async def start(self):
        """Start the agent"""
        if self._running:
            return
        
        self._running = True
        self.state = TSState.READY
        
        # Initialize PyAudio
        if AUDIO_AVAILABLE and self._pya is None:
            self._pya = pyaudio.PyAudio()
        
        # Start processing tasks
        asyncio.create_task(self._process_text_to_speech())
        asyncio.create_task(self._process_speech_to_text())
        asyncio.create_task(self._play_audio_output())
        
        # Start auto mode if enabled
        if self.config.auto_mode:
            self._auto_task = asyncio.create_task(self._auto_mode_loop())
        
        await self.errors.post(b"TS Agent started\n")
    
    async def stop(self):
        """Stop the agent"""
        if not self._running:
            return
        
        self._running = False
        self.state = TSState.IDLE
        
        # Cancel auto mode
        if self._auto_task:
            self._auto_task.cancel()
            try:
                await self._auto_task
            except asyncio.CancelledError:
                pass
            self._auto_task = None
        
        # Close audio streams
        if self._input_stream:
            await asyncio.to_thread(self._input_stream.stop_stream)
            await asyncio.to_thread(self._input_stream.close)
            self._input_stream = None
        
        if self._output_stream:
            await asyncio.to_thread(self._output_stream.stop_stream)
            await asyncio.to_thread(self._output_stream.close)
            self._output_stream = None
        
        # Terminate PyAudio
        if self._pya:
            await asyncio.to_thread(self._pya.terminate)
            self._pya = None
        
        await self.errors.post(b"TS Agent stopped\n")
    
    async def _process_text_to_speech(self):
        """Convert text input to audio output using Cartesia"""
        await self.errors.post(b"TTS processor started\n")
        
        while self._running:
            try:
                # Get text from queue
                text = await asyncio.wait_for(
                    self._text_input_queue.get(),
                    timeout=0.1
                )
                
                await self.errors.post(f"TTS: Processing '{text[:50]}...'\n".encode())
                self.state = TSState.PROCESSING
                
                # Generate audio using Cartesia
                try:
                    chunk_count = 0
                    total_bytes = 0
                    wav_data = b""
                    
                    # Use WAV container like the working example
                    output_format = {
                        "container": "wav",
                        "sample_rate": 44100,
                        "encoding": "pcm_s16le",
                    }
                    
                    await self.errors.post(f"TTS: Requesting audio with format: {output_format}\n".encode())
                    
                    # Use bytes API (not websocket) for more reliable output
                    bytes_iter = await self.cartesia.tts.bytes(
                        model_id=self.config.model,
                        transcript=text,
                        voice={
                            "mode": "id",
                            "id": self.config.voice,
                        },
                        language="en",
                        output_format=output_format,
                    )
                    
                    await self.errors.post(b"TTS: Generating audio...\n")
                    
                    # Collect all WAV data
                    async for chunk in bytes_iter:
                        wav_data += chunk
                        chunk_count += 1
                    
                    total_bytes = len(wav_data)
                    await self.errors.post(f"TTS: Generated {chunk_count} chunks, {total_bytes} bytes WAV total\n".encode())
                    
                    # Extract PCM data from WAV (skip 44-byte header)
                    if len(wav_data) > 44:
                        pcm_data = wav_data[44:]  # Standard WAV header is 44 bytes
                        await self.errors.post(f"TTS: Extracted {len(pcm_data)} bytes of PCM from WAV\n".encode())
                        
                        # Queue PCM audio for output
                        await self._audio_output_queue.put(pcm_data)
                    else:
                        await self.errors.post(f"TTS: WAV data too short ({len(wav_data)} bytes)\n".encode())
                    
                except Exception as e:
                    await self.errors.post(f"TTS error: {e}\n".encode())
                
                self.state = TSState.AUTO if self.config.auto_mode else TSState.READY
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.errors.post(f"Text processing error: {e}\n".encode())
    
    async def _process_speech_to_text(self):
        """Convert audio input to text output using speech_recognition"""
        if not STT_AVAILABLE:
            await self.errors.post(b"Speech recognition not available\n")
            return
        
        audio_buffer = b""
        
        while self._running:
            try:
                # Get audio from queue
                chunk = await asyncio.wait_for(
                    self._audio_input_queue.get(),
                    timeout=0.1
                )
                
                # Accumulate audio
                audio_buffer += chunk
                
                # Process when we have enough audio (e.g., 2 seconds)
                min_length = SAMPLE_RATE * 2 * 2  # 2 seconds of 16-bit audio
                if len(audio_buffer) >= min_length:
                    self.state = TSState.PROCESSING
                    
                    try:
                        # Convert to speech_recognition AudioData
                        audio_data = sr.AudioData(
                            audio_buffer,
                            SAMPLE_RATE,
                            2  # 16-bit = 2 bytes per sample
                        )
                        
                        # Transcribe
                        text = await asyncio.to_thread(
                            self._recognizer.recognize_google,
                            audio_data,
                            language=self.config.language
                        )
                        
                        # Write to output
                        await self.output.broadcast(f"{text}\n".encode())
                        
                    except sr.UnknownValueError:
                        await self.errors.post(b"Could not understand audio\n")
                    except sr.RequestError as e:
                        await self.errors.post(f"STT error: {e}\n".encode())
                    except Exception as e:
                        await self.errors.post(f"Transcription error: {e}\n".encode())
                    
                    # Clear buffer
                    audio_buffer = b""
                    
                    self.state = TSState.AUTO if self.config.auto_mode else TSState.READY
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                await self.errors.post(f"Audio input error: {e}\n".encode())
    
    async def _play_audio_output(self):
        """Play generated audio and broadcast to audio_out file"""
        stream = None
        chunks_played = 0
        chunks_broadcast = 0
        
        try:
            # Open output stream
            if AUDIO_AVAILABLE and self._pya:
                try:
                    await self.errors.post(f"Opening output device {self.config.output_device or 'default'}\n".encode())
                    await self.errors.post(f"Audio params: rate={SAMPLE_RATE}, channels={CHANNELS}, format={FORMAT}\n".encode())
                    
                    stream = await asyncio.to_thread(
                        self._pya.open,
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        output=True,
                        output_device_index=self.config.output_device,
                        frames_per_buffer=CHUNK_SIZE,
                    )
                    self._output_stream = stream
                    
                    # Verify actual parameters
                    actual_rate = stream._rate if hasattr(stream, '_rate') else SAMPLE_RATE
                    await self.errors.post(f"Output stream opened: actual_rate={actual_rate}\n".encode())
                except Exception as e:
                    await self.errors.post(f"Failed to open output stream: {e}\n".encode())
            else:
                if not AUDIO_AVAILABLE:
                    await self.errors.post(b"PyAudio not available - no local playback\n")
                elif not self._pya:
                    await self.errors.post(b"PyAudio not initialized - no local playback\n")
            
            while self._running:
                try:
                    # Get audio from queue
                    audio_chunk = await asyncio.wait_for(
                        self._audio_output_queue.get(),
                        timeout=0.1
                    )
                    
                    # Audio is already raw bytes from Cartesia
                    audio_bytes = audio_chunk
                    
                    # Log first chunk details
                    if chunks_broadcast == 0:
                        await self.errors.post(f"First audio chunk: {len(audio_bytes)} bytes\n".encode())
                    
                    # Calculate output level
                    if len(audio_bytes) > 0:
                        try:
                            import audioop
                            rms = audioop.rms(audio_bytes, 2)
                            self._output_level = min(1.0, (rms / 32768.0) * 5.0)
                        except Exception:
                            pass
                    
                    # Broadcast to filesystem
                    await self.audio_out.broadcast(audio_bytes)
                    chunks_broadcast += 1
                    
                    # Play through speakers
                    if stream:
                        await asyncio.to_thread(stream.write, audio_bytes)
                        chunks_played += 1
                        
                        # Log progress every 50 chunks
                        if chunks_played % 50 == 0:
                            await self.errors.post(f"Played {chunks_played} chunks (broadcast {chunks_broadcast})\n".encode())
                
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    await self.errors.post(f"Audio output error: {e}\n".encode())
        
        finally:
            if stream:
                try:
                    await asyncio.to_thread(stream.stop_stream)
                    await asyncio.to_thread(stream.close)
                except Exception:
                    pass
    
    async def _auto_mode_loop(self):
        """Auto mode: continuously listen and respond"""
        if not AUDIO_AVAILABLE or not self._pya:
            await self.errors.post(b"Audio not available for auto mode\n")
            return
        
        if not STT_AVAILABLE:
            await self.errors.post(b"Speech recognition not available for auto mode\n")
            return
        
        stream = None
        
        try:
            # Open input stream
            stream = await asyncio.to_thread(
                self._pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=self.config.input_device,
                frames_per_buffer=CHUNK_SIZE,
            )
            self._input_stream = stream
            
            await self.errors.post(b"Auto mode listening started\n")
            
            while self._running and self.config.auto_mode:
                try:
                    # Read audio from microphone
                    audio_chunk = await asyncio.to_thread(
                        stream.read,
                        CHUNK_SIZE,
                        exception_on_overflow=False
                    )
                    
                    # Calculate input level
                    try:
                        import audioop
                        rms = audioop.rms(audio_chunk, 2)
                        self._input_level = min(1.0, (rms / 32768.0) * 5.0)
                    except Exception:
                        pass
                    
                    # Send to speech-to-text processor
                    await self._audio_input_queue.put(audio_chunk)
                
                except Exception as e:
                    if self._running:
                        await self.errors.post(f"Auto mode error: {e}\n".encode())
                    await asyncio.sleep(0.1)
        
        except asyncio.CancelledError:
            pass
        finally:
            if stream:
                try:
                    await asyncio.to_thread(stream.stop_stream)
                    await asyncio.to_thread(stream.close)
                except Exception:
                    pass
            self._input_stream = None


class TSConfigFile(SyntheticFile):
    """Configuration file for TS agent"""
    
    def __init__(self, agent: TSAgent):
        super().__init__("config")
        self.agent = agent
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        config_dict = self.agent.config.to_dict()
        data = (json.dumps(config_dict, indent=2) + "\n").encode()
        return data[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        if offset != 0:
            raise ValueError("Config must be written atomically")
        
        config_dict = json.loads(data.decode())
        self.agent.config = TSConfig.from_dict(config_dict)
        return len(data)


class TSStatusFile(SyntheticFile):
    """Status file for TS agent"""
    
    def __init__(self, agent: TSAgent):
        super().__init__("status")
        self.agent = agent
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        lines = [
            f"state: {self.agent.state.value}",
            f"auto_mode: {self.agent.config.auto_mode}",
            f"voice: {self.agent.config.voice}",
            f"model: {self.agent.config.model}",
            f"input_level: {self.agent._input_level:.2f}",
            f"output_level: {self.agent._output_level:.2f}",
            f"input_device: {self.agent.config.input_device or 'default'}",
            f"output_device: {self.agent.config.output_device or 'default'}",
        ]
        data = ("\n".join(lines) + "\n").encode()
        return data[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Status file is read-only")