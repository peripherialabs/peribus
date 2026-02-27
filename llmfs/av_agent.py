"""
AudioVisual Agent for LLMFS

This agent uses Gemini's Live API via raw WebSocket for real-time
audio/video streaming.  Bypasses the google-genai SDK for lower
latency on connection setup, audio send/receive, and tool calls.

Filesystem structure:
    agents/av_gemini/
    ├── ctl        # Control: start, stop, mode, voice, etc.
    ├── input      # Write text messages here (triggers response)
    ├── context    # Write context/info here (no response triggered)
    ├── OUTPUT     # Read text responses (CAPS = blocking read)
    ├── history    # Full conversation as JSON
    ├── system     # System prompt file
    ├── config     # Configuration as JSON
    ├── status     # Real-time status (audio levels, connection)
    ├── CODE       # Code output from function tool calls (CAPS = blocking read)
    ├── AUDIO      # Raw PCM audio output from AI (CAPS = blocking read)
    ├── mic        # Write raw PCM audio to send to AI
    └── errors     # Error queue
"""

import asyncio
import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum

# Core LLMFS imports
from core.files import (
    SyntheticDir, SyntheticFile, StreamFile, QueueFile,
    CtlFile, CtlHandler
)
from core.types import FidState

# WebSocket for raw Gemini Live API
try:
    from websockets.asyncio.client import connect as websocket_connect
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Optional audio/video imports
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import cv2
    import PIL.Image
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False

try:
    import mss
    SCREEN_AVAILABLE = True
except ImportError:
    SCREEN_AVAILABLE = False


# Audio configuration
if AUDIO_AVAILABLE:
    FORMAT = pyaudio.paInt16
else:
    FORMAT = 8  # paInt16 value as fallback
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# Default model
DEFAULT_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

# Gemini WebSocket endpoint
GEMINI_WEBSOCKET_HOST = "generativelanguage.googleapis.com"

# Available voices
VOICES = [
    "Aoede", "Leda", "Puck", "Charon", "Kore", "Fenrir",
    "Orion", "Perseus", "Zephyr"
]


class AVState(Enum):
    """AV Agent state"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


class VideoMode(Enum):
    """Video input mode"""
    NONE = "none"
    CAMERA = "camera"
    SCREEN = "screen"


@dataclass
class Message:
    """A single message in agent history"""
    role: str        # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    source: str = "text"  # "text", "audio", "function"


@dataclass
class AVConfig:
    """Configuration for AV agent"""
    model: str = DEFAULT_MODEL
    voice: str = "Aoede"
    video_mode: str = "none"
    system: Optional[str] = None
    functions: List[Dict] = field(default_factory=list)
    google_search: bool = True

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "voice": self.voice,
            "video_mode": self.video_mode,
            "system": self.system,
            "functions": self.functions,
            "google_search": self.google_search,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'AVConfig':
        return cls(
            model=d.get("model", DEFAULT_MODEL),
            voice=d.get("voice", "Aoede"),
            video_mode=d.get("video_mode", "none"),
            system=d.get("system"),
            functions=d.get("functions", []),
            google_search=d.get("google_search", True),
        )


class AVCtlHandler(CtlHandler):
    """Control handler for AV agent"""

    def __init__(self, agent: 'AVAgent'):
        self.agent = agent

    async def execute(self, command: str) -> Optional[str]:
        parts = command.split(' ', 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "start":
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

        elif cmd == "model":
            if arg:
                self.agent.config.model = arg
                return f"Model set to {arg}"
            return self.agent.config.model

        elif cmd == "voice":
            if arg:
                if arg not in VOICES:
                    raise ValueError(f"Unknown voice: {arg}. Available: {', '.join(VOICES)}")
                self.agent.config.voice = arg
                return f"Voice set to {arg}"
            return self.agent.config.voice

        elif cmd == "mode" or cmd == "video":
            if arg:
                try:
                    VideoMode(arg)
                    self.agent.config.video_mode = arg
                    return f"Video mode set to {arg}"
                except ValueError:
                    raise ValueError(f"Unknown mode: {arg}. Available: none, camera, screen")
            return self.agent.config.video_mode

        elif cmd == "system":
            if arg:
                self.agent.config.system = arg
                return "System prompt set"
            return self.agent.config.system or "(none)"

        elif cmd == "clear":
            await self.agent.clear()
            return "History cleared"

        elif cmd == "mute":
            self.agent._mute_local = True
            return "Local playback muted"

        elif cmd == "unmute":
            self.agent._mute_local = False
            return "Local playback unmuted"

        elif cmd == "devices":
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
            if arg:
                try:
                    self.agent._output_device = int(arg)
                    return f"Output device set to {arg} (restart to apply)"
                except ValueError:
                    raise ValueError("Usage: output <device_index>")
            return str(self.agent._output_device) if self.agent._output_device is not None else "(default)"

        elif cmd == "input":
            if arg:
                try:
                    self.agent._input_device = int(arg)
                    return f"Input device set to {arg} (restart to apply)"
                except ValueError:
                    raise ValueError("Usage: input <device_index>")
            return str(self.agent._input_device) if self.agent._input_device is not None else "(default)"

        elif cmd == "voices":
            return "\n".join(VOICES)

        else:
            raise ValueError(
                f"Unknown command: {cmd}. "
                "Available: start, stop, restart, model, voice, mode, system, "
                "clear, mute, unmute, voices, devices, output, input"
            )

    async def get_status(self) -> bytes:
        a = self.agent
        lines = [
            f"state {a.state.value}",
            f"model {a.config.model}",
            f"voice {a.config.voice}",
            f"video {a.config.video_mode}",
            f"messages {len(a.history)}",
        ]

        if a.config.system:
            sys_preview = a.config.system[:50] + "..." if len(a.config.system) > 50 else a.config.system
            lines.append(f"system {sys_preview}")

        if a.state == AVState.ERROR and a.last_error:
            lines.append(f"error {a.last_error}")

        if a.state == AVState.STREAMING:
            lines.append(f"mic_level {a._mic_level:.2f}")
            lines.append(f"output_level {a._output_level:.2f}")

        lines.append(f"input_device {a._input_device if a._input_device is not None else 'default'}")
        lines.append(f"output_device {a._output_device if a._output_device is not None else 'default'}")

        if AUDIO_AVAILABLE:
            lines.append("")
            lines.append("# Audio Devices:")
            try:
                pya = pyaudio.PyAudio()
                for i in range(pya.get_device_count()):
                    info = pya.get_device_info_by_index(i)
                    direction = []
                    if info['maxInputChannels'] > 0:
                        direction.append("in")
                    if info['maxOutputChannels'] > 0:
                        direction.append("out")
                    lines.append(f"#   {i}: {info['name']} ({','.join(direction)})")
                pya.terminate()
            except Exception as e:
                lines.append(f"#   (error listing devices: {e})")

        return ("\n".join(lines) + "\n").encode()


# ─── Filesystem Files ──────────────────────────────────────────────


class AVInputFile(SyntheticFile):
    """Write text messages to send during live session."""

    def __init__(self, agent: 'AVAgent'):
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
        await self.agent.send_text(text)
        return len(data)


class AVContextFile(SyntheticFile):
    """
    Write context/information to the agent without triggering a response.

    Sends clientContent with turnComplete=false directly (no mic pause
    needed — Gemini handles interleaving).

    Usage:
        echo "The user's name is Alice" > /mnt/llm/av/context
        echo '{"temperature": 22, "city": "NYC"}' > /mnt/llm/av/context
    """

    def __init__(self, agent: 'AVAgent'):
        super().__init__("context")
        self.agent = agent
        self._context_blocks: List[str] = []

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        data = ("\n".join(self._context_blocks) + "\n").encode() if self._context_blocks else b""
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        text = data.decode('utf-8').strip()
        if not text:
            return len(data)

        if text.lower() == "clear":
            self._context_blocks.clear()
            return len(data)

        self._context_blocks.append(text)
        await self.agent.send_context(text)
        return len(data)

    async def getattr(self):
        data = ("\n".join(self._context_blocks) + "\n").encode() if self._context_blocks else b""
        return {"st_size": len(data)}


class AVHistoryFile(SyntheticFile):
    """Read agent history as JSON."""

    def __init__(self, agent: 'AVAgent'):
        super().__init__("history")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        history = [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "source": m.source
            }
            for m in self.agent.history
        ]
        data = json.dumps(history, indent=2).encode()
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        try:
            history = json.loads(data.decode())
            self.agent.history = [
                Message(
                    role=m["role"],
                    content=m["content"],
                    timestamp=m.get("timestamp", time.time()),
                    source=m.get("source", "text")
                )
                for m in history
            ]
            return len(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")


class AVConfigFile(SyntheticFile):
    """Read/write agent configuration as JSON."""

    def __init__(self, agent: 'AVAgent'):
        super().__init__("config")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        data = json.dumps(self.agent.config.to_dict(), indent=2).encode()
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        try:
            config = json.loads(data.decode())
            self.agent.config = AVConfig.from_dict(config)
            return len(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")


class AVSystemFile(SyntheticFile):
    """Directly read/write the system prompt"""

    def __init__(self, agent: 'AVAgent'):
        super().__init__("system")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        val = (self.agent.config.system or "").encode()
        return val[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        current_data = bytearray((self.agent.config.system or "").encode('utf-8'))
        if offset > len(current_data):
            current_data.extend(b'\0' * (offset - len(current_data)))
        current_data[offset:offset + len(data)] = data
        self.agent.config.system = current_data.decode('utf-8', errors='replace')
        return len(data)


class AVStatusFile(SyntheticFile):
    """Real-time status including audio levels"""

    def __init__(self, agent: 'AVAgent'):
        super().__init__("status")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        status = {
            "state": self.agent.state.value,
            "mic_level": self.agent._mic_level,
            "output_level": self.agent._output_level,
            "connected": self.agent.state in (AVState.CONNECTED, AVState.STREAMING),
            "streaming": self.agent.state == AVState.STREAMING,
        }
        data = json.dumps(status).encode()
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Status file is read-only")


class AVCodeFile(SyntheticFile):
    """
    Code output from function tool calls.

    STATE-AWARE BLOCKING (enables `while true; do cat $av_gemini/code; done`):

    1. WAITING: read() blocks until set_code() fires
    2. READY: read() returns content
    3. CONSUMED: read() returns b"" (EOF — cat exits)
    4. Next read at offset 0: rearms, blocks again at step 1
    """

    def __init__(self, agent: 'AVAgent'):
        super().__init__("CODE")
        self.agent = agent
        self._code = ""
        self._code_history: List[Dict[str, Any]] = []
        self._content_ready = asyncio.Event()
        self._content_consumed = False
        self._lock = asyncio.Lock()

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        if offset == 0 and self._content_consumed:
            async with self._lock:
                if self._content_consumed:
                    self._content_consumed = False
                    self._content_ready.clear()

        await self._content_ready.wait()

        async with self._lock:
            data = (self._code + "\n").encode() if self._code else b""
            chunk = data[offset:offset + count]
            if offset + len(chunk) >= len(data):
                self._content_consumed = True
            return chunk

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        self._code = data.decode('utf-8')
        self._content_ready.set()
        return len(data)

    async def set_code(self, code: str, function_name: str = "", call_id: str = ""):
        async with self._lock:
            self._code = code
            self._content_consumed = False
        self._code_history.append({
            "code": code,
            "function_name": function_name,
            "call_id": call_id,
            "timestamp": time.time(),
        })
        self._content_ready.set()

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self._code_history)


class AVAudioOutFile(SyntheticFile):
    """
    Raw PCM audio output from the AI.
    Format: 16-bit signed PCM, mono, 24000 Hz
    """

    def __init__(self, agent: 'AVAgent'):
        super().__init__("AUDIO")
        self.agent = agent
        self._reader_queues: Dict[int, asyncio.Queue] = {}

    async def open(self, fid: FidState, mode: int):
        self._reader_queues[fid.fid] = asyncio.Queue(maxsize=100)

    async def close(self, fid: FidState):
        self._reader_queues.pop(fid.fid, None)

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        queue = self._reader_queues.get(fid.fid)
        if queue is None:
            queue = asyncio.Queue(maxsize=100)
            self._reader_queues[fid.fid] = queue

        chunks = []
        bytes_read = 0

        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=30.0)
            chunks.append(chunk)
            bytes_read += len(chunk)
        except asyncio.TimeoutError:
            return b""

        while bytes_read < count:
            try:
                chunk = queue.get_nowait()
                chunks.append(chunk)
                bytes_read += len(chunk)
            except asyncio.QueueEmpty:
                break

        data = b"".join(chunks)
        return data[:count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Audio output is read-only")

    async def broadcast(self, data: bytes):
        for queue in self._reader_queues.values():
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(data)
                except:
                    pass


class AVAudioInFile(SyntheticFile):
    """
    Write raw PCM audio to send to the AI.
    Format: 16-bit signed PCM, mono, 16000 Hz
    """

    def __init__(self, agent: 'AVAgent'):
        super().__init__("mic")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        return b""

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        if self.agent._websocket and self.agent.state == AVState.STREAMING:
            audio_b64 = base64.b64encode(data).decode('ascii')
            msg = {
                "realtimeInput": {
                    "audio": {
                        "data": audio_b64,
                        "mimeType": "audio/pcm",
                    }
                }
            }
            try:
                await self.agent._websocket.send(json.dumps(msg))
            except Exception:
                pass
        return len(data)


# ─── Main Agent ─────────────────────────────────────────────────────


class AVAgent(SyntheticDir):
    """
    AudioVisual Agent using Gemini Live API via raw WebSocket.

    Bypasses the google-genai Python SDK entirely for lower latency.
    Uses the BidiGenerateContent WebSocket protocol directly.
    """

    def __init__(
        self,
        name: str = "av_gemini",
        route_manager: 'RouteManager' = None,
        function_registry: Dict[str, Callable] = None
    ):
        super().__init__(name)
        self.agent_name = name
        self.route_manager = route_manager
        self.function_registry = function_registry or {}

        # State
        self.state = AVState.DISCONNECTED
        self.history: List[Message] = []
        self.last_error: Optional[str] = None

        # Audio levels
        self._mic_level = 0.0
        self._output_level = 0.0
        self._mute_local = False
        self._output_device = None
        self._input_device = None

        # Configuration
        self.config = AVConfig()

        # Session management
        self._websocket = None
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._audio_in_queue: Optional[asyncio.Queue] = None
        self._pending_message: Optional[str] = None
        self._interrupting: bool = False
        self._mic_paused: bool = False
        self._text_message_event: asyncio.Event = asyncio.Event()

        # PyAudio
        self._pya = None
        self._audio_stream = None
        self._output_stream = None

        # Child files
        self.output = StreamFile("OUTPUT")
        self.errors = QueueFile("errors")
        self.audio_out_file = AVAudioOutFile(self)
        self.code_file = AVCodeFile(self)
        self.context_file = AVContextFile(self)

        self.add(CtlFile("ctl", AVCtlHandler(self)))
        self.add(AVInputFile(self))
        self.add(self.context_file)
        self.add(self.output)
        self.add(AVHistoryFile(self))
        self.add(AVConfigFile(self))
        self.add(AVSystemFile(self))
        self.add(AVStatusFile(self))
        self.add(self.code_file)
        self.add(self.audio_out_file)
        self.add(AVAudioInFile(self))
        self.add(self.errors)

    def _check_dependencies(self):
        if not WEBSOCKET_AVAILABLE:
            raise RuntimeError("websockets package not installed")
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_API_KEY not set")
        if not AUDIO_AVAILABLE:
            raise RuntimeError("pyaudio not installed")

    def _build_websocket_url(self) -> str:
        """Build the raw WebSocket URL for Gemini Live API"""
        api_key = os.getenv("GEMINI_API_KEY")
        return (
            f"wss://{GEMINI_WEBSOCKET_HOST}/ws/"
            f"google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
            f"?key={api_key}"
        )

    def _build_setup_message(self) -> dict:
        """Build the BidiGenerateContentSetup JSON message"""
        model = self.config.model
        if not model.startswith("models/"):
            model = f"models/{model}"

        generation_config = {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": self.config.voice,
                    }
                }
            },
        }

        setup = {
            "model": model,
            "generationConfig": generation_config,
        }

        # System instruction
        if self.config.system:
            setup["systemInstruction"] = {
                "role": "user",
                "parts": [{"text": self.config.system}]
            }

        # Tools
        tools = []

        # Function declarations (Gemini format)
        if self.config.functions:
            tools.append({
                "functionDeclarations": self.config.functions
            })

        # Google Search
        if self.config.google_search:
            tools.append({"googleSearch": {}})

        if tools:
            setup["tools"] = tools

        return {"setup": setup}

    async def start(self):
        if self.state in (AVState.CONNECTED, AVState.STREAMING, AVState.CONNECTING):
            await self.errors.post(b"Already running\n")
            return

        try:
            self._check_dependencies()
        except RuntimeError as e:
            self.state = AVState.ERROR
            self.last_error = str(e)
            await self.errors.post(f"{e}\n".encode())
            return

        self.state = AVState.CONNECTING
        self._running = True
        asyncio.create_task(self._run_session())

    async def stop(self):
        self._running = False

        for task in self._tasks:
            if not task.done():
                task.cancel()
        self._tasks.clear()

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None

        await self._cleanup_audio()
        self.state = AVState.DISCONNECTED

    async def _cleanup_audio(self):
        if self._audio_stream:
            try:
                if self._audio_stream.is_active():
                    self._audio_stream.stop_stream()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None

        if self._output_stream:
            try:
                if self._output_stream.is_active():
                    self._output_stream.stop_stream()
                self._output_stream.close()
            except Exception:
                pass
            self._output_stream = None

        if self._pya:
            try:
                self._pya.terminate()
            except Exception:
                pass
            self._pya = None

    async def _run_session(self):
        """Main session loop — raw WebSocket to Gemini Live API"""
        try:
            self._pya = pyaudio.PyAudio()

            ws_url = self._build_websocket_url()
            await self.errors.post(b"Connecting to Gemini Live API (raw WS)...\n")

            self._websocket = await websocket_connect(ws_url)

            # Send setup message
            setup_msg = self._build_setup_message()
            await self._websocket.send(json.dumps(setup_msg))

            # Wait for setupComplete
            raw = await asyncio.wait_for(self._websocket.recv(), timeout=10.0)
            resp = json.loads(raw)
            if "setupComplete" in resp:
                await self.errors.post(b"Session configured (setupComplete)\n")
            else:
                await self.errors.post(
                    f"Setup response: {json.dumps(resp)[:200]}\n".encode()
                )

            self.state = AVState.CONNECTED
            self._audio_in_queue = asyncio.Queue()

            await asyncio.sleep(0.1)

            async with asyncio.TaskGroup() as tg:
                self._tasks = [
                    tg.create_task(self._listen_audio()),
                    tg.create_task(self._receive_messages()),
                    tg.create_task(self._play_audio()),
                    tg.create_task(self._handle_text_messages()),
                ]

                if self.config.video_mode == "camera" and CAMERA_AVAILABLE:
                    self._tasks.append(tg.create_task(self._get_camera_frames()))
                elif self.config.video_mode == "screen" and SCREEN_AVAILABLE:
                    self._tasks.append(tg.create_task(self._get_screen_frames()))

                self.state = AVState.STREAMING

                while self._running:
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.state = AVState.ERROR
            self.last_error = str(e)
            await self.errors.post(f"Session error: {e}\n".encode())
        finally:
            if self._websocket:
                try:
                    await self._websocket.close()
                except Exception:
                    pass
                self._websocket = None
            await self._cleanup_audio()
            if self.state != AVState.ERROR:
                self.state = AVState.DISCONNECTED

    # ─── Audio I/O ──────────────────────────────────────────────────

    async def _listen_audio(self):
        """Listen to microphone and send base64 PCM to Gemini via WebSocket"""
        try:
            if self._input_device is not None:
                mic_info = self._pya.get_device_info_by_index(self._input_device)
            else:
                mic_info = self._pya.get_default_input_device_info()
            await self.errors.post(f"Using mic: {mic_info['name']}\n".encode())

            sample_rate = SEND_SAMPLE_RATE
            try:
                self._pya.is_format_supported(
                    sample_rate,
                    input_device=mic_info["index"],
                    input_channels=CHANNELS,
                    input_format=FORMAT,
                )
            except ValueError:
                await self.errors.post(b"16kHz not supported, trying default rate\n")
                sample_rate = int(mic_info['defaultSampleRate'])

            stream = await asyncio.to_thread(
                self._pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=sample_rate,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
            )
            self._audio_stream = stream
            self._actual_input_rate = sample_rate
            await self.errors.post(f"Mic opened at {sample_rate}Hz\n".encode())

            resample_state = None

            while self._running:
                data = await asyncio.to_thread(
                    stream.read, CHUNK_SIZE, exception_on_overflow=False
                )

                # Mic level
                if len(data) > 0:
                    try:
                        import audioop
                        rms = audioop.rms(data, 2)
                        self._mic_level = min(1.0, (rms / 32768.0) * 5.0)
                    except Exception:
                        pass

                # Resample to 16kHz if needed
                if sample_rate != SEND_SAMPLE_RATE and len(data) > 0:
                    try:
                        import audioop
                        data, resample_state = audioop.ratecv(
                            data, 2, CHANNELS,
                            sample_rate, SEND_SAMPLE_RATE,
                            resample_state,
                        )
                    except Exception as e:
                        await self.errors.post(f"Resample error: {e}\n".encode())

                # Send as base64 over WebSocket (Gemini realtimeInput format)
                # Skip sending mic audio while paused (text input in progress).
                # Sending realtimeInput concurrently with clientContent causes
                # Gemini to not process the text turn reliably.
                if self._mic_paused:
                    await asyncio.sleep(0.01)
                    continue

                audio_b64 = base64.b64encode(data).decode('ascii')
                msg = {
                    "realtimeInput": {
                        "audio": {
                            "data": audio_b64,
                            "mimeType": "audio/pcm",
                        }
                    }
                }
                try:
                    await self._websocket.send(json.dumps(msg))
                except Exception as e:
                    if self._running:
                        await self.errors.post(f"Send audio error: {e}\n".encode())
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                await self.errors.post(f"Audio input error: {e}\n".encode())

    async def _receive_messages(self):
        """Receive all messages from Gemini WebSocket — audio, text, tool calls"""
        try:
            async for raw_message in self._websocket:
                if not self._running:
                    break

                response = json.loads(raw_message)

                # ─── Tool calls ───
                tool_call = response.get("toolCall")
                if tool_call is not None:
                    await self._handle_tool_call(tool_call)
                    continue

                # ─── Tool call cancellation ───
                if "toolCallCancellation" in response:
                    cancelled_ids = response["toolCallCancellation"].get("ids", [])
                    if cancelled_ids:
                        await self.errors.post(
                            f"Tool calls cancelled: {cancelled_ids}\n".encode()
                        )
                    continue

                # ─── Server content ───
                server_content = response.get("serverContent")
                if server_content is None:
                    continue

                # Model turn — audio and text parts
                model_turn = server_content.get("modelTurn")
                if model_turn:
                    parts = model_turn.get("parts", [])
                    for part in parts:
                        # Audio data (inlineData)
                        inline_data = part.get("inlineData")
                        if inline_data:
                            # Drop audio while we're interrupting for a text input
                            if self._interrupting:
                                continue
                            b64data = inline_data.get("data", "")
                            if b64data:
                                pcm_data = base64.b64decode(b64data)
                                if self._audio_in_queue:
                                    self._audio_in_queue.put_nowait(pcm_data)
                            continue

                        # Text content
                        text = part.get("text")
                        if text:
                            self.history.append(Message(
                                role="assistant",
                                content=text,
                                source="audio",
                            ))
                            await self.output.append(text.encode())
                            if self.route_manager:
                                await self.route_manager.broadcast(text.encode())

                # Output transcription
                output_tx = server_content.get("outputTranscription")
                if output_tx:
                    text = output_tx.get("text", "")
                    if text:
                        await self.output.append(text.encode())

                # Input transcription
                input_tx = server_content.get("inputTranscription")
                if input_tx:
                    text = input_tx.get("text", "")
                    if text:
                        self.history.append(Message(
                            role="user",
                            content=text,
                            source="audio",
                        ))

                # Turn complete — flush audio queue, clear interrupt flag,
                # and resume mic if it was paused for text input.
                if server_content.get("turnComplete"):
                    while self._audio_in_queue and not self._audio_in_queue.empty():
                        self._audio_in_queue.get_nowait()
                    self._interrupting = False
                    if self._mic_paused:
                        self._mic_paused = False

                # Interrupted
                if server_content.get("interrupted"):
                    while self._audio_in_queue and not self._audio_in_queue.empty():
                        self._audio_in_queue.get_nowait()
                    self._interrupting = False
                    if self._mic_paused:
                        self._mic_paused = False

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                await self.errors.post(f"Receive error: {e}\n".encode())

    async def _handle_tool_call(self, tool_call: dict):
        """Handle function calls from Gemini and send results back"""
        function_calls = tool_call.get("functionCalls", [])
        function_responses = []

        for fc in function_calls:
            func_name = fc.get("name", "")
            func_id = fc.get("id", "")
            args = fc.get("args", {})

            if not func_name:
                continue

            await self.errors.post(
                f"Function call: {func_name}({json.dumps(args)[:200]})\n".encode()
            )

            # Write code to code file if present
            code_content = args.get("code", "")
            if code_content:
                await self.code_file.set_code(
                    code_content,
                    function_name=func_name,
                    call_id=func_id,
                )
                await self.errors.post(
                    f"Code written to code file ({len(code_content)} chars)\n".encode()
                )

            # Execute function if registered
            result = {"result": "no handler registered", "function": func_name}

            if func_name in self.function_registry:
                try:
                    func = self.function_registry[func_name]
                    import inspect
                    if inspect.iscoroutinefunction(func):
                        result = await func(**args)
                    else:
                        result = await asyncio.to_thread(func, **args)
                except Exception as e:
                    result = {"result": "error", "message": str(e)}

            # Log to history
            self.history.append(Message(
                role="assistant",
                content=f"[function:{func_name}] {json.dumps(result)[:500]}",
                source="function",
            ))

            fr = {"name": func_name, "response": result}
            if func_id:
                fr["id"] = func_id
            function_responses.append(fr)

        # Send tool response back (BidiGenerateContentToolResponse)
        if self._websocket and function_responses:
            tool_response_msg = {
                "toolResponse": {
                    "functionResponses": function_responses
                }
            }
            try:
                await self._websocket.send(json.dumps(tool_response_msg))
            except Exception as e:
                await self.errors.post(
                    f"Error sending tool response: {e}\n".encode()
                )

    async def _play_audio(self):
        """Play audio responses and broadcast for filesystem access"""
        try:
            stream = None
            try:
                if self._output_device is not None:
                    out_info = self._pya.get_device_info_by_index(self._output_device)
                else:
                    out_info = self._pya.get_default_output_device_info()

                await self.errors.post(f"Using output: {out_info['name']}\n".encode())

                sample_rate = RECEIVE_SAMPLE_RATE
                try:
                    self._pya.is_format_supported(
                        sample_rate,
                        output_device=out_info["index"],
                        output_channels=CHANNELS,
                        output_format=FORMAT,
                    )
                except ValueError:
                    await self.errors.post(b"24kHz not supported, trying default rate\n")
                    sample_rate = int(out_info['defaultSampleRate'])

                stream = await asyncio.to_thread(
                    self._pya.open,
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=sample_rate,
                    output=True,
                    output_device_index=out_info["index"],
                )
                self._output_stream = stream
                self._actual_output_rate = sample_rate
                await self.errors.post(f"Output opened at {sample_rate}Hz\n".encode())
            except Exception as e:
                await self.errors.post(f"Failed to open output stream: {e}\n".encode())

            output_resample_state = None
            chunks_played = 0

            while self._running:
                try:
                    bytestream = await self._audio_in_queue.get()

                    if len(bytestream) > 0:
                        try:
                            import audioop
                            rms = audioop.rms(bytestream, 2)
                            self._output_level = min(1.0, (rms / 32768.0) * 5.0)
                        except Exception:
                            pass

                    # Broadcast to filesystem readers
                    await self.audio_out_file.broadcast(bytestream)

                    # Play through speakers (unless muted)
                    if stream and not self._mute_local:
                        audio_to_play = bytestream
                        if hasattr(self, '_actual_output_rate') and self._actual_output_rate != RECEIVE_SAMPLE_RATE:
                            try:
                                import audioop
                                audio_to_play, output_resample_state = audioop.ratecv(
                                    bytestream, 2, CHANNELS,
                                    RECEIVE_SAMPLE_RATE, self._actual_output_rate,
                                    output_resample_state,
                                )
                            except Exception as e:
                                await self.errors.post(f"Resample out error: {e}\n".encode())

                        await asyncio.to_thread(stream.write, audio_to_play)
                        chunks_played += 1

                        if chunks_played % 100 == 0:
                            await self.errors.post(
                                f"Played {chunks_played} audio chunks\n".encode()
                            )

                except asyncio.QueueEmpty:
                    continue
                except Exception as e:
                    if self._running:
                        await self.errors.post(f"Audio play error: {e}\n".encode())

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                await self.errors.post(f"Audio output error: {e}\n".encode())

    async def _handle_text_messages(self):
        """Handle text messages from filesystem input file"""
        while self._running:
            try:
                # Wait for a message to be signalled
                await self._text_message_event.wait()
                self._text_message_event.clear()

                if self._pending_message:
                    message = self._pending_message
                    self._pending_message = None

                    if message and self._websocket:
                        self.history.append(Message(
                            role="user",
                            content=message,
                            source="text",
                        ))

                        # Gemini clientContent format
                        text_msg = {
                            "clientContent": {
                                "turnComplete": True,
                                "turns": [
                                    {
                                        "role": "user",
                                        "parts": [{"text": message}]
                                    }
                                ]
                            }
                        }
                        await self._websocket.send(json.dumps(text_msg))

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    await self.errors.post(f"Message error: {e}\n".encode())

    # ─── Video capture ──────────────────────────────────────────────

    async def _get_camera_frames(self):
        """Capture frames from camera and send as realtimeInput video"""
        cap = None
        try:
            cap = await asyncio.to_thread(cv2.VideoCapture, 0)
            while self._running:
                ret, frame = await asyncio.to_thread(cap.read)
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(frame_rgb)
                img.thumbnail([1024, 1024])

                image_io = io.BytesIO()
                img.save(image_io, format="jpeg")
                image_io.seek(0)
                image_bytes = image_io.read()
                image_b64 = base64.b64encode(image_bytes).decode('ascii')

                msg = {
                    "realtimeInput": {
                        "video": {
                            "data": image_b64,
                            "mimeType": "image/jpeg",
                        }
                    }
                }

                try:
                    await self._websocket.send(json.dumps(msg))
                except Exception as e:
                    if self._running:
                        await self.errors.post(f"Send video error: {e}\n".encode())
                    break

                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                await self.errors.post(f"Camera error: {e}\n".encode())
        finally:
            if cap:
                cap.release()

    async def _get_screen_frames(self):
        """Capture screen frames and send as realtimeInput video"""
        try:
            sct = mss.mss()
            while self._running:
                monitor = sct.monitors[0]
                screenshot = sct.grab(monitor)

                image_bytes = mss.tools.to_png(screenshot.rgb, screenshot.size)
                img = PIL.Image.open(io.BytesIO(image_bytes))

                image_io = io.BytesIO()
                img.save(image_io, format="jpeg")
                image_io.seek(0)
                image_bytes = image_io.read()
                image_b64 = base64.b64encode(image_bytes).decode('ascii')

                msg = {
                    "realtimeInput": {
                        "video": {
                            "data": image_b64,
                            "mimeType": "image/jpeg",
                        }
                    }
                }

                try:
                    await self._websocket.send(json.dumps(msg))
                except Exception as e:
                    if self._running:
                        await self.errors.post(f"Send screen error: {e}\n".encode())
                    break

                await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                await self.errors.post(f"Screen capture error: {e}\n".encode())

    # ─── Public API ─────────────────────────────────────────────────

    async def send_text(self, text: str):
        """Send text message to the session.

        Mirrors the SDK's session.send(input=text, end_of_turn=True):
          - Interrupt any current model output (drain playback queue)
          - Send clientContent with turnComplete=true
          - Mic keeps streaming (no pause needed, SDK never pauses mic)
        """
        if self.state not in (AVState.CONNECTED, AVState.STREAMING):
            await self.errors.post(b"Not connected\n")
            return

        # Interrupt current model output (SDK does this internally)
        self._interrupting = True

        # Drain playback queue to silence current speech immediately
        if self._audio_in_queue:
            while not self._audio_in_queue.empty():
                try:
                    self._audio_in_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Stop/restart output stream to flush buffered audio
        if self._output_stream:
            try:
                if self._output_stream.is_active():
                    await asyncio.to_thread(self._output_stream.stop_stream)
                    await asyncio.to_thread(self._output_stream.start_stream)
            except Exception:
                pass

        # Dispatch the text message (sent by _handle_text_messages)
        # Mic keeps running — no pause needed, matching SDK behavior
        self._pending_message = text
        self._text_message_event.set()

    async def send_context(self, text: str):
        """
        Inject context into the session without triggering a response.

        Sends clientContent with turnComplete=false directly.  No mic
        pause needed — the SDK never pauses the mic for text sends.
        """
        if self.state not in (AVState.CONNECTED, AVState.STREAMING):
            await self.errors.post(b"Not connected\n")
            return

        if not self._websocket:
            await self.errors.post(b"No websocket connection\n")
            return

        self.history.append(Message(
            role="user",
            content=text,
            source="context",
        ))

        context_msg = {
            "clientContent": {
                "turnComplete": False,
                "turns": [
                    {
                        "role": "user",
                        "parts": [{"text": f"[CONTEXT] {text}"}]
                    }
                ]
            }
        }
        try:
            await self._websocket.send(json.dumps(context_msg))
            await self.errors.post(f"Context injected ({len(text)} chars)\n".encode())
        except Exception as e:
            await self.errors.post(f"Context send error: {e}\n".encode())

    async def clear(self):
        self.history.clear()
        await self.output.reset()

    async def cancel(self):
        await self.stop()


def register_av_function(registry: Dict[str, Callable], name: str, func: Callable):
    """Register a function that can be called by the AV agent"""
    registry[name] = func