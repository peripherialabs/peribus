"""
AudioVisual OpenAI Agent for LLMFS

This agent uses OpenAI's Realtime API (WebSocket) for real-time audio streaming
with function calling support. Code from function tools is delivered to
the agent's `code` file.

The OpenAI Realtime API GA uses the same event protocol that the Grok agent
implements (Grok cloned OpenAI's format). Key differences from the Grok agent:
  - Endpoint: wss://api.openai.com/v1/realtime?model=<model>
  - Auth: Authorization: Bearer <OPENAI_API_KEY>
  - Session config nests audio settings under `audio.input` / `audio.output`
  - Voice options differ (alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar)
  - Model: gpt-realtime (GA) or gpt-4o-realtime-preview

Filesystem structure:
    agents/openai_av/
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

# WebSocket for OpenAI Realtime
try:
    from websockets.asyncio.client import connect as websocket_connect
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Optional audio imports
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


# Audio configuration — OpenAI Realtime uses 24kHz PCM16 for both input and output
if AUDIO_AVAILABLE:
    FORMAT = pyaudio.paInt16
else:
    FORMAT = 8  # paInt16 value as fallback
CHANNELS = 1
SEND_SAMPLE_RATE = 24000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# OpenAI Realtime WebSocket endpoint
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

# Default model
DEFAULT_MODEL = "gpt-realtime-1.5"

# Available voices (GA)
OPENAI_VOICES = [
    "alloy", "ash", "ballad", "coral", "echo",
    "sage", "shimmer", "verse", "marin", "cedar"
]


class OpenAIAVState(Enum):
    """OpenAI AV Agent state"""
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
class OpenAIAVConfig:
    """Configuration for OpenAI AV agent"""
    model: str = DEFAULT_MODEL
    voice: str = "marin"
    video_mode: str = "none"
    system: Optional[str] = None
    functions: List[Dict] = field(default_factory=list)
    temperature: float = 0.8
    turn_detection_threshold: float = 0.6
    silence_duration_ms: int = 500
    prefix_padding_ms: int = 300
    tool_choice: str = "auto"
    noise_reduction: Optional[str] = "near_field"  # "near_field" or "far_field" or None
    speed: float = 1.0  # 0.25 to 1.5

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "voice": self.voice,
            "video_mode": self.video_mode,
            "system": self.system,
            "functions": self.functions,
            "temperature": self.temperature,
            "turn_detection_threshold": self.turn_detection_threshold,
            "silence_duration_ms": self.silence_duration_ms,
            "prefix_padding_ms": self.prefix_padding_ms,
            "tool_choice": self.tool_choice,
            "noise_reduction": self.noise_reduction,
            "speed": self.speed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'OpenAIAVConfig':
        return cls(
            model=d.get("model", DEFAULT_MODEL),
            voice=d.get("voice", "marin"),
            video_mode=d.get("video_mode", "none"),
            system=d.get("system"),
            functions=d.get("functions", []),
            temperature=d.get("temperature", 0.8),
            turn_detection_threshold=d.get("turn_detection_threshold", 0.5),
            silence_duration_ms=d.get("silence_duration_ms", 200),
            prefix_padding_ms=d.get("prefix_padding_ms", 300),
            tool_choice=d.get("tool_choice", "auto"),
            noise_reduction=d.get("noise_reduction"),
            speed=d.get("speed", 1.0),
        )


# ─── Filesystem Files ──────────────────────────────────────────────


class OpenAIAVCtlHandler(CtlHandler):
    """Control handler for OpenAI AV agent"""

    def __init__(self, agent: 'OpenAIAVAgent'):
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
                if arg not in OPENAI_VOICES:
                    raise ValueError(f"Unknown voice: {arg}. Available: {', '.join(OPENAI_VOICES)}")
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
            return "\n".join(OPENAI_VOICES)

        elif cmd == "temperature":
            if arg:
                self.agent.config.temperature = float(arg)
                return f"Temperature set to {arg}"
            return str(self.agent.config.temperature)

        elif cmd == "speed":
            if arg:
                val = float(arg)
                if val < 0.25 or val > 1.5:
                    raise ValueError("Speed must be between 0.25 and 1.5")
                self.agent.config.speed = val
                return f"Speed set to {val}"
            return str(self.agent.config.speed)

        elif cmd == "bargein":
            if arg:
                try:
                    val = float(arg)
                    self.agent._barge_in_threshold = val
                    return f"Barge-in threshold set to {val}"
                except ValueError:
                    raise ValueError("Usage: bargein <threshold_0.0_to_1.0>")
            return f"{self.agent._barge_in_threshold}"

        elif cmd == "interrupt":
            # Manual interrupt — flush playback immediately
            self.agent._interrupted.set()
            self.agent._model_speaking = False
            return "Interrupted"

        elif cmd == "noise":
            if arg:
                if arg in ("near_field", "far_field"):
                    self.agent.config.noise_reduction = arg
                    return f"Noise reduction set to {arg}"
                elif arg in ("off", "none", "null"):
                    self.agent.config.noise_reduction = None
                    return "Noise reduction disabled"
                else:
                    raise ValueError("Usage: noise <near_field|far_field|off>")
            return self.agent.config.noise_reduction or "off"

        else:
            raise ValueError(
                f"Unknown command: {cmd}. "
                "Available: start, stop, restart, model, voice, mode, system, clear, "
                "mute, unmute, voices, devices, output, input, temperature, speed, "
                "bargein, interrupt, noise"
            )

    async def get_status(self) -> bytes:
        a = self.agent
        lines = [
            f"state {a.state.value}",
            f"model {a.config.model}",
            f"voice {a.config.voice}",
            f"video {a.config.video_mode}",
            f"messages {len(a.history)}",
            f"temperature {a.config.temperature}",
            f"speed {a.config.speed}",
        ]

        if a.config.system:
            sys_preview = a.config.system[:50] + "..." if len(a.config.system) > 50 else a.config.system
            lines.append(f"system {sys_preview}")

        if a.state == OpenAIAVState.ERROR and a.last_error:
            lines.append(f"error {a.last_error}")

        if a.state == OpenAIAVState.STREAMING:
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


class OpenAIAVInputFile(SyntheticFile):
    """Write text messages to send during live session."""

    def __init__(self, agent: 'OpenAIAVAgent'):
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


class OpenAIAVContextFile(SyntheticFile):
    """
    Write context/information to the agent without triggering a response.

    Uses a two-pronged approach for reliability:
      1. conversation.item.create — injects the context as a user message
         into the conversation history (the model sees it in context)
      2. session.update — appends context to instructions as a fallback

    No response.create is sent, so the model does not reply immediately.
    The context will be visible on the next turn.

    Usage:
        echo "The user's name is Alice" > /mnt/llm/openai_av/context
        echo '{"temperature": 22, "city": "NYC"}' > /mnt/llm/openai_av/context

    Reading returns all accumulated context blocks.
    Writing with just "clear" resets the context.
    """

    CONTEXT_SEPARATOR = "\n\n---\n[CONTEXT]\n"

    def __init__(self, agent: 'OpenAIAVAgent'):
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
            await self.agent._update_instructions()
            return len(data)

        self._context_blocks.append(text)
        await self.agent.send_context(text)
        return len(data)

    async def getattr(self):
        data = ("\n".join(self._context_blocks) + "\n").encode() if self._context_blocks else b""
        return {"st_size": len(data)}


class OpenAIAVHistoryFile(SyntheticFile):
    """Read agent history as JSON."""

    def __init__(self, agent: 'OpenAIAVAgent'):
        super().__init__("history")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        history = [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "source": m.source,
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
                    source=m.get("source", "text"),
                )
                for m in history
            ]
            return len(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")


class OpenAIAVConfigFile(SyntheticFile):
    """Read/write agent configuration as JSON."""

    def __init__(self, agent: 'OpenAIAVAgent'):
        super().__init__("config")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        data = json.dumps(self.agent.config.to_dict(), indent=2).encode()
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        try:
            config = json.loads(data.decode())
            self.agent.config = OpenAIAVConfig.from_dict(config)
            return len(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")


class OpenAIAVSystemFile(SyntheticFile):
    """Directly read/write the system prompt."""

    def __init__(self, agent: 'OpenAIAVAgent'):
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


class OpenAIAVStatusFile(SyntheticFile):
    """Real-time status including audio levels."""

    def __init__(self, agent: 'OpenAIAVAgent'):
        super().__init__("status")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        status = {
            "state": self.agent.state.value,
            "mic_level": self.agent._mic_level,
            "output_level": self.agent._output_level,
            "connected": self.agent.state in (OpenAIAVState.CONNECTED, OpenAIAVState.STREAMING),
            "streaming": self.agent.state == OpenAIAVState.STREAMING,
        }
        data = json.dumps(status).encode()
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Status file is read-only")


class OpenAIAVCodeFile(SyntheticFile):
    """
    Code output from function tool calls.

    STATE-AWARE BLOCKING (enables `while true; do cat $openai_av/code; done`):

    1. WAITING: read() blocks until set_code() fires
    2. READY: read() returns content
    3. CONSUMED: read() returns b"" (EOF — cat exits)
    4. Next read at offset 0: rearms, blocks again at step 1
    """

    def __init__(self, agent: 'OpenAIAVAgent'):
        super().__init__("CODE")
        self.agent = agent
        self._code = ""
        self._code_history: List[Dict[str, Any]] = []
        self._content_ready = asyncio.Event()
        self._content_consumed = False
        self._lock = asyncio.Lock()

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """State-aware blocking read."""
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
        """Allow manual writes to the code file."""
        self._code = data.decode('utf-8')
        self._content_ready.set()
        return len(data)

    async def set_code(self, code: str, function_name: str = "", call_id: str = ""):
        """Set code content from a function call result — unblocks readers."""
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
        """Get all code outputs from function calls."""
        return list(self._code_history)


class OpenAIAVAudioOutFile(SyntheticFile):
    """
    Raw PCM audio output from the AI.

    Format: 16-bit signed PCM, mono, 24000 Hz

    Usage:
        cat /mnt/llm/agents/openai_av/AUDIO | aplay -f S16_LE -r 24000 -c 1
        cat /mnt/llm/agents/openai_av/AUDIO | ffplay -f s16le -ar 24000 -ac 1 -nodisp -
    """

    def __init__(self, agent: 'OpenAIAVAgent'):
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


class OpenAIAVAudioInFile(SyntheticFile):
    """
    Write raw PCM audio to send to the AI.

    Format: 16-bit signed PCM, mono, 24000 Hz

    Usage:
        arecord -f S16_LE -r 24000 -c 1 > /mnt/llm/agents/openai_av/mic
    """

    def __init__(self, agent: 'OpenAIAVAgent'):
        super().__init__("mic")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        return b""

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        if self.agent._websocket and self.agent.state == OpenAIAVState.STREAMING:
            audio_base64 = base64.b64encode(data).decode('utf-8')
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": audio_base64,
            }
            await self.agent._websocket.send(json.dumps(audio_event))
        return len(data)


# ─── Main Agent ─────────────────────────────────────────────────────


class OpenAIAVAgent(SyntheticDir):
    """
    AudioVisual Agent using OpenAI Realtime API (WebSocket).

    Provides real-time audio interaction through the filesystem interface
    with function calling support. Code from function tools is delivered
    to the `CODE` file.

    Uses the OpenAI Realtime GA protocol:
      - WebSocket URL: wss://api.openai.com/v1/realtime?model=<model>
      - Auth: Authorization: Bearer <key>
      - Audio: 24kHz PCM16 mono, both directions
      - Session config via session.update with nested audio.input/output
      - Supports server VAD, function calling, image input
    """

    def __init__(
        self,
        name: str = "openai_av",
        route_manager: 'RouteManager' = None,
        function_registry: Dict[str, Callable] = None,
    ):
        super().__init__(name)
        self.agent_name = name
        self.route_manager = route_manager
        self.function_registry = function_registry or {}

        # State
        self.state = OpenAIAVState.DISCONNECTED
        self.history: List[Message] = []
        self.last_error: Optional[str] = None

        # Audio levels
        self._mic_level = 0.0
        self._output_level = 0.0
        self._mute_local = False
        self._output_device = None
        self._input_device = None

        # Configuration
        self.config = OpenAIAVConfig()

        # Session management
        self._websocket = None
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._audio_in_queue: Optional[asyncio.Queue] = None
        self._pending_message: Optional[str] = None

        # Barge-in: client-side interrupt for immediate audio cutoff
        self._interrupted = asyncio.Event()
        self._barge_in_threshold = 0.5   # Higher default — avoid speaker bleed triggers
        self._model_speaking = False
        self._barge_in_cooldown = 0.0
        self._last_audio_enqueued = 0.0
        self._model_done_at = 0.0        # When model stopped speaking
        self._mic_suppression_ms = 300   # ms to suppress mic after model stops

        # Function call accumulation (streaming args)
        self._pending_function_calls: Dict[str, Dict] = {}

        # PyAudio
        self._pya = None
        self._audio_stream = None
        self._output_stream = None

        # Child files
        self.output = StreamFile("OUTPUT")
        self.errors = QueueFile("errors")
        self.audio_out_file = OpenAIAVAudioOutFile(self)
        self.code_file = OpenAIAVCodeFile(self)
        self.context_file = OpenAIAVContextFile(self)

        self.add(CtlFile("ctl", OpenAIAVCtlHandler(self)))
        self.add(OpenAIAVInputFile(self))
        self.add(self.context_file)
        self.add(self.output)
        self.add(OpenAIAVHistoryFile(self))
        self.add(OpenAIAVConfigFile(self))
        self.add(OpenAIAVSystemFile(self))
        self.add(OpenAIAVStatusFile(self))
        self.add(self.code_file)
        self.add(self.audio_out_file)
        self.add(OpenAIAVAudioInFile(self))
        self.add(self.errors)

    def _check_dependencies(self):
        """Check if required dependencies are available"""
        if not WEBSOCKET_AVAILABLE:
            raise RuntimeError("websockets package not installed")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        if not AUDIO_AVAILABLE:
            raise RuntimeError("pyaudio not installed")

    async def start(self):
        """Start the live session"""
        if self.state in (OpenAIAVState.CONNECTED, OpenAIAVState.STREAMING, OpenAIAVState.CONNECTING):
            await self.errors.post(b"Already running\n")
            return

        try:
            self._check_dependencies()
        except RuntimeError as e:
            self.state = OpenAIAVState.ERROR
            self.last_error = str(e)
            await self.errors.post(f"{e}\n".encode())
            return

        self.state = OpenAIAVState.CONNECTING
        self._running = True

        asyncio.create_task(self._run_session())

    async def stop(self):
        """Stop the live session"""
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
        self.state = OpenAIAVState.DISCONNECTED

    async def _cleanup_audio(self):
        """Clean up audio resources"""
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
        """Main session loop — connects via WebSocket to OpenAI Realtime API"""
        try:
            self._pya = pyaudio.PyAudio()

            api_key = os.getenv("OPENAI_API_KEY")
            model = self.config.model

            ws_url = f"{OPENAI_REALTIME_URL}?model={model}"

            await self.errors.post(
                f"Connecting to OpenAI Realtime API ({model})...\n".encode()
            )

            self._websocket = await websocket_connect(
                uri=ws_url,
                additional_headers={
                    "Authorization": f"Bearer {api_key}",
                },
            )

            # Wait for session.created
            raw = await asyncio.wait_for(self._websocket.recv(), timeout=10.0)
            resp = json.loads(raw)
            if resp.get("type") == "session.created":
                session_id = resp.get("session", {}).get("id", "unknown")
                await self.errors.post(
                    f"Session created: {session_id}\n".encode()
                )
            else:
                await self.errors.post(
                    f"Unexpected first event: {resp.get('type', 'unknown')}\n".encode()
                )

            # Configure session
            await self._configure_session()

            self.state = OpenAIAVState.CONNECTED
            self._audio_in_queue = asyncio.Queue()

            await asyncio.sleep(0.2)

            async with asyncio.TaskGroup() as tg:
                self._tasks = [
                    tg.create_task(self._listen_audio()),
                    tg.create_task(self._receive_messages()),
                    tg.create_task(self._play_audio()),
                    tg.create_task(self._handle_text_messages()),
                ]

                # Add video task if enabled
                if self.config.video_mode == "camera" and CAMERA_AVAILABLE:
                    self._tasks.append(tg.create_task(self._get_camera_frames()))
                elif self.config.video_mode == "screen" and SCREEN_AVAILABLE:
                    self._tasks.append(tg.create_task(self._get_screen_frames()))

                self.state = OpenAIAVState.STREAMING

                while self._running:
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.state = OpenAIAVState.ERROR
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
            if self.state != OpenAIAVState.ERROR:
                self.state = OpenAIAVState.DISCONNECTED

    async def _configure_session(self):
        """Configure OpenAI Realtime session via session.update (GA format)"""

        # Build the GA session.update payload
        audio_input_config = {
            "format": {
                "type": "audio/pcm",
                "rate": SEND_SAMPLE_RATE,
            },
            "transcription": {
                "model": "gpt-4o-mini-transcribe",
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": self.config.turn_detection_threshold,
                "prefix_padding_ms": self.config.prefix_padding_ms,
                "silence_duration_ms": self.config.silence_duration_ms,
                "create_response": True,
                "interrupt_response": True,
            },
        }

        # Noise reduction (optional)
        if self.config.noise_reduction:
            audio_input_config["noise_reduction"] = {
                "type": self.config.noise_reduction,
            }

        audio_output_config = {
            "format": {
                "type": "audio/pcm",
                "rate": RECEIVE_SAMPLE_RATE,
            },
            "voice": self.config.voice,
            "speed": self.config.speed,
        }

        session_config = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": self.config.system or "You are a helpful AI assistant.",
                "audio": {
                    "input": audio_input_config,
                    "output": audio_output_config,
                },
                "tool_choice": self.config.tool_choice,
            },
        }

        # Add function tools (OpenAI Realtime format)
        if self.config.functions:
            tools = []
            for func_def in self.config.functions:
                tool = {
                    "type": "function",
                    "name": func_def["name"],
                    "description": func_def.get("description", ""),
                    "parameters": func_def.get("parameters", {}),
                }
                tools.append(tool)
            session_config["session"]["tools"] = tools
            await self.errors.post(f"Configured {len(tools)} tools\n".encode())

        await self._websocket.send(json.dumps(session_config))

        # Wait for session.updated confirmation
        try:
            response = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
            data = json.loads(response)
            event_type = data.get("type", "")
            if event_type == "session.updated":
                await self.errors.post(b"Session configured (session.updated)\n")
            elif event_type == "error":
                error_msg = json.dumps(data.get("error", {}))
                await self.errors.post(f"Session config error: {error_msg}\n".encode())
            else:
                await self.errors.post(f"Session config response: {event_type}\n".encode())
        except asyncio.TimeoutError:
            await self.errors.post(b"Session config: no confirmation received\n")

    # ─── Audio I/O ──────────────────────────────────────────────────

    async def _listen_audio(self):
        """Listen to microphone and send base64-encoded PCM to OpenAI.

        We send real mic data at all times so OpenAI's server-side VAD
        (with interrupt_response=True) can detect user speech and cancel
        the model's response.  Client-side barge-in provides immediate
        local audio cutoff as a backup.

        If speaker-to-mic feedback causes self-interruption loops, use
        headphones or increase the barge-in threshold via `ctl bargein`.
        """
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
                await self.errors.post(b"24kHz not supported, trying default rate\n")
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
            await self.errors.post(f"Mic opened at {sample_rate}Hz\n".encode())

            resample_state = None

            while self._running:
                data = await asyncio.to_thread(
                    stream.read, CHUNK_SIZE, exception_on_overflow=False
                )

                # Mic level (always compute for status display)
                if len(data) > 0:
                    try:
                        import audioop
                        rms = audioop.rms(data, 2)
                        self._mic_level = min(1.0, (rms / 32768.0) * 5.0)
                    except Exception:
                        pass

                # ── Barge-in handling ──
                # We send real mic data at all times so OpenAI's server-side
                # VAD (with interrupt_response=True) can detect user speech
                # and cancel the model's response.  Client-side barge-in
                # provides immediate local audio cutoff as a backup.
                if (
                    self._model_speaking
                    and self._mic_level > self._barge_in_threshold
                    and time.monotonic() > self._barge_in_cooldown
                ):
                    self._interrupted.set()
                    self._model_speaking = False
                    self._barge_in_cooldown = time.monotonic() + 0.5
                    await self.errors.post(b"Barge-in: client mic activity\n")

                send_data = data

                # Resample to 24kHz if needed
                if sample_rate != SEND_SAMPLE_RATE and len(send_data) > 0:
                    try:
                        import audioop
                        send_data, resample_state = audioop.ratecv(
                            send_data, 2, CHANNELS,
                            sample_rate, SEND_SAMPLE_RATE,
                            resample_state,
                        )
                    except Exception as e:
                        await self.errors.post(f"Resample error: {e}\n".encode())

                # Send as base64 over WebSocket (input_audio_buffer.append)
                audio_base64 = base64.b64encode(send_data).decode('utf-8')
                audio_event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_base64,
                }
                try:
                    await self._websocket.send(json.dumps(audio_event))
                except Exception as e:
                    if self._running:
                        await self.errors.post(f"Send audio error: {e}\n".encode())
                    break

                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                await self.errors.post(f"Audio input error: {e}\n".encode())

    async def _receive_messages(self):
        """Receive all messages from OpenAI WebSocket — audio, transcripts, tool calls"""
        try:
            async for message in self._websocket:
                if not self._running:
                    break

                data = json.loads(message)
                event_type = data.get("type", "")

                # ─── Function call handling (streaming format) ───

                # 1. Streaming function call arguments
                if event_type == "response.function_call_arguments.delta":
                    call_id = data.get("call_id")
                    delta = data.get("delta", "")

                    if call_id not in self._pending_function_calls:
                        self._pending_function_calls[call_id] = {
                            "name": data.get("name"),
                            "arguments": "",
                        }
                    self._pending_function_calls[call_id]["arguments"] += delta

                # 2. Function call arguments complete → execute
                elif event_type == "response.function_call_arguments.done":
                    call_id = data.get("call_id")
                    func_name = data.get("name")
                    args_str = data.get("arguments", "")

                    try:
                        args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        args = {}

                    await self.errors.post(
                        f"Function call: {func_name}({json.dumps(args)[:200]})\n".encode()
                    )

                    await self._handle_tool_call(call_id, func_name, args)

                    self._pending_function_calls.pop(call_id, None)

                # 3. output_item.done with function_call (backup path)
                elif event_type == "response.output_item.done":
                    item = data.get("item", {})
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id")
                        func_name = item.get("name")
                        args_str = item.get("arguments", "")
                        try:
                            args = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            args = {}
                        if call_id in self._pending_function_calls:
                            await self._handle_tool_call(call_id, func_name, args)
                            self._pending_function_calls.pop(call_id, None)

                # ─── Audio output ───
                # NOTE: OpenAI's GA docs say the event is response.output_audio.delta
                # but the server actually sends response.audio.delta (confirmed by
                # multiple developers). We handle both names for robustness.

                elif event_type in ("response.audio.delta", "response.output_audio.delta"):
                    audio_base64 = data.get("delta", "")
                    if audio_base64:
                        audio_data = base64.b64decode(audio_base64)
                        if self._audio_in_queue and not self._interrupted.is_set():
                            self._audio_in_queue.put_nowait(audio_data)
                            self._model_speaking = True
                            self._last_audio_enqueued = time.monotonic()

                # ─── Transcripts ───

                elif event_type in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
                    delta = data.get("delta", "")
                    if delta:
                        self.history.append(Message(
                            role="assistant",
                            content=delta,
                            source="audio",
                        ))
                        await self.output.append(delta.encode())
                        if self.route_manager:
                            await self.route_manager.broadcast(delta.encode())

                # Output text delta (for text-modality responses)
                elif event_type in ("response.text.delta", "response.output_text.delta"):
                    delta = data.get("delta", "")
                    if delta:
                        self.history.append(Message(
                            role="assistant",
                            content=delta,
                            source="text",
                        ))
                        await self.output.append(delta.encode())
                        if self.route_manager:
                            await self.route_manager.broadcast(delta.encode())

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    transcript = data.get("transcript", "")
                    if transcript:
                        self.history.append(Message(
                            role="user",
                            content=transcript,
                            source="audio",
                        ))

                # ─── Errors ───

                elif event_type == "error":
                    error_data = data.get("error", {})
                    error_msg = json.dumps(error_data) if isinstance(error_data, dict) else str(error_data)
                    await self.errors.post(f"OpenAI error: {error_msg}\n".encode())

                # ─── Barge-in: server detects user speech ───

                elif event_type == "input_audio_buffer.speech_started":
                    if self._model_speaking:
                        self._interrupted.set()
                        self._model_speaking = False
                        await self.errors.post(b"Barge-in: speech_started while model speaking\n")

                elif event_type == "response.cancelled":
                    self._interrupted.set()
                    self._model_speaking = False

                elif event_type == "response.done":
                    # Model finished — let playback queue drain naturally
                    pass

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                await self.errors.post(f"Receive error: {e}\n".encode())

    async def _handle_tool_call(self, call_id: str, func_name: str, args: dict):
        """Execute a function call and send the result back, write code to code file"""

        if not func_name:
            return

        # If the function produces code, write it to the code file
        code_content = args.get("code", "")
        code_written = False
        if code_content:
            await self.code_file.set_code(code_content, function_name=func_name, call_id=call_id)
            await self.errors.post(f"Code written to code file ({len(code_content)} chars)\n".encode())
            code_written = True

        # Execute function if registered
        result = None

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

        if result is None:
            if code_written:
                result = {"result": "ok", "executed": True, "chars": len(code_content)}
            else:
                result = {"result": "no handler registered", "function": func_name}

        # Log to history
        self.history.append(Message(
            role="assistant",
            content=f"[function:{func_name}] {json.dumps(result)[:500]}",
            source="function",
        ))

        # Send result back (Realtime API format — conversation.item.create)
        if self._websocket:
            function_output_event = {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }
            try:
                await self._websocket.send(json.dumps(function_output_event))
            except Exception as e:
                await self.errors.post(f"Error sending function result: {e}\n".encode())

            # NOTE: Same as Grok — we intentionally do NOT send response.create
            # after function output. The OpenAI Realtime API will auto-generate
            # a follow-up response if needed. Unconditionally sending
            # response.create can cause infinite loops when the model's response
            # includes both speech and a tool call in the same turn.

    async def _play_audio(self):
        """Play audio responses with barge-in support."""
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
                    # Handle interrupt: drain queue atomically
                    if self._interrupted.is_set():
                        drained = 0
                        while not self._audio_in_queue.empty():
                            try:
                                self._audio_in_queue.get_nowait()
                                drained += 1
                            except asyncio.QueueEmpty:
                                break
                        if drained > 0:
                            await self.errors.post(
                                f"Barge-in: flushed {drained} audio chunks\n".encode()
                            )
                        output_resample_state = None
                        self._output_level = 0.0
                        self._model_speaking = False
                        self._model_done_at = time.monotonic()
                        self._interrupted.clear()
                        continue

                    # Get next audio chunk (short timeout)
                    try:
                        bytestream = await asyncio.wait_for(
                            self._audio_in_queue.get(), timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        if (
                            self._model_speaking
                            and self._audio_in_queue.empty()
                            and (time.monotonic() - self._last_audio_enqueued) > 0.3
                        ):
                            self._model_speaking = False
                            self._model_done_at = time.monotonic()
                            self._output_level = 0.0
                        continue

                    # Re-check interrupt after dequeue
                    if self._interrupted.is_set():
                        continue

                    # Output level
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
                            await self.errors.post(f"Played {chunks_played} audio chunks\n".encode())

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
                if self._pending_message:
                    message = self._pending_message
                    self._pending_message = None

                    if message and self._websocket:
                        self.history.append(Message(
                            role="user",
                            content=message,
                            source="text",
                        ))

                        # Send as a conversation item (Realtime API text input)
                        text_event = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": message,
                                    }
                                ],
                            },
                        }
                        await self._websocket.send(json.dumps(text_event))
                        # Trigger response
                        await self._websocket.send(json.dumps({"type": "response.create"}))

                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    await self.errors.post(f"Message error: {e}\n".encode())

    # ─── Video capture ──────────────────────────────────────────────
    # OpenAI Realtime GA supports image input via conversation items.
    # We capture frames and send them as image content parts.

    async def _get_camera_frames(self):
        """Capture frames from camera and send as image conversation items"""
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

                # Send image as a conversation item
                image_event = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image": {
                                    "data": image_b64,
                                    "format": "jpeg",
                                },
                            }
                        ],
                    },
                }
                try:
                    await self._websocket.send(json.dumps(image_event))
                except Exception as e:
                    if self._running:
                        await self.errors.post(f"Send video error: {e}\n".encode())
                    break

                await asyncio.sleep(2.0)  # Every 2 seconds

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                await self.errors.post(f"Camera error: {e}\n".encode())
        finally:
            if cap:
                cap.release()

    async def _get_screen_frames(self):
        """Capture screen frames and send as image conversation items"""
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

                image_event = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image": {
                                    "data": image_b64,
                                    "format": "jpeg",
                                },
                            }
                        ],
                    },
                }
                try:
                    await self._websocket.send(json.dumps(image_event))
                except Exception as e:
                    if self._running:
                        await self.errors.post(f"Send screen error: {e}\n".encode())
                    break

                await asyncio.sleep(2.0)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                await self.errors.post(f"Screen capture error: {e}\n".encode())

    # ─── Public API ─────────────────────────────────────────────────

    async def send_text(self, text: str):
        """Send text message to the session"""
        if self.state not in (OpenAIAVState.CONNECTED, OpenAIAVState.STREAMING):
            await self.errors.post(b"Not connected\n")
            return
        self._pending_message = text

    async def send_context(self, text: str):
        """
        Inject context into the session without triggering a response.

        Two-pronged approach:
          1. conversation.item.create — adds a user message to the conversation
             so the model sees it in its context window
          2. session.update — appends to instructions as fallback

        No response.create is sent, so the model stays silent.
        """
        if self.state not in (OpenAIAVState.CONNECTED, OpenAIAVState.STREAMING):
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

        # Prong 1: inject as conversation item (no response.create)
        context_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"[CONTEXT] {text}",
                    }
                ],
            },
        }
        try:
            await self._websocket.send(json.dumps(context_event))
        except Exception as e:
            await self.errors.post(f"Context item error: {e}\n".encode())

        # Prong 2: update instructions with accumulated context
        await self._update_instructions()

        await self.errors.post(f"Context injected ({len(text)} chars)\n".encode())

    async def _update_instructions(self):
        """
        Re-send session.update with current system prompt + accumulated context.

        Partial session.update is supported by OpenAI's Realtime API —
        only the fields present are updated.
        """
        if not self._websocket:
            return
        if self.state not in (OpenAIAVState.CONNECTED, OpenAIAVState.STREAMING):
            return

        base = self.config.system or "You are a helpful AI assistant."
        context_blocks = self.context_file._context_blocks

        if context_blocks:
            instructions = (
                base
                + OpenAIAVContextFile.CONTEXT_SEPARATOR
                + "\n".join(context_blocks)
            )
        else:
            instructions = base

        session_update = {
            "type": "session.update",
            "session": {
                "instructions": instructions,
            },
        }
        try:
            await self._websocket.send(json.dumps(session_update))
        except Exception as e:
            await self.errors.post(f"Context update error: {e}\n".encode())

    async def clear(self):
        """Clear history"""
        self.history.clear()
        await self.output.reset()

    async def cancel(self):
        """Cancel/stop the session"""
        await self.stop()