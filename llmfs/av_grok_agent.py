"""
AudioVisual Grok Agent for LLMFS

This agent uses Grok's Realtime API (WebSocket) for real-time audio streaming
with function calling support. Code from function tools is delivered to
the agent's `code` file.

Filesystem structure:
    agents/grok_av/
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

# WebSocket for Grok
try:
    from websockets.asyncio.client import connect as websocket_connect
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False

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


# Audio configuration
if AUDIO_AVAILABLE:
    FORMAT = pyaudio.paInt16
else:
    FORMAT = 8  # paInt16 value as fallback
CHANNELS = 1
SEND_SAMPLE_RATE = 24000  # Grok expects 24kHz
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

# Grok configuration
GROK_WEBSOCKET_URL = "wss://api.x.ai/v1/realtime"

# Available voices for Grok
GROK_VOICES = [
    "Ara", "Cora", "Sage", "Ember", "Ivy",
    "Kai", "Nova", "Sol", "Tara", "Vale"
]


class GrokAVState(Enum):
    """Grok AV Agent state"""
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
class GrokAVConfig:
    """Configuration for Grok AV agent"""
    voice: str = "Ara"
    video_mode: str = "none"
    system: Optional[str] = None
    functions: List[Dict] = field(default_factory=list)
    temperature: float = 0.8
    turn_detection_threshold: float = 0.5
    silence_duration_ms: int = 700
    prefix_padding_ms: int = 300
    tool_choice: str = "auto"

    def to_dict(self) -> dict:
        return {
            "voice": self.voice,
            "video_mode": self.video_mode,
            "system": self.system,
            "functions": self.functions,
            "temperature": self.temperature,
            "turn_detection_threshold": self.turn_detection_threshold,
            "silence_duration_ms": self.silence_duration_ms,
            "prefix_padding_ms": self.prefix_padding_ms,
            "tool_choice": self.tool_choice,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'GrokAVConfig':
        return cls(
            voice=d.get("voice", "Ara"),
            video_mode=d.get("video_mode", "none"),
            system=d.get("system"),
            functions=d.get("functions", []),
            temperature=d.get("temperature", 0.8),
            turn_detection_threshold=d.get("turn_detection_threshold", 0.5),
            silence_duration_ms=d.get("silence_duration_ms", 700),
            prefix_padding_ms=d.get("prefix_padding_ms", 300),
            tool_choice=d.get("tool_choice", "auto"),
        )


# ─── Filesystem Files ──────────────────────────────────────────────


class GrokAVCtlHandler(CtlHandler):
    """Control handler for Grok AV agent"""

    def __init__(self, agent: 'GrokAVAgent'):
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

        elif cmd == "voice":
            if arg:
                if arg not in GROK_VOICES:
                    raise ValueError(f"Unknown voice: {arg}. Available: {', '.join(GROK_VOICES)}")
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
            return "\n".join(GROK_VOICES)

        elif cmd == "temperature":
            if arg:
                self.agent.config.temperature = float(arg)
                return f"Temperature set to {arg}"
            return str(self.agent.config.temperature)

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

        else:
            raise ValueError(
                f"Unknown command: {cmd}. "
                "Available: start, stop, restart, voice, mode, system, clear, "
                "mute, unmute, voices, devices, output, input, temperature, "
                "bargein, interrupt"
            )

    async def get_status(self) -> bytes:
        a = self.agent
        lines = [
            f"state {a.state.value}",
            f"voice {a.config.voice}",
            f"video {a.config.video_mode}",
            f"messages {len(a.history)}",
            f"temperature {a.config.temperature}",
        ]

        if a.config.system:
            sys_preview = a.config.system[:50] + "..." if len(a.config.system) > 50 else a.config.system
            lines.append(f"system {sys_preview}")

        if a.state == GrokAVState.ERROR and a.last_error:
            lines.append(f"error {a.last_error}")

        if a.state == GrokAVState.STREAMING:
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


class GrokAVInputFile(SyntheticFile):
    """Write text messages to send during live session."""

    def __init__(self, agent: 'GrokAVAgent'):
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


class GrokAVContextFile(SyntheticFile):
    """
    Write context/information to the agent without triggering a response.

    Writing here injects context that the model will see on its next turn,
    without generating an immediate response.

    The implementation appends context blocks to the session instructions
    via session.update, which is the reliable way to get Grok's Realtime
    API to incorporate new information mid-session. The base system prompt
    is preserved and context blocks are appended below it.

    Usage:
        echo "The user's name is Alice" > /mnt/llm/grok_av/context
        echo '{"temperature": 22, "city": "NYC"}' > /mnt/llm/grok_av/context

    Reading returns all accumulated context blocks.
    Writing with just "clear" resets the context.
    """

    CONTEXT_SEPARATOR = "\n\n---\n[CONTEXT]\n"

    def __init__(self, agent: 'GrokAVAgent'):
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
        await self.agent._update_instructions()
        return len(data)

    async def getattr(self):
        data = ("\n".join(self._context_blocks) + "\n").encode() if self._context_blocks else b""
        return {"st_size": len(data)}


class GrokSystemFile(SyntheticFile):
    """Dedicated system prompt file - handles full content writes"""
    
    def __init__(self, agent: 'GrokAVAgent'):
        super().__init__()
        self.agent = agent

    async def read(self, fid, off, size):
        content = (self.agent.config.system or "") + "\n"
        data = content.encode()
        return data[off:off + size]
        
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        current_data = bytearray((self.agent.config.system or "").encode('utf-8'))
        
        if offset > len(current_data):
            current_data.extend(b'\0' * (offset - len(current_data)))
        
        current_data[offset:offset + len(data)] = data
        
        self.agent.config.system = current_data.decode('utf-8', errors='replace')
        return len(data)
    
    async def getattr(self):
        content = (self.agent.config.system or "") + "\n"
        return {"st_size": len(content.encode())}


class GrokAVHistoryFile(SyntheticFile):
    """Read agent history as JSON."""

    def __init__(self, agent: 'GrokAVAgent'):
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


class GrokAVConfigFile(SyntheticFile):
    """Read/write agent configuration as JSON."""

    def __init__(self, agent: 'GrokAVAgent'):
        super().__init__("config")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        data = json.dumps(self.agent.config.to_dict(), indent=2).encode()
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        try:
            config = json.loads(data.decode())
            self.agent.config = GrokAVConfig.from_dict(config)
            return len(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")


class GrokAVSystemFile(SyntheticFile):
    """Directly read/write the system prompt."""

    def __init__(self, agent: 'GrokAVAgent'):
        super().__init__("system")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        val = (self.agent.config.system or "").encode()
        return val[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        # Convert existing string to bytearray for manipulation
        current_data = bytearray((self.agent.config.system or "").encode('utf-8'))
        
        # Ensure the bytearray is large enough for the incoming offset
        if offset > len(current_data):
            current_data.extend(b'\0' * (offset - len(current_data)))
            
        # Write the chunk at the specific offset
        current_data[offset:offset + len(data)] = data
        
        # Update the agent config
        self.agent.config.system = current_data.decode('utf-8', errors='replace')
        
        return len(data)


class GrokAVStatusFile(SyntheticFile):
    """Real-time status including audio levels."""

    def __init__(self, agent: 'GrokAVAgent'):
        super().__init__("status")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        status = {
            "state": self.agent.state.value,
            "mic_level": self.agent._mic_level,
            "output_level": self.agent._output_level,
            "connected": self.agent.state in (GrokAVState.CONNECTED, GrokAVState.STREAMING),
            "streaming": self.agent.state == GrokAVState.STREAMING,
        }
        data = json.dumps(status).encode()
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Status file is read-only")


class GrokAVCodeFile(SyntheticFile):
    """
    Code output from function tool calls.

    STATE-AWARE BLOCKING (enables `while true; do cat $grok_av/code; done`):

    1. WAITING: read() blocks until set_code() fires
    2. READY: read() returns content
    3. CONSUMED: read() returns b"" (EOF — cat exits)
    4. Next read at offset 0: rearms, blocks again at step 1
    """

    def __init__(self, agent: 'GrokAVAgent'):
        super().__init__("CODE")
        self.agent = agent
        self._code = ""
        self._code_history: List[Dict[str, Any]] = []
        self._content_ready = asyncio.Event()
        self._content_consumed = False
        self._lock = asyncio.Lock()

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """State-aware blocking read."""
        # If consumed and back at offset 0 → rearm for next function call
        if offset == 0 and self._content_consumed:
            async with self._lock:
                if self._content_consumed:
                    self._content_consumed = False
                    self._content_ready.clear()

        # Block until code arrives from a function call
        await self._content_ready.wait()

        async with self._lock:
            data = (self._code + "\n").encode() if self._code else b""
            chunk = data[offset:offset + count]

            if offset + len(chunk) >= len(data):
                self._content_consumed = True

            return chunk

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        """Allow manual writes to the code file (e.g., injecting code)."""
        self._code = data.decode('utf-8')
        # Manual write also unblocks readers
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
        # Unblock any waiting readers
        self._content_ready.set()

    def get_history(self) -> List[Dict[str, Any]]:
        """Get all code outputs from function calls."""
        return list(self._code_history)

class GrokAVAudioOutFile(SyntheticFile):
    """
    Raw PCM audio output from the AI.

    Format: 16-bit signed PCM, mono, 24000 Hz

    Usage:
        cat /mnt/llm/agents/grok_av/audio | aplay -f S16_LE -r 24000 -c 1
        cat /mnt/llm/agents/grok_av/audio | ffplay -f s16le -ar 24000 -ac 1 -nodisp -
    """

    def __init__(self, agent: 'GrokAVAgent'):
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


class GrokAVAudioInFile(SyntheticFile):
    """
    Write raw PCM audio to send to the AI.

    Format: 16-bit signed PCM, mono, 24000 Hz

    Usage:
        arecord -f S16_LE -r 24000 -c 1 > /mnt/llm/agents/grok_av/mic
    """

    def __init__(self, agent: 'GrokAVAgent'):
        super().__init__("mic")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        return b""

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        if self.agent._websocket and self.agent.state == GrokAVState.STREAMING:
            audio_base64 = base64.b64encode(data).decode('utf-8')
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": audio_base64,
            }
            await self.agent._websocket.send(json.dumps(audio_event))
        return len(data)


# ─── Main Agent ─────────────────────────────────────────────────────


class GrokAVAgent(SyntheticDir):
    """
    AudioVisual Agent using Grok Realtime API (WebSocket).

    Provides real-time audio interaction through the filesystem interface
    with function calling support. Code from function tools is delivered
    to the `code` file.
    """

    def __init__(
        self,
        name: str = "grok_av",
        route_manager: 'RouteManager' = None,
        function_registry: Dict[str, Callable] = None,
    ):
        super().__init__(name)
        self.agent_name = name
        self.route_manager = route_manager
        self.function_registry = function_registry or {}

        # State
        self.state = GrokAVState.DISCONNECTED
        self.history: List[Message] = []
        self.last_error: Optional[str] = None

        # Audio levels
        self._mic_level = 0.0
        self._output_level = 0.0
        self._mute_local = False
        self._output_device = None
        self._input_device = None

        # Configuration
        self.config = GrokAVConfig()

        # Session management
        self._websocket = None
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._audio_in_queue: Optional[asyncio.Queue] = None
        self._pending_message: Optional[str] = None

        # Barge-in: client-side interrupt for immediate audio cutoff
        self._interrupted = asyncio.Event()
        self._barge_in_threshold = 0.15  # mic RMS level that triggers flush
        self._model_speaking = False      # True while audio queue has content
        self._barge_in_cooldown = 0.0     # timestamp: ignore barge-in until this time
        self._last_audio_enqueued = 0.0   # timestamp of last audio chunk received

        # Function call accumulation (streaming args)
        self._pending_function_calls: Dict[str, Dict] = {}

        # PyAudio
        self._pya = None
        self._audio_stream = None
        self._output_stream = None

        # Child files
        self.output = StreamFile("OUTPUT")
        self.errors = QueueFile("errors")
        self.audio_out_file = GrokAVAudioOutFile(self)
        self.code_file = GrokAVCodeFile(self)
        self.context_file = GrokAVContextFile(self)

        self.add(CtlFile("ctl", GrokAVCtlHandler(self)))
        self.add(GrokAVInputFile(self))
        self.add(self.context_file)
        self.add(self.output)
        self.add(GrokAVHistoryFile(self))
        self.add(GrokAVConfigFile(self))
        self.add(GrokAVSystemFile(self))
        self.add(GrokAVStatusFile(self))
        self.add(self.code_file)
        self.add(self.audio_out_file)
        self.add(GrokAVAudioInFile(self))
        self.add(self.errors)

    def _check_dependencies(self):
        """Check if required dependencies are available"""
        if not GROK_AVAILABLE:
            raise RuntimeError("websockets package not installed")
        if not os.getenv("XAI_API_KEY"):
            raise RuntimeError("XAI_API_KEY not set")
        if not AUDIO_AVAILABLE:
            raise RuntimeError("pyaudio not installed")

    async def start(self):
        """Start the live session"""
        if self.state in (GrokAVState.CONNECTED, GrokAVState.STREAMING, GrokAVState.CONNECTING):
            await self.errors.post(b"Already running\n")
            return

        try:
            self._check_dependencies()
        except RuntimeError as e:
            self.state = GrokAVState.ERROR
            self.last_error = str(e)
            await self.errors.post(f"{e}\n".encode())
            return

        self.state = GrokAVState.CONNECTING
        self._running = True

        asyncio.create_task(self._run_session())

    async def stop(self):
        """Stop the live session"""
        self._running = False

        for task in self._tasks:
            if not task.done():
                task.cancel()
        self._tasks.clear()

        # Close websocket
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None

        await self._cleanup_audio()
        self.state = GrokAVState.DISCONNECTED

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
        """Main session loop — connects via WebSocket to Grok Realtime API"""
        try:
            self._pya = pyaudio.PyAudio()

            api_key = os.getenv("XAI_API_KEY")

            self._websocket = await websocket_connect(
                uri=GROK_WEBSOCKET_URL,
                additional_headers={"Authorization": f"Bearer {api_key}"},
            )

            await self.errors.post(b"Connected to Grok Realtime API\n")

            # Configure session
            await self._configure_session()

            self.state = GrokAVState.CONNECTED
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

                self.state = GrokAVState.STREAMING

                while self._running:
                    await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.state = GrokAVState.ERROR
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
            if self.state != GrokAVState.ERROR:
                self.state = GrokAVState.DISCONNECTED

    async def _configure_session(self):
        """Configure Grok session with tools via session.update"""
        session_config = {
            "type": "session.update",
            "session": {
                "voice": self.config.voice,
                "instructions": self.config.system or "You are a helpful AI assistant.",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {"model": "whisper-1"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self.config.turn_detection_threshold,
                    "prefix_padding_ms": self.config.prefix_padding_ms,
                    "silence_duration_ms": self.config.silence_duration_ms,
                },
                "temperature": self.config.temperature,
                "tool_choice": self.config.tool_choice,
            },
        }

        # Add function tools (Realtime API format)
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

        # Wait for confirmation
        try:
            response = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
            data = json.loads(response)
            await self.errors.post(f"Session configured: {data.get('type')}\n".encode())
        except asyncio.TimeoutError:
            await self.errors.post(b"Session config: no confirmation received\n")

    # ─── Audio I/O ──────────────────────────────────────────────────

    async def _listen_audio(self):
        """Listen to microphone and send base64-encoded PCM to Grok"""
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

                # Mic level
                if len(data) > 0:
                    try:
                        import audioop
                        rms = audioop.rms(data, 2)
                        self._mic_level = min(1.0, (rms / 32768.0) * 5.0)
                    except Exception:
                        pass

                # Client-side barge-in: if user is speaking while model
                # is playing audio, flush the playback queue immediately.
                # Use a cooldown to prevent spurious triggers from ambient noise
                # right after the model finishes.
                if (
                    self._model_speaking
                    and self._mic_level > self._barge_in_threshold
                    and time.monotonic() > self._barge_in_cooldown
                ):
                    self._interrupted.set()
                    self._model_speaking = False
                    # Small cooldown to prevent re-triggering on the same speech burst
                    self._barge_in_cooldown = time.monotonic() + 0.5
                    await self.errors.post(b"Barge-in: client mic activity\n")

                # Resample to 24kHz if needed
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

                # Send as base64 over WebSocket
                audio_base64 = base64.b64encode(data).decode('utf-8')
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
        """Receive all messages from Grok WebSocket — audio, transcripts, tool calls"""
        try:
            async for message in self._websocket:
                if not self._running:
                    break

                data = json.loads(message)
                event_type = data.get("type", "")

                # ─── Function call handling (Realtime API streaming format) ───

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
                        # Only handle if not already handled via arguments.done
                        if call_id in self._pending_function_calls:
                            await self._handle_tool_call(call_id, func_name, args)
                            self._pending_function_calls.pop(call_id, None)

                # ─── Audio output ───

                elif event_type == "response.output_audio.delta":
                    audio_base64 = data.get("delta", "")
                    if audio_base64:
                        audio_data = base64.b64decode(audio_base64)
                        if self._audio_in_queue and not self._interrupted.is_set():
                            self._audio_in_queue.put_nowait(audio_data)
                            self._model_speaking = True
                            self._last_audio_enqueued = time.monotonic()

                # ─── Transcripts ───

                elif event_type == "response.output_audio_transcript.delta":
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
                    await self.errors.post(f"Grok error: {error_msg}\n".encode())

                # ─── Barge-in: server detects user speech ───

                elif event_type == "input_audio_buffer.speech_started":
                    # Server detected the user speaking — cancel playback
                    if self._model_speaking:
                        self._interrupted.set()
                        self._model_speaking = False
                        await self.errors.post(b"Barge-in: speech_started while model speaking\n")

                elif event_type == "response.cancelled":
                    # Server cancelled the response (e.g. due to barge-in)
                    self._interrupted.set()
                    self._model_speaking = False

                elif event_type == "response.done":
                    # Model finished sending audio — do NOT interrupt/flush.
                    # The playback queue may still have chunks to play out.
                    # Just clear the speaking flag once queue drains naturally.
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

        # Execute function if registered in the local Python registry
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

        # If no local handler but code was written to the code file
        # (routed to scene/parse), report success so the model
        # does NOT retry the same call in a loop.
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

        # Send result back to Grok (Realtime API format)
        if self._websocket:
            # Send function_call_output so Grok knows the tool completed
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

            # NOTE: We intentionally do NOT send response.create here.
            #
            # The Grok Realtime API will auto-generate a follow-up
            # response after receiving the function_call_output if it
            # determines one is needed (e.g. the model's turn was only
            # a tool call with no speech).
            #
            # Unconditionally sending response.create was causing an
            # infinite loop: Grok's response already included both
            # speech AND a tool call in the same turn.  After we
            # returned the tool result + response.create, Grok would
            # produce another full response (speech + tool call),
            # which triggered another response.create, ad infinitum.

    async def _play_audio(self):
        """Play audio responses with barge-in support.

        When _interrupted is set (by server speech_started/response.cancelled
        or client-side mic activity), flush the playback queue immediately.
        """
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
                    # ── Handle interrupt: drain queue atomically ──
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
                        # Clear the flag AFTER draining so no new chunks
                        # sneak in between drain and clear
                        self._interrupted.clear()
                        continue

                    # ── Get next audio chunk (short timeout) ──
                    try:
                        bytestream = await asyncio.wait_for(
                            self._audio_in_queue.get(), timeout=0.1
                        )
                    except asyncio.TimeoutError:
                        # No audio for 100ms — if model was speaking and no
                        # new audio has arrived recently, it finished naturally
                        if (
                            self._model_speaking
                            and self._audio_in_queue.empty()
                            and (time.monotonic() - self._last_audio_enqueued) > 0.3
                        ):
                            self._model_speaking = False
                            self._output_level = 0.0
                        continue

                    # Re-check interrupt after dequeue (chunk may be stale)
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

    # ─── Video capture (same as Gemini AV agent) ────────────────────

    async def _get_camera_frames(self):
        """Capture frames from camera"""
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
                frame_data = {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_bytes).decode(),
                }
                await asyncio.sleep(1.0)
                # NOTE: Grok Realtime API may not support image input
                # This is kept for forward compatibility
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                await self.errors.post(f"Camera error: {e}\n".encode())
        finally:
            if cap:
                cap.release()

    async def _get_screen_frames(self):
        """Capture screen frames"""
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
                frame_data = {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_bytes).decode(),
                }
                await asyncio.sleep(1.0)
                # NOTE: Grok Realtime API may not support image input
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._running:
                await self.errors.post(f"Screen capture error: {e}\n".encode())

    # ─── Public API ─────────────────────────────────────────────────

    async def send_text(self, text: str):
        """Send text message to the session"""
        if self.state not in (GrokAVState.CONNECTED, GrokAVState.STREAMING):
            await self.errors.post(b"Not connected\n")
            return
        self._pending_message = text

    async def _update_instructions(self):
        """
        Re-send session.update with current system prompt + accumulated context.

        This is the reliable mechanism to inject context mid-session:
        Grok's Realtime API always respects the instructions field from
        the most recent session.update.
        """
        if not self._websocket:
            return
        if self.state not in (GrokAVState.CONNECTED, GrokAVState.STREAMING):
            return

        base = self.config.system or "You are a helpful AI assistant."
        context_blocks = self.context_file._context_blocks

        if context_blocks:
            instructions = (
                base
                + GrokAVContextFile.CONTEXT_SEPARATOR
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
            await self.errors.post(
                f"Instructions updated ({len(context_blocks)} context blocks, "
                f"{len(instructions)} chars)\n".encode()
            )
        except Exception as e:
            await self.errors.post(f"Context update error: {e}\n".encode())

    async def clear(self):
        """Clear history"""
        self.history.clear()
        await self.output.reset()

    async def cancel(self):
        """Cancel/stop the session"""
        await self.stop()