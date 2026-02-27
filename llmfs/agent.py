"""
Agent Management for LLMFS

An agent represents a named LLM session with its own configuration,
history, and streaming capabilities.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import re

from core.files import (
    SyntheticDir, SyntheticFile, StreamFile, QueueFile, 
    CtlFile, CtlHandler
)
from core.types import FidState
from .providers import LLMProvider, ProviderConfig
from .media import (
    ContentBlock, MediaInfo, parse_input_data,
    detect_media, estimate_media_tokens
)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a string.
    
    Uses a heuristic of ~4 characters per token (covers most LLM tokenizers).
    This is intentionally conservative — better to trim early than hit a 400 error.
    
    For more accuracy, you could integrate tiktoken or the provider's own
    tokenizer, but this is good enough for context window management.
    """
    return max(1, len(text) // 4)


def estimate_message_tokens(msg: 'Message') -> int:
    """
    Estimate token count for a message, including any media attachments.
    """
    tokens = estimate_tokens(msg.content) + 4  # +4 for role/framing
    for block in msg.content_blocks:
        if block.type == "media" and block.media is not None:
            tokens += estimate_media_tokens(block.media)
    return tokens


class AgentState(Enum):
    """Agent state"""
    IDLE = "idle"           # Waiting for input
    STREAMING = "streaming" # Generating response
    DONE = "done"           # Response complete
    ERROR = "error"         # Error occurred
    CANCELLED = "cancelled" # Generation cancelled


@dataclass
class Message:
    """A single message in agent history"""
    role: str        # "user" or "assistant"
    content: str     # Text content (always present for display/history)
    timestamp: float = field(default_factory=time.time)
    content_blocks: List[ContentBlock] = field(default_factory=list)
    
    @property
    def has_media(self) -> bool:
        """Whether this message contains media attachments"""
        return any(b.type == "media" for b in self.content_blocks)
    
    @property
    def is_multimodal(self) -> bool:
        """Whether this message has content blocks (vs plain text)"""
        return len(self.content_blocks) > 0


class SupplementaryOutputFile(SyntheticFile):
    """
    A supplementary output file that contains extracted content blocks.
    Created dynamically when a plumbing rule is added.
    
    STATE-AWARE BLOCKING (enables `while true; do cat $agent/bash; done`):
    
    1. WAITING: read() blocks until mark_ready() fires
    2. READY: read() returns content
    3. CONSUMED: read() returns b"" (EOF — cat exits)
    4. Next read at offset 0: rearms, blocks again at step 1
    
    WRITE SEMANTICS:
    Writing to a supplementary output file sets a context string that
    gets injected into the agent's system prompt as:
        CONTEXT FOR <NAME>:
        <written content>
    Each write replaces the previous context entirely.
    
    The 9P server dispatches each message as a concurrent asyncio task,
    so a blocked read() never prevents writes to other files.
    """
    
    def __init__(self, name: str):
        super().__init__(name)
        self.blocks: List[str] = []
        self._content_ready = asyncio.Event()
        self._content_consumed = False
        self._lock = asyncio.Lock()
        # Context injected into the agent's system prompt
        self.context: str = ""
        self._write_buf: Dict[int, bytearray] = {}  # fid -> write buffer
    
    def add_block(self, content: str):
        """Add a new content block (called during extraction)"""
        self.blocks.append(content)
    
    def mark_ready(self):
        """Mark content as ready for reading (called when generation completes)"""
        self._content_ready.set()
    
    def clear(self):
        """Clear all blocks and reset state for next generation."""
        self.blocks.clear()
        self._content_ready.clear()
        self._content_consumed = False
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """
        State-aware blocking read.
        """
        # If consumed and back at offset 0 → rearm for next generation
        if offset == 0 and self._content_consumed:
            async with self._lock:
                if self._content_consumed:
                    self._content_consumed = False
                    self._content_ready.clear()
        
        # Block until content is ready
        await self._content_ready.wait()
        
        async with self._lock:
            content = "\n\n".join(self.blocks)
            if content:
                content += "\n"
            data = content.encode()
            
            chunk = data[offset:offset + count]
            
            if offset + len(chunk) >= len(data):
                self._content_consumed = True
            
            return chunk
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        """
        Write context for this supplementary output.
        
        The written content becomes a section in the agent's system prompt:
            CONTEXT FOR <NAME>:
            <content>
        
        Writes are buffered per-fid and applied on clunk, replacing
        any previous context entirely.
        """
        fid_key = id(fid)
        if fid_key not in self._write_buf:
            self._write_buf[fid_key] = bytearray()
        
        buf = self._write_buf[fid_key]
        
        # Offset 0 with existing data = new write sequence
        if offset == 0 and len(buf) > 0:
            buf.clear()
        
        if offset + len(data) > len(buf):
            buf.extend(b'\x00' * (offset + len(data) - len(buf)))
        buf[offset:offset + len(data)] = data
        
        return len(data)
    
    async def clunk(self, fid: FidState):
        """Apply buffered context write on fid close."""
        fid_key = id(fid)
        buf = self._write_buf.pop(fid_key, None)
        
        if buf is None:
            return
        
        text = bytes(buf).decode('utf-8', errors='replace').strip()
        # Replace context entirely (empty write = clear context)
        self.context = text


class AgentCtlHandler(CtlHandler):
    """Control handler for agent"""
    
    def __init__(self, agent: 'Agent'):
        self.agent = agent
    
    async def execute(self, command: str) -> Optional[str]:
        parts = command.split(' ', 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd == "provider":
            if arg:
                # Switch provider (and optionally model): provider <name> [model]
                sub_parts = arg.split(None, 1)
                provider_name = sub_parts[0]
                model = sub_parts[1] if len(sub_parts) > 1 else None
                from .providers import get_provider
                new_provider = get_provider(provider_name)
                self.agent.provider = new_provider
                if model:
                    self.agent.config.model = model
                else:
                    # Default to provider's first model
                    self.agent.config.model = new_provider.default_model
                return f"Provider set to {provider_name}, model {self.agent.config.model}"
            return self.agent.provider.name
        
        elif cmd == "model":
            if arg:
                self.agent.config.model = arg
                return f"Model set to {arg}"
            return self.agent.config.model
        
        elif cmd == "system":
            if arg:
                self.agent.config.system = arg
                return "System prompt set"
            return self.agent.config.system or "(none)"
        
        elif cmd == "temperature":
            if arg:
                self.agent.config.temperature = float(arg)
                return f"Temperature set to {arg}"
            return str(self.agent.config.temperature)
        
        elif cmd == "max_tokens":
            if arg:
                self.agent.config.max_tokens = int(arg)
                return f"Max tokens set to {arg}"
            return str(self.agent.config.max_tokens)
        
        elif cmd == "max_history":
            if arg:
                self.agent.config.max_history = int(arg)
                return f"Max history set to {arg} messages (0 = unlimited)"
            return str(self.agent.config.max_history)
        
        elif cmd == "max_context_tokens":
            if arg:
                self.agent.max_context_tokens = int(arg)
                return f"Max context tokens set to {arg} (0 = disabled)"
            return str(self.agent.max_context_tokens)
        
        elif cmd == "exec":
            try:
                rule_idx = int(arg)
                if 0 <= rule_idx < len(self.agent.plumbing_rules):
                    rule = self.agent.plumbing_rules[rule_idx]
                    await self.agent.execute_history_rule(rule)
                    return f"Executed rule {rule_idx} on history"
                else:
                    return f"Error: Rule index {rule_idx} out of range"
            except ValueError:
                return "Usage: exec <rule_index>"
        
        elif cmd == "clear":
            await self.agent.clear()
            return "Agent history cleared"
        
        elif cmd == "cancel":
            await self.agent.cancel()
            return "Generation cancelled"
        
        elif cmd == "retry":
            await self.agent.retry()
            return "Retrying last message"
        
        elif cmd == "clearout":
            # Clear specific supplementary output
            if not arg:
                raise ValueError("Usage: clearout <name>")
            if arg in self.agent.supplementary_outputs:
                self.agent.supplementary_outputs[arg].clear()
                return f"Cleared output '{arg}'"
            else:
                return f"Output '{arg}' not found"
        
        elif cmd == "register":
            if not arg:
                return "on" if self.agent.register_machines else "off"
            if arg.lower() == "on":
                self.agent.register_machines = True
                # Apply current machines immediately
                fs_root = getattr(self.agent, '_fs_root', None)
                if fs_root is not None:
                    for machine in fs_root.get_machines():
                        self.agent.add_machine_rule(machine)
                return "Machine registration enabled"
            elif arg.lower() == "off":
                self.agent.register_machines = False
                # Remove all machine rules
                for machine in list(self.agent._machine_rules.keys()):
                    self.agent.remove_machine_rule(machine)
                return "Machine registration disabled"
            else:
                return "Usage: register on|off"
        
        elif cmd == "history":
            if not arg:
                return "on" if self.agent.history_active else "off"
            if arg.lower() == "on":
                self.agent.history_active = True
                return "History enabled (existing history will be sent to provider)"
            elif arg.lower() == "off":
                self.agent.history_active = False
                return "History disabled (only latest message sent, history still accumulates)"
            else:
                return "Usage: history on|off"
        
        else:
            raise ValueError(f"Unknown command: {cmd}")
    
    async def get_status(self) -> bytes:
        a = self.agent
        
        # Calculate current token usage
        history_tokens = sum(estimate_message_tokens(m) for m in a.history)
        system_tokens = estimate_tokens(a.config.system) if a.config.system else 0
        total_context_tokens = history_tokens + system_tokens
        
        lines = [
            f"state {a.state.value}",
            f"provider {a.provider.name}",
            f"model {a.config.model}",
            f"messages {len(a.history)}",
            f"temperature {a.config.temperature}",
            f"max_tokens {a.config.max_tokens}",
            f"max_history {a.config.max_history}",
            f"max_context_tokens {a.max_context_tokens}",
            f"context_tokens {total_context_tokens}",
            f"rules {len(a.plumbing_rules)}",
            f"supplementary_outputs {len(a.supplementary_outputs)}",
            f"register {'on' if a.register_machines else 'off'}",
            f"history {'on' if a.history_active else 'off'}",
        ]
        
        if a.config.system:
            sys_preview = a.config.system[:50] + "..." if len(a.config.system) > 50 else a.config.system
            lines.append(f"system {sys_preview}")
        
        if a.state == AgentState.ERROR and a.last_error:
            lines.append(f"error {a.last_error}")
        
        if a.state == AgentState.STREAMING:
            lines.append(f"tokens_out ~{a._tokens_out}")
        
        if a._tokens_in > 0:
            lines.append(f"last_tokens_in ~{a._tokens_in}")
        
        return ("\n".join(lines) + "\n").encode()


class AgentInputFile(SyntheticFile):
    """
    Write prompts to this file to trigger generation.
    
    Writing is buffered across 9P chunks. Generation is triggered
    on clunk (fid close) so that multi-chunk prompts don't spawn
    multiple generate() calls.
    """
    
    def __init__(self, agent: 'Agent'):
        super().__init__("input")
        self.agent = agent
        self._last_input = ""
        self._write_buffers = {}  # fid -> bytearray
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """Read last input"""
        return self._last_input.encode()[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        """Buffer write data - generation happens on clunk"""
        fid_key = id(fid)
        if fid_key not in self._write_buffers:
            self._write_buffers[fid_key] = bytearray()
        
        buf = self._write_buffers[fid_key]
        
        if offset == 0 and len(buf) > 0:
            buf.clear()
        
        if offset + len(data) > len(buf):
            buf.extend(b'\x00' * (offset + len(data) - len(buf)))
        buf[offset:offset + len(data)] = data
        
        return len(data)
    
    async def clunk(self, fid: FidState):
        """Trigger generation with the complete buffered prompt."""
        fid_key = id(fid)
        buf = self._write_buffers.pop(fid_key, None)
        
        if not buf:
            return
        
        raw_data = bytes(buf)
        if not raw_data.strip():
            return
        
        # Parse input: detect media vs text vs mixed
        blocks = parse_input_data(raw_data)
        
        if not blocks:
            return
        
        # Extract text content for display/history
        text_parts = [b.text for b in blocks if b.type == "text" and b.text]
        media_parts = [b for b in blocks if b.type == "media"]
        
        # Build display text
        if text_parts:
            display_text = " ".join(text_parts)
        elif media_parts:
            descriptions = []
            for b in media_parts:
                size_kb = len(b.media.data) / 1024
                descriptions.append(f"[{b.media.mime_type} {size_kb:.0f}KB]")
            display_text = " ".join(descriptions)
        else:
            display_text = "(empty)"
        
        self._last_input = display_text
        
        # Start generation with content blocks
        asyncio.create_task(self.agent.generate(display_text, content_blocks=blocks))


class AgentHistoryFile(SyntheticFile):
    """
    Read agent history as JSON.
    
    Write semantics follow Unix conventions:
    
        echo 'hello' > $agent/history   # truncate: clears history, adds as user message
        echo 'hello' >> $agent/history  # append: adds to existing history as user message
    
    Truncate vs append is detected via the first write offset per fid:
    offset 0 = truncate (shell '>' opens with OTRUNC, writes at 0)
    offset > 0 = append  (shell '>>' seeks to end-of-file first)
    
    JSON input is also supported:
    - JSON array → replaces (>) or extends (>>) history
    - JSON object with role+content → replaces history with single msg (>) or appends (>>)
    - Plain text → treated as a user message
    - "clear" or "delete" → always clears history regardless of mode
    """
    
    def __init__(self, agent: 'Agent'):
        super().__init__("history")
        self.agent = agent
        self._write_buffers = {}   # fid -> bytearray
        self._first_offset = {}    # fid -> int (first write offset, to detect > vs >>)
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        history = []
        for m in self.agent.history:
            entry = {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
            }
            if m.has_media:
                entry["media"] = [
                    {
                        "type": b.media.media_type,
                        "mime": b.media.mime_type,
                        "size": len(b.media.data),
                    }
                    for b in m.content_blocks if b.type == "media" and b.media
                ]
            history.append(entry)
        raw = json.dumps(history, indent=2, ensure_ascii=False)
        # Unescape \\n inside JSON string values to real newlines
        # so that `cat history` shows readable multi-line content
        data = re.sub(
            r'"(?:[^"\\]|\\.)*"',
            lambda m: m.group(0).replace('\\n', '\n'),
            raw
        ).encode()
        return data[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        """
        Buffer write data. Processing happens on clunk so that
        multi-chunk 9P writes don't create multiple history entries.
        
        Tracks the first write offset to distinguish > (offset 0) from >> (offset > 0).
        """
        fid_key = id(fid)
        if fid_key not in self._write_buffers:
            self._write_buffers[fid_key] = bytearray()
            self._first_offset[fid_key] = offset  # Record first write offset
        
        buf = self._write_buffers[fid_key]
        
        # Offset 0 with existing data = new write sequence
        if offset == 0 and len(buf) > 0:
            buf.clear()
        
        # Extend buffer to fit
        if offset + len(data) > len(buf):
            buf.extend(b'\x00' * (offset + len(data) - len(buf)))
        buf[offset:offset + len(data)] = data
        
        return len(data)
    
    async def clunk(self, fid: FidState):
        """Process the complete buffered write on fid close."""
        fid_key = id(fid)
        buf = self._write_buffers.pop(fid_key, None)
        first_offset = self._first_offset.pop(fid_key, 0)
        
        if not buf:
            return
        
        text = bytes(buf).decode('utf-8', errors='replace').strip()
        
        # first_offset == 0 means '>' (truncate), > 0 means '>>' (append)
        truncate = (first_offset == 0)
        self._process_history_write(text, truncate=truncate)
    
    def _process_history_write(self, text: str, truncate: bool = True):
        """
        Process a complete history write.
        
        truncate=True  (echo '...' > history):
            Clears existing history first, then adds the new content.
        truncate=False (echo '...' >> history):
            Appends to existing history without clearing.
        
        Content parsing (applies in both modes):
        - Empty string, "delete", or "clear" → clears history
        - Valid JSON array of messages → replaces (truncate) or extends (append) history
        - Valid JSON object with "role" and "content" → single message
        - Any other text → user message
        """
        # Clear commands always clear regardless of mode
        if not text or text.lower() in ("delete", "clear"):
            self.agent.history.clear()
            return
        
        # Try JSON first (legacy import + single message object)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                messages = [
                    Message(
                        role=m["role"],
                        content=m["content"],
                        timestamp=m.get("timestamp", time.time())
                    )
                    for m in parsed
                ]
                if truncate:
                    self.agent.history = messages
                else:
                    self.agent.history.extend(messages)
                return
            elif isinstance(parsed, dict) and "role" in parsed and "content" in parsed:
                msg = Message(
                    role=parsed["role"],
                    content=parsed["content"],
                    timestamp=parsed.get("timestamp", time.time())
                )
                if truncate:
                    self.agent.history = [msg]
                else:
                    self.agent.history.append(msg)
                return
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        
        # Default: treat as plain text user message
        msg = Message(role="user", content=text)
        if truncate:
            self.agent.history = [msg]
        else:
            self.agent.history.append(msg)


class AgentConfigFile(SyntheticFile):
    """
    Read/write agent configuration as JSON.
    """
    
    def __init__(self, agent: 'Agent'):
        super().__init__("config")
        self.agent = agent
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        config = {
            "provider": self.agent.provider.name,
            "model": self.agent.config.model,
            "system": self.agent.config.system,
            "temperature": self.agent.config.temperature,
            "max_tokens": self.agent.config.max_tokens,
            "max_history": self.agent.config.max_history,
            "max_context_tokens": self.agent.max_context_tokens,
        }
        data = json.dumps(config, indent=2).encode()
        return data[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        try:
            config = json.loads(data.decode())
            if "model" in config:
                self.agent.config.model = config["model"]
            if "system" in config:
                self.agent.config.system = config["system"]
            if "temperature" in config:
                self.agent.config.temperature = config["temperature"]
            if "max_tokens" in config:
                self.agent.config.max_tokens = config["max_tokens"]
            if "max_history" in config:
                self.agent.config.max_history = config["max_history"]
            if "max_context_tokens" in config:
                self.agent.max_context_tokens = int(config["max_context_tokens"])
            return len(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")


class AgentSystemFile(SyntheticFile):
    """
    Read/write system prompt.
    
    Reading returns the effective system prompt: the base system prompt
    followed by any supplementary context sections (from writes to
    supplementary output files like DAVID, ALICE, etc.).
    
    Writing sets only the base system prompt. Supplementary contexts
    are managed by writing to the individual supplementary output files.
    """
    
    def __init__(self, agent: 'Agent'):
        super().__init__("system")
        self.agent = agent
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        content = self.agent.get_effective_system()
        data = content.encode()
        return data[offset:offset + count]
    
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


class AgentRulesFile(SyntheticFile):
    """
    Plumbing rules define regex patterns that extract structured data
    from agent output and create supplementary output files.
    
    Full format: pattern -> {capture_name}
    Example: (?P<rioa>\\S+)\\n(?P<code>.*?) -> {rioa}
    
    Shorthand format: just a word (no '->' needed)
    Example: echo 'bash' > $agent/rules
    Expands to: (?:^|\\n)```(?P<bash>\\S*)\\n(?P<code>.*?)\\n```(?=\\s|\\Z) -> {bash}
    
    The shorthand pattern matches OUTER fenced code blocks only.
    Inner fences (e.g. ```python inside a ```bash block) are treated
    as part of the code content, not as block boundaries.  This lets
    the agent safely write things like:
    
        ```bash
        printf '```python
        import foo
        ```' > /n/rioa/scene/parse
        ```
    
    The bash rule captures the entire printf command (including the
    inner ```python ... ``` content).  A subsequent python rule will
    NOT re-match the inner fence because consumed-region tracking in
    _apply_plumbing skips already-extracted regions.
    
    This means `echo 'bash' > rules` creates a rule that extracts
    fenced code blocks tagged with ```bash into a 'bash' file.
    
    Multiple rules can create multiple output files:
    - echo 'bash' > rules
    - echo 'python' > rules
    - echo '(?P<rioa>\\S+)\\n(?P<code>.*?)' -> {rioa}' > rules
    
    This creates supplementary files 'bash', 'python', and 'rioa'.
    """
    
    def __init__(self, agent: 'Agent'):
        super().__init__("rules")
        self.agent = agent
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        lines = []
        for i, rule in enumerate(self.agent.plumbing_rules):
            lines.append(f"{i}: {rule['pattern']} -> {{{rule['output_name']}}}")
        data = ("\n".join(lines) + "\n").encode() if lines else b"(no rules)\n"
        return data[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        """Add plumbing rule(s) - supports multiple lines"""
        rule_text = data.decode().strip()
        if not rule_text:
            return len(data)
        
        # Split by newlines to support multiple rules at once
        lines = [line.strip() for line in rule_text.split('\n') if line.strip()]
        
        import re as re_module
        for rule_line in lines:
            # Check if this is a full rule (contains '->') or a shorthand
            if '->' in rule_line:
                # Full format: pattern -> {output_name}
                parts = rule_line.split("->", 1)
                if len(parts) != 2:
                    raise ValueError(f"Format: pattern -> {{output_name}}. Got: {rule_line}")
                
                pattern = parts[0].strip()
                target = parts[1].strip()
                
                # Extract output name from {brackets}
                match = re_module.match(r'\{(\w+)\}', target)
                if not match:
                    raise ValueError(f"Target must be {{capture_name}} format. Got: {target}")
                
                output_name = match.group(1)
            else:
                # Shorthand: just a word like "bash" or "rioa"
                # Expands to a pattern that matches OUTER fenced code blocks only.
                # The opening fence must be at start-of-string or after a newline,
                # and the closing fence must be on its own line, so inner fences
                # (e.g. ```python ... ``` inside a bash block) are treated as
                # part of the code content, not as block boundaries.
                word = rule_line.strip()
                if not re_module.match(r'^\w+$', word):
                    raise ValueError(
                        f"Shorthand must be a single word (e.g. 'bash'). "
                        f"For custom patterns use: pattern -> {{output_name}}. Got: {rule_line}"
                    )
                output_name = word
                pattern = f"(?:^|\\n)```(?P<{word}>\\S*)\\n(?P<code>.*?)\\n```(?=\\s|\\Z)"
            
            # Validate regex and ensure it has the output_name capture group
            try:
                compiled = re.compile(pattern)
                if output_name not in compiled.groupindex:
                    raise ValueError(f"Pattern must contain (?P<{output_name}>...) capture group")
            except re.error as e:
                raise ValueError(f"Invalid regex: {e}")
            
            # Create the supplementary output file if it doesn't exist
            self.agent.create_supplementary_output(output_name)
            
            self.agent.plumbing_rules.append({
                "pattern": pattern,
                "output_name": output_name
            })
        
        return len(data)


class AgentStateFile(SyntheticFile):
    """
    Snapshot/restore full agent state.
    
    Reading this file serializes the complete agent state (provider, model,
    temperature, max_tokens, system prompt, history, plumbing rules) as JSON.
    
    Writing a JSON snapshot restores all of that state, enabling:
    
        cp agent_1/state agent_2/state     # clone agent_1 into agent_2
        cp agent/state /tmp/agent.state     # save to disk
        cp /tmp/agent.state agent/state     # restore from disk
    
    The state file is buffered: multi-chunk 9P writes are assembled and
    applied on clunk (fid close), just like input and history.
    """
    
    def __init__(self, agent: 'Agent'):
        super().__init__("state")
        self.agent = agent
        self._write_buffers = {}  # fid -> bytearray
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """Serialize complete agent state to JSON."""
        snapshot = {
            "provider": self.agent.provider.name,
            "model": self.agent.config.model,
            "system": self.agent.config.system,
            "temperature": self.agent.config.temperature,
            "max_tokens": self.agent.config.max_tokens,
            "max_history": self.agent.config.max_history,
            "max_context_tokens": self.agent.max_context_tokens,
            "history_active": self.agent.history_active,
            "register_machines": self.agent.register_machines,
            "history": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "content_blocks": [b.to_dict() for b in m.content_blocks] if m.content_blocks else [],
                }
                for m in self.agent.history
            ],
            "rules": [
                {
                    "pattern": r["pattern"],
                    "output_name": r["output_name"],
                }
                for r in self.agent.plumbing_rules
            ],
            "sup_contexts": {
                name: sup.context
                for name, sup in self.agent.supplementary_outputs.items()
                if sup.context
            },
        }
        data = json.dumps(snapshot, indent=2, ensure_ascii=False).encode()
        return data[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        """Buffer write data — state is restored on clunk."""
        fid_key = id(fid)
        if fid_key not in self._write_buffers:
            self._write_buffers[fid_key] = bytearray()
        
        buf = self._write_buffers[fid_key]
        
        if offset == 0 and len(buf) > 0:
            buf.clear()
        
        if offset + len(data) > len(buf):
            buf.extend(b'\x00' * (offset + len(data) - len(buf)))
        buf[offset:offset + len(data)] = data
        
        return len(data)
    
    async def clunk(self, fid: FidState):
        """Apply the buffered snapshot on fid close."""
        fid_key = id(fid)
        buf = self._write_buffers.pop(fid_key, None)
        
        if not buf:
            return
        
        text = bytes(buf).decode('utf-8', errors='replace').strip()
        if not text:
            return
        
        try:
            snapshot = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid state JSON: {e}")
        
        # --- Restore provider (best-effort; fall back to current) ---
        if "provider" in snapshot:
            try:
                from .providers import get_provider
                self.agent.provider = get_provider(snapshot["provider"])
            except Exception:
                pass  # keep current provider if requested one is unavailable
        
        # --- Restore config ---
        if "model" in snapshot:
            self.agent.config.model = snapshot["model"]
        if "system" in snapshot:
            self.agent.config.system = snapshot["system"]
        if "temperature" in snapshot:
            self.agent.config.temperature = float(snapshot["temperature"])
        if "max_tokens" in snapshot:
            self.agent.config.max_tokens = int(snapshot["max_tokens"])
        if "max_history" in snapshot:
            self.agent.config.max_history = int(snapshot["max_history"])
        if "max_context_tokens" in snapshot:
            self.agent.max_context_tokens = int(snapshot["max_context_tokens"])
        if "history_active" in snapshot:
            self.agent.history_active = bool(snapshot["history_active"])
        if "register_machines" in snapshot:
            self.agent.register_machines = bool(snapshot["register_machines"])
        
        # --- Restore history ---
        if "history" in snapshot:
            self.agent.history = []
            for m in snapshot["history"]:
                blocks = []
                for bd in m.get("content_blocks", []):
                    try:
                        blocks.append(ContentBlock.from_dict(bd))
                    except Exception:
                        pass  # Skip malformed blocks
                self.agent.history.append(Message(
                    role=m["role"],
                    content=m["content"],
                    timestamp=m.get("timestamp", time.time()),
                    content_blocks=blocks,
                ))
        
        # --- Restore plumbing rules & supplementary outputs ---
        if "rules" in snapshot:
            # Clear existing rules and outputs
            self.agent.plumbing_rules.clear()
            # Remove old supplementary output files from the directory
            for out_name in list(self.agent.supplementary_outputs.keys()):
                self.agent.remove(out_name.upper())
            self.agent.supplementary_outputs.clear()
            
            # Re-create from snapshot
            for rule in snapshot["rules"]:
                pattern = rule["pattern"]
                output_name = rule["output_name"]
                
                # Validate regex
                try:
                    compiled = re.compile(pattern)
                    if output_name not in compiled.groupindex:
                        continue  # skip invalid rule silently
                except re.error:
                    continue
                
                self.agent.create_supplementary_output(output_name)
                self.agent.plumbing_rules.append({
                    "pattern": pattern,
                    "output_name": output_name,
                })
        
        # --- Restore supplementary contexts ---
        if "sup_contexts" in snapshot:
            for name, context in snapshot["sup_contexts"].items():
                if name in self.agent.supplementary_outputs:
                    self.agent.supplementary_outputs[name].context = context


class Agent(SyntheticDir):
    """
    Agent represents a conversational LLM session.
    
    Each agent has:
    - ctl: Control interface
    - input: Write prompts here
    - OUTPUT: Stream of assistant responses (CAPS = blocking read)
    - history: JSON conversation history
    - config: JSON configuration
    - system: System prompt
    - rules: Plumbing rules for extraction
    - state: Snapshot/restore complete agent state
    - errors: Error messages
    - {NAME1}, {NAME2}, ...: Supplementary outputs created by rules (CAPS = blocking read)
    
    Files with blocking read semantics use CAPITAL names as a visual
    indicator: reading them will block until content is available.
    
    Supplementary outputs are dynamically created when rules are added.
    """
    
    def __init__(
        self,
        name: str,
        provider: LLMProvider,
        route_manager = None,
        default_model: str = None
    ):
        super().__init__(name)
        self.agent_name = name
        self.provider = provider
        
        # State
        self.state = AgentState.IDLE
        self.history: List[Message] = []
        self.last_error: Optional[str] = None
        self._cancel_event = asyncio.Event()
        self._current_task: Optional[asyncio.Task] = None
        self._tokens_out = 0
        self._tokens_in = 0  # Estimated input tokens for last request
        
        # Context window management
        # Set to the model's context limit minus max_tokens (output reserve).
        # 0 = disabled (use max_history message count only).
        self.max_context_tokens: int = 0
        
        # Configuration
        self.config = ProviderConfig(
            model=default_model or provider.default_model
        )
        
        # Plumbing rules and supplementary outputs
        self.plumbing_rules: List[Dict[str, str]] = []
        self.supplementary_outputs: Dict[str, SupplementaryOutputFile] = {}
        
        # Machine registration: when True, auto-create rules for mounted machines
        self.register_machines: bool = False
        # Track which machine rules we auto-created (so we can clean up)
        self._machine_rules: Dict[str, int] = {}  # machine_name -> rule index
        
        # History toggle: when False, history is not sent to the provider
        # but still accumulates. Can be re-enabled at any time.
        self.history_active: bool = True
        
        # Create child files
        self.output = StreamFile("OUTPUT")
        # StreamFile starts with generation gate closed (not set), so reads
        # naturally block until the first generate() calls reset(). No hack needed.
        self.errors = QueueFile("errors")
        
        self.add(CtlFile("ctl", AgentCtlHandler(self)))
        self.add(AgentInputFile(self))
        self.add(self.output)
        self.add(AgentHistoryFile(self))
        self.add(AgentConfigFile(self))
        self.add(AgentSystemFile(self))
        self.add(AgentRulesFile(self))
        self.add(AgentStateFile(self))
        self.add(self.errors)
    
    def get_effective_system(self) -> str:
        """
        Build the complete system prompt: base system + supplementary contexts.
        
        Returns the agent's config.system followed by context sections
        for ALL supplementary output files. Files with written context
        show their content; files without context show "empty":
        
            <base system prompt>
            
            CONTEXT FOR DAVID:
            <content written to DAVID sup file>
            
            CONTEXT FOR ALICE:
            empty
        """
        parts = []
        
        # Base system prompt
        base = self.config.system or ""
        if base:
            parts.append(base)
        
        # Supplementary contexts (from writes to sup files)
        for name, sup_file in sorted(self.supplementary_outputs.items()):
            if sup_file.context:
                parts.append(f"CONTEXT FOR {sup_file.name.lower()}:\n{sup_file.context}")
            else:
                parts.append(f"CONTEXT FOR {sup_file.name.lower()}:\nempty")
        
        return "\n\n".join(parts)
    
    def create_supplementary_output(self, name: str) -> SupplementaryOutputFile:
        """Create a new supplementary output file.
        
        The file is created with an UPPERCASED name as a visual indicator
        that reads on it will block (waiting for content to be generated).
        The supplementary_outputs dict still keys by the original lowercase
        name so that plumbing rule lookups work unchanged.
        """
        if name in self.supplementary_outputs:
            return self.supplementary_outputs[name]
        
        output_file = SupplementaryOutputFile(name.upper())
        self.supplementary_outputs[name] = output_file
        self.add(output_file)
        return output_file
    
    def add_machine_rule(self, machine_name: str):
        """
        Auto-create a plumbing rule for a mounted machine.
        
        Creates a rule that extracts fenced code blocks tagged with the
        machine name (e.g. ```david ... ```) into a supplementary output
        file named DAVID.
        
        Only creates the rule if register_machines is True and the
        machine hasn't already been registered.
        """
        name_lower = machine_name.lower()
        
        # Skip if already registered
        if name_lower in self._machine_rules:
            return
        
        # Create the shorthand pattern (same as AgentRulesFile shorthand)
        pattern = f"(?:^|\\n)```(?P<{name_lower}>\\S*)\\n(?P<code>.*?)\\n```(?=\\s|\\Z)"
        
        # Validate
        import re as re_module
        try:
            compiled = re_module.compile(pattern)
            if name_lower not in compiled.groupindex:
                return
        except re_module.error:
            return
        
        # Create supplementary output file
        self.create_supplementary_output(name_lower)
        
        # Add rule
        rule = {"pattern": pattern, "output_name": name_lower}
        self.plumbing_rules.append(rule)
        rule_idx = len(self.plumbing_rules) - 1
        self._machine_rules[name_lower] = rule_idx
    
    def remove_machine_rule(self, machine_name: str):
        """
        Remove an auto-created machine rule.
        
        Removes the plumbing rule and supplementary output file.
        """
        name_lower = machine_name.lower()
        
        if name_lower not in self._machine_rules:
            return
        
        del self._machine_rules[name_lower]
        
        # Remove matching plumbing rules
        self.plumbing_rules = [
            r for r in self.plumbing_rules
            if r["output_name"] != name_lower
        ]
        
        # Remove supplementary output file
        if name_lower in self.supplementary_outputs:
            self.remove(name_lower.upper())
            del self.supplementary_outputs[name_lower]
    
    async def generate(self, prompt: str, content_blocks: List[ContentBlock] = None):
        """
        Generate response for the given prompt.
        
        This is the main entry point for LLM interaction.
        
        Args:
            prompt: Text content for display/history
            content_blocks: Optional list of ContentBlock (text + media).
                           If None, a single text block is created from prompt.
        """
        # Check if already streaming
        if self.state == AgentState.STREAMING:
            await self.errors.post(b"Already streaming. Cancel first.\n")
            return
        
        # Reset state
        self._cancel_event.clear()
        self.state = AgentState.STREAMING
        self.last_error = None
        self._tokens_out = 0
        
        # Build content blocks if not provided
        if content_blocks is None:
            content_blocks = [ContentBlock(type="text", text=prompt)]
        
        # Add user message with content blocks
        self.history.append(Message(
            role="user",
            content=prompt,
            content_blocks=content_blocks,
        ))
        
        # Reset output stream
        await self.output.reset()
        
        # Clear all supplementary outputs for next generation
        for output in self.supplementary_outputs.values():
            output.clear()
        
        # Roll history if over max_history (keep most recent messages)
        if self.config.max_history > 0 and len(self.history) > self.config.max_history:
            overflow = len(self.history) - self.config.max_history
            del self.history[:overflow]
        
        # Token-based history rolling
        # If max_context_tokens is set, trim oldest messages until we fit
        if self.max_context_tokens > 0:
            effective_system = self.get_effective_system()
            system_tokens = estimate_tokens(effective_system) if effective_system else 0
            # Reserve space for system prompt + some overhead for message framing
            budget = self.max_context_tokens - system_tokens
            
            # Calculate tokens from newest to oldest, keep what fits
            kept = []
            used = 0
            for msg in reversed(self.history):
                msg_tokens = estimate_message_tokens(msg)
                if used + msg_tokens > budget:
                    break
                kept.append(msg)
                used += msg_tokens
            
            # If we had to trim, update history (keep at least the latest message)
            if len(kept) < len(self.history):
                trimmed_count = len(self.history) - len(kept)
                self.history = list(reversed(kept)) if kept else [self.history[-1]]
                # Post a notice to errors stream
                await self.errors.post(
                    f"Auto-trimmed {trimmed_count} old messages to fit context window "
                    f"(~{used} of {self.max_context_tokens} token budget used)\n".encode()
                )
            
            self._tokens_in = used + system_tokens
        
        # Build config with current history (content blocks for provider formatting)
        # Use the effective system prompt (base + supplementary contexts)
        # Save and restore config.system so the base prompt is not overwritten
        base_system = self.config.system
        self.config.system = self.get_effective_system()
        
        self.config.history = []
        if self.history_active:
            for m in self.history:
                if m.is_multimodal:
                    self.config.history.append({
                        "role": m.role,
                        "content": m.content,
                        "content_blocks": m.content_blocks,
                    })
                else:
                    self.config.history.append({
                        "role": m.role,
                        "content": m.content,
                    })
        else:
            # History disabled: send only the latest user message
            # (provider needs at least one message to respond to)
            latest = self.history[-1] if self.history else None
            if latest:
                if latest.is_multimodal:
                    self.config.history.append({
                        "role": latest.role,
                        "content": latest.content,
                        "content_blocks": latest.content_blocks,
                    })
                else:
                    self.config.history.append({
                        "role": latest.role,
                        "content": latest.content,
                    })
        
        assistant_content = ""
        
        try:
            async for chunk in self.provider.stream_response(self.config):
                # Check for cancellation
                if self._cancel_event.is_set():
                    self.state = AgentState.CANCELLED
                    break
                
                assistant_content += chunk
                self._tokens_out += len(chunk.split())  # Rough token estimate
                
                chunk_bytes = chunk.encode('utf-8')
                
                # Write to output stream
                await self.output.append(chunk_bytes)
            
            # Add assistant message to history
            if assistant_content:
                self.history.append(Message(role="assistant", content=assistant_content))
                # Apply plumbing rules to extract content into supplementary outputs
                await self._apply_plumbing(assistant_content)
            
            if self.state == AgentState.STREAMING:
                self.state = AgentState.DONE
        
        except asyncio.CancelledError:
            self.state = AgentState.CANCELLED
            if assistant_content:
                self.history.append(Message(role="assistant", content=assistant_content + " [cancelled]"))
        
        except Exception as e:
            self.state = AgentState.ERROR
            self.last_error = f"{type(e).__name__}: {e}"
            error_msg = f"Provider error: {self.last_error}\n"
            await self.errors.post(error_msg.encode())
        
        finally:
            # Restore the base system prompt (effective system was temporary)
            self.config.system = base_system
            # Signal stream complete
            await self.output.finish()
    
    async def _apply_plumbing(self, content: str):
        """
        Apply plumbing rules to extract content into supplementary outputs.
        
        Each rule extracts content matching its pattern and writes ONLY to
        its designated output file when the named capture group's VALUE
        matches the output_name.
        
        Example: ```(?P<rioa>\S+)\n(?P<code>.*?)``` -> {rioa}
        Only extracts when the captured value of rioa IS "rioa"
        
        CONSUMED REGIONS: Once a rule matches a region of the content,
        that region is marked as consumed. Later rules skip matches
        that overlap consumed regions. This prevents e.g. a 'python'
        rule from extracting inner ```python ... ``` fences that live
        inside a 'bash' block already captured by the 'bash' rule.
        
        After extraction, marks outputs as ready so blocked reads can proceed.
        """
        # Track consumed byte ranges so later rules don't re-extract
        # content that was already captured by an earlier rule.
        consumed_ranges = []  # list of (start, end) tuples
        
        def is_consumed(start: int, end: int) -> bool:
            """Check if a match overlaps any already-consumed region."""
            for cs, ce in consumed_ranges:
                if start < ce and end > cs:  # overlapping
                    return True
            return False
        
        for rule in self.plumbing_rules:
            output_name = rule["output_name"]
            pattern = rule["pattern"]
            
            try:
                matches = re.finditer(pattern, content, re.DOTALL)
                for m in matches:
                    # Skip matches that overlap already-consumed regions
                    if is_consumed(m.start(), m.end()):
                        continue
                    
                    groups = m.groupdict()
                    
                    # CRITICAL FIX: Only extract if this match has the specific
                    # named capture group AND its value matches the output name
                    # This prevents cross-contamination:
                    #   Pattern: ```(?P<rioa>\S+)\n(?P<code>.*?)``` -> {rioa}
                    #   Matches both ```rioa\ncode``` and ```riob\ncode```
                    #   But only extracts when groups['rioa'] == 'rioa'
                    if output_name in groups and groups[output_name] == output_name:
                        # Extract the code/content payload
                        # First try "code" group, then "content", then fall back to output_name group
                        payload = groups.get("code") or groups.get("content") or groups.get(output_name)
                        
                        if payload:
                            # Mark this region as consumed
                            consumed_ranges.append((m.start(), m.end()))
                            
                            # Add block to THIS supplementary output only
                            if output_name in self.supplementary_outputs:
                                self.supplementary_outputs[output_name].add_block(payload)
                            else:
                                await self.errors.post(
                                    f"Warning: Output '{output_name}' not found for rule\n".encode()
                                )
                        
            except Exception as e:
                await self.errors.post(
                    f"Plumbing rule execution failed for {output_name}: {e}\n".encode()
                )
        
        # Mark all supplementary outputs as ready (unblocks waiting reads)
        for output in self.supplementary_outputs.values():
            output.mark_ready()
    
    async def execute_history_rule(self, rule: Dict[str, str]):
        """
        Execute a rule against all history messages.
        
        Scans all history, extracts matches, and adds them as blocks
        to the supplementary output file.
        
        Uses the same consumed-region tracking as _apply_plumbing
        (per-message) to avoid double-extraction within a single message.
        """
        output_name = rule["output_name"]
        pattern = rule["pattern"]
        
        # Clear existing content in this output
        if output_name in self.supplementary_outputs:
            self.supplementary_outputs[output_name].clear()
        
        # Extract matches from entire history
        for msg in self.history:
            try:
                consumed_ranges = []
                
                def is_consumed(start: int, end: int) -> bool:
                    for cs, ce in consumed_ranges:
                        if start < ce and end > cs:
                            return True
                    return False
                
                matches = re.finditer(pattern, msg.content, re.DOTALL)
                for m in matches:
                    if is_consumed(m.start(), m.end()):
                        continue
                    
                    groups = m.groupdict()
                    
                    # Only extract if the captured value matches the output name
                    if output_name in groups and groups[output_name] == output_name:
                        payload = groups.get("code") or groups.get("content") or groups.get(output_name)
                        
                        if payload:
                            consumed_ranges.append((m.start(), m.end()))
                            # Add block to supplementary output
                            if output_name in self.supplementary_outputs:
                                self.supplementary_outputs[output_name].add_block(payload)
                        
            except Exception as e:
                await self.errors.post(
                    f"History extraction failed for {output_name}: {e}\n".encode()
                )
        
        # Mark output as ready
        if output_name in self.supplementary_outputs:
            self.supplementary_outputs[output_name].mark_ready()
    
    async def clear(self):
        """Clear agent history"""
        await self.cancel()
        self.history.clear()
        self.state = AgentState.IDLE
        self.last_error = None
        await self.output.reset()
        
        # Clear all supplementary outputs (though they auto-clear on read)
        for output in self.supplementary_outputs.values():
            output.clear()
    
    async def cancel(self):
        """Cancel current generation"""
        if self.state == AgentState.STREAMING:
            self._cancel_event.set()
            # Give it a moment to stop
            await asyncio.sleep(0.1)
    
    async def retry(self):
        """Retry the last user message"""
        if len(self.history) >= 2:
            # Remove last assistant response if any
            if self.history[-1].role == "assistant":
                self.history.pop()
            
            # Get last user message
            if self.history and self.history[-1].role == "user":
                last_prompt = self.history.pop().content
                await self.generate(last_prompt)