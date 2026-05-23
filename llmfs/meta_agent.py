"""
Meta-Agent for LLMFS

The meta-agent is a special text Agent that writes the source code for new
agent types. You talk to it like any other agent ("Create an image-gen
agent"); it streams a complete Python module into its OUTPUT, and a
plumbing rule extracts the ```python ... ``` block into the supplementary
output PYTHON.

A custom ctl command `build <name>` then:
  1. Reads PYTHON
  2. Writes it to $LLMFS_AGENTS_DIR/<name>.py
  3. Imports it
  4. Calls its `create(name)` factory
  5. Mounts the returned SyntheticDir at /n/llm/<name>

A `load <name>` command re-imports a previously-saved agent without
talking to the LLM (used at startup and to manually re-load).

Generated-agent contract
------------------------
Files saved to $LLMFS_AGENTS_DIR must expose:

    def create(name: str) -> SyntheticDir:
        ...

The returned object must be a core.files.SyntheticDir whose .name has
already been set to `name`. Anything else is up to the generated code.
"""

import asyncio
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

from core.files import SyntheticDir, SyntheticFile, CtlFile
from core.types import FidState

from .agent import Agent, AgentCtlHandler
from .providers import LLMProvider


# ---------------------------------------------------------------------------
# HelpFile — shared helper that generated agents embed in their tree
# ---------------------------------------------------------------------------

class HelpFile(SyntheticFile):
    """
    Read-only documentation file. Generated agents add one of these to
    their tree so `cat /n/llm/<name>/help` always works the same way.

    Held in meta_agent.py rather than core/ because the help-file format
    is a meta-agent convention, not a 9P-server primitive — generated
    agents import it via `from llmfs.meta_agent import HelpFile`.
    """

    def __init__(self, text: str, name: str = "help"):
        super().__init__(name)
        # Normalize line endings and ensure a trailing newline so terminal
        # `cat` output looks clean.
        if not text.endswith("\n"):
            text = text + "\n"
        self._data = text.encode("utf-8")

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        return self._data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("help file is read-only")




def get_agents_dir() -> Path:
    """
    Resolve the directory where generated agents are persisted.

    Honours $LLMFS_AGENTS_DIR; falls back to ~/.llmfs/agents. The directory
    is created on first access.
    """
    env = os.environ.get("LLMFS_AGENTS_DIR")
    if env:
        path = Path(env).expanduser().resolve()
    else:
        path = Path.home() / ".llmfs" / "agents"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

META_SYSTEM_PROMPT = """\
You are the LLMFS meta-agent. Your sole job is to write a complete,
self-contained Python module that implements a new LLMFS agent.

When the user asks for an agent (e.g. "create an image-gen agent",
"build a weather agent"), respond with ONE ```python``` fenced code
block containing the full module, and nothing else of substance
afterwards. Brief preamble before the block is fine; never put prose
*inside* the block.

================================================================
HARD CONTRACT — the loader depends on this
================================================================

Every module you produce MUST:

1. Define a class deriving from `core.files.SyntheticDir`.
2. Build its filesystem tree in __init__ by calling `self.add(...)` on
   each child file or subdir.
3. Expose a module-level factory:

       def create(name: str) -> SyntheticDir:
           return MyAgent(name)

   The loader calls this with the agent's name; do NOT hardcode the name.

4. Be importable in isolation: only depend on the standard library,
   `core.files`, `core.types`, and `llmfs.*`. Do NOT import third-party
   packages unless the user explicitly asked for them.

================================================================
STANDARD FILESYSTEM LAYOUT
================================================================

Every agent MUST expose these children:

    <name>/
    ├── ctl        # CtlFile with a CtlHandler subclass
    ├── input      # SyntheticFile: write to trigger work
    ├── OUTPUT     # StreamFile: blocking read for streaming results
    ├── errors     # QueueFile: error stream
    ├── help       # SyntheticFile: human-readable docs (read-only)
    └── (whatever else the task needs)

UPPERCASE names are a visual cue that reads on the file may block.

================================================================
THE help FILE — REQUIRED
================================================================

Every agent MUST include a `help` file. Reading it tells the user
exactly how to use the agent. The format is fixed:

    NAME
        <agent-name> — <one-line tagline>

    SYNOPSIS
        echo "<example input>" > /n/llm/<name>/input
        cat /n/llm/<name>/OUTPUT

    DESCRIPTION
        2-5 sentence prose description of what the agent does and
        when to reach for it.

    FILES
        ctl       Control commands (see COMMANDS).
        input     Write a prompt/query here to trigger the agent.
        OUTPUT    Blocking read; streams the agent's response.
        errors    Error log (read to see what went wrong).
        help      This file.
        <other>   ...one line per non-obvious file.

    COMMANDS
        <cmd1> [args]      One-line description.
        <cmd2> [args]      One-line description.
        ...

    EXAMPLES
        # Brief, concrete usage example
        echo "..." > /n/llm/<name>/input
        cat /n/llm/<name>/OUTPUT

        # Another example showing a less-obvious feature
        echo "<some ctl cmd>" > /n/llm/<name>/ctl

    NOTES
        Anything the user needs to know: required env vars, rate
        limits, side-effects, what state persists across calls, etc.

A helper `llmfs.meta_agent.HelpFile` is available — see the
skeleton below. Use it; don't roll your own.

================================================================
IMPORTS YOU CAN USE
================================================================

    from core.files import (
        SyntheticDir, SyntheticFile, StreamFile, QueueFile,
        CtlFile, CtlHandler,
    )
    from core.types import FidState
    from llmfs.meta_agent import HelpFile

================================================================
SKELETON — adapt this to the task
================================================================

```python
\"\"\"
<one-line description of the agent>
\"\"\"

import asyncio
from typing import Optional

from core.files import (
    SyntheticDir, SyntheticFile, StreamFile, QueueFile,
    CtlFile, CtlHandler,
)
from core.types import FidState
from llmfs.meta_agent import HelpFile


HELP_TEXT = \"\"\"\\
NAME
    {name} — short tagline goes here

SYNOPSIS
    echo "your input" > /n/llm/{name}/input
    cat /n/llm/{name}/OUTPUT

DESCRIPTION
    Describe what this agent does.

FILES
    ctl       Control commands (see COMMANDS).
    input     Write a query here to trigger the agent.
    OUTPUT    Blocking read; streams the response.
    errors    Error log.
    help      This file.

COMMANDS
    ping      Returns 'pong'. Useful for liveness checks.

EXAMPLES
    echo "hello" > /n/llm/{name}/input
    cat /n/llm/{name}/OUTPUT

NOTES
    Mention any required env vars, rate limits, etc.
\"\"\"


class MyCtlHandler(CtlHandler):
    def __init__(self, agent):
        self.agent = agent

    async def execute(self, command: str) -> Optional[str]:
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        if cmd == "ping":
            return "pong"
        raise ValueError(f"Unknown command: {cmd}")

    async def get_status(self) -> bytes:
        return b"ok\\n"


class MyInputFile(SyntheticFile):
    def __init__(self, agent):
        super().__init__("input")
        self.agent = agent
        self._buf: dict = {}

    async def read(self, fid, offset, count):
        return b""

    async def write(self, fid, offset, data):
        buf = self._buf.setdefault(id(fid), bytearray())
        buf.extend(data)
        return len(data)

    async def clunk(self, fid):
        buf = self._buf.pop(id(fid), None)
        if buf:
            text = bytes(buf).decode("utf-8", errors="replace").strip()
            if text:
                asyncio.create_task(self.agent.handle(text))


class MyAgent(SyntheticDir):
    def __init__(self, name: str):
        super().__init__(name)
        self.output = StreamFile("OUTPUT")
        self.errors = QueueFile("errors")
        self.add(CtlFile("ctl", MyCtlHandler(self)))
        self.add(MyInputFile(self))
        self.add(self.output)
        self.add(self.errors)
        # Render the agent name into the help text so paths match reality.
        self.add(HelpFile(HELP_TEXT.format(name=name)))

    async def handle(self, text: str):
        await self.output.reset()
        try:
            # ... real work goes here ...
            await self.output.append(f"received: {text}\\n".encode())
        except Exception as e:
            await self.errors.post(f"{type(e).__name__}: {e}\\n".encode())
        finally:
            await self.output.finish()


def create(name: str) -> SyntheticDir:
    return MyAgent(name)
```

================================================================
QUALITY BAR
================================================================

- Wrap I/O and external calls in try/except and post failures to
  `self.errors`. Never let an exception escape into the 9P layer.
- For long-running work, kick off `asyncio.create_task(...)` from
  `clunk` so the writer isn't blocked.
- Use `StreamFile.reset()` before each new generation and
  `StreamFile.finish()` in a finally clause so blocked readers wake up.
- ALWAYS include a `help` file built with `HelpFile(...)`. The text
  must follow the format above and use `{name}` placeholders so paths
  render correctly regardless of what the user mounts the agent as.
- Comment briefly where intent isn't obvious; skip ceremonial comments.
- One file, one ```python``` block. No commentary inside the block.
"""


# ---------------------------------------------------------------------------
# Meta-agent ctl handler
# ---------------------------------------------------------------------------

class MetaCtlHandler(AgentCtlHandler):
    """
    Extends the standard agent ctl with build/save/load/list commands.

    The base AgentCtlHandler already handles provider/model/system/temperature
    etc., so we only add the meta-specific verbs.
    """

    def __init__(self, agent: "MetaAgent"):
        super().__init__(agent)
        # AgentCtlHandler stores the agent as self.agent; we also keep
        # a more specific reference for type clarity.
        self.meta: MetaAgent = agent

    async def execute(self, command: str) -> Optional[str]:
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "build":
            if not arg:
                raise ValueError("Usage: build <agent_name>")
            name = arg.strip().split()[0]
            return await self.meta.build_from_last_output(name)

        if cmd == "load":
            if not arg:
                raise ValueError("Usage: load <agent_name>")
            name = arg.strip().split()[0]
            return await self.meta.load_saved(name)

        if cmd == "save":
            # Re-save the last generated module under a (possibly new) name
            # without instantiating the agent. Useful if you only want the
            # file on disk for later inspection.
            if not arg:
                raise ValueError("Usage: save <agent_name>")
            name = arg.strip().split()[0]
            return await self.meta.save_from_last_output(name)

        if cmd == "list":
            names = self.meta.list_saved()
            return " ".join(names) if names else "(none)"

        if cmd == "forget":
            if not arg:
                raise ValueError("Usage: forget <agent_name>")
            name = arg.strip().split()[0]
            return self.meta.forget_saved(name)

        # Fall through to base agent commands (provider, model, system, ...).
        return await super().execute(command)

    async def get_status(self) -> bytes:
        base = await super().get_status()
        saved = self.meta.list_saved()
        extra = (
            f"agents_dir {self.meta.agents_dir}\n"
            f"saved {' '.join(saved) if saved else '(none)'}\n"
        ).encode()
        return base + extra


# ---------------------------------------------------------------------------
# Meta-agent
# ---------------------------------------------------------------------------

class MetaAgent(Agent):
    """
    A text agent specialised for generating other agents.

    Differences from a vanilla Agent:
      - System prompt is pre-loaded with META_SYSTEM_PROMPT.
      - A plumbing rule is pre-installed that captures ```python ... ```
        fenced blocks into the supplementary output PYTHON.
      - Custom ctl handler adds build/save/load/list/forget.
      - Holds a back-reference to LLMFSRoot so it can mount built agents.
    """

    def __init__(
        self,
        name: str,
        provider: LLMProvider,
        fs_root=None,
        default_model: str = None,
    ):
        super().__init__(
            name=name,
            provider=provider,
            route_manager=None,
            default_model=default_model,
        )

        # Swap in our extended ctl handler. The base Agent already added a
        # CtlFile named "ctl" with the standard handler; replace that node.
        self.remove("ctl")
        self.add(CtlFile("ctl", MetaCtlHandler(self)))

        # Pre-load system prompt unless the user overrides it later.
        self.config.system = META_SYSTEM_PROMPT

        # Install a plumbing rule that catches python fences into PYTHON.
        # The Agent plumbing engine looks for a named group equal to the
        # output name AND uses a `code` group as the payload.
        python_pattern = r"```(?P<python>python)\s*\n(?P<code>.*?)```"
        self.create_supplementary_output("python")
        self.plumbing_rules.append({
            "pattern": python_pattern,
            "output_name": "python",
        })

        # Back-reference: set by LLMFSRoot.create_meta_agent so build/load
        # can mount the produced agent under the root.
        self._fs_root = fs_root

        self.agents_dir: Path = get_agents_dir()

    # ── Helpers ─────────────────────────────────────────────────────────

    def _last_generated_code(self) -> Optional[str]:
        """
        Return the most recent python block extracted by plumbing, or None.
        """
        sup = self.supplementary_outputs.get("python")
        if sup is None:
            return None

        # SupplementaryOutputFile stores its captured payloads in a queue
        # of blocks. Different implementations expose it differently; try
        # a few sensible attribute names so we don't break on a refactor.
        blocks = (
            getattr(sup, "blocks", None)
            or getattr(sup, "_blocks", None)
            or getattr(sup, "queue", None)
        )
        if blocks:
            try:
                # Take the most recent block; handle list or deque.
                last = list(blocks)[-1]
                if isinstance(last, (bytes, bytearray)):
                    return last.decode("utf-8", errors="replace")
                if isinstance(last, str):
                    return last
            except Exception:
                pass

        # Fall back to scanning the assistant's last message in history.
        for msg in reversed(self.history):
            if msg.role == "assistant":
                return _extract_python_block(msg.content)
        return None

    def list_saved(self) -> List[str]:
        """Names (without .py) of every persisted agent on disk."""
        try:
            return sorted(
                p.stem for p in self.agents_dir.glob("*.py")
                if not p.stem.startswith("_")
            )
        except Exception:
            return []

    def forget_saved(self, name: str) -> str:
        target = self.agents_dir / f"{name}.py"
        if not target.exists():
            return f"No saved agent named '{name}'"
        target.unlink()
        return f"Deleted {target}"

    # ── Build pipeline ──────────────────────────────────────────────────

    async def build_from_last_output(self, name: str) -> str:
        """
        Pull the last generated python module, persist it, import it, and
        mount the resulting agent under /n/llm/<name>.
        """
        code = self._last_generated_code()
        if not code:
            raise ValueError(
                "No python module found in the last response. "
                "Ask the meta-agent for a new one first."
            )

        # Persist before loading: if import fails the user can inspect.
        path = self._write_module(name, code)

        try:
            module = _import_from_path(name, path)
        except Exception as e:
            await self.errors.post(
                f"Failed to import {path}: {type(e).__name__}: {e}\n".encode()
            )
            raise

        if not hasattr(module, "create"):
            raise ValueError(
                f"{path.name} is missing a module-level `create(name)` factory"
            )

        # Mount under the LLMFS root.
        if self._fs_root is None:
            raise RuntimeError("Meta-agent has no _fs_root; cannot mount")

        try:
            instance = module.create(name)
        except Exception as e:
            await self.errors.post(
                f"create() raised: {type(e).__name__}: {e}\n".encode()
            )
            raise

        self._fs_root.mount_custom_agent(name, instance, source_path=path)
        return f"Built '{name}' from {path} and mounted at /<root>/{name}"

    async def save_from_last_output(self, name: str) -> str:
        """Persist the last generated module without instantiating it."""
        code = self._last_generated_code()
        if not code:
            raise ValueError("No python module found in the last response.")
        path = self._write_module(name, code)
        return f"Saved to {path}"

    async def load_saved(self, name: str) -> str:
        """
        Load a previously-saved agent (by name) from disk and mount it.
        This is what gets called at startup for every file in the agents dir.
        """
        path = self.agents_dir / f"{name}.py"
        if not path.exists():
            raise ValueError(f"No saved agent at {path}")

        if self._fs_root is None:
            raise RuntimeError("Meta-agent has no _fs_root; cannot mount")

        module = _import_from_path(name, path)
        if not hasattr(module, "create"):
            raise ValueError(
                f"{path.name} is missing a module-level `create(name)` factory"
            )

        instance = module.create(name)
        self._fs_root.mount_custom_agent(name, instance, source_path=path)
        return f"Loaded '{name}' from {path}"

    def _write_module(self, name: str, code: str) -> Path:
        if not name.isidentifier():
            raise ValueError(f"Invalid agent name '{name}' (must be a Python identifier)")
        path = self.agents_dir / f"{name}.py"
        # Prepend a small header so the file is self-documenting on disk.
        header = (
            f"# Auto-generated by llmfs meta-agent\n"
            f"# Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"# Agent name: {name}\n\n"
        )
        path.write_text(header + code, encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Module-loading utilities
# ---------------------------------------------------------------------------

def _import_from_path(name: str, path: Path):
    """
    Import a file as `llmfs._custom_<name>`. The leading underscore keeps
    the namespace tidy and avoids colliding with anything we ship.
    """
    mod_name = f"llmfs._custom_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module can import itself if it tries.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Don't leave a half-initialised module in sys.modules.
        sys.modules.pop(mod_name, None)
        raise
    return module


def load_saved_module(name: str, agents_dir: Optional[Path] = None):
    """
    Import a saved agent file and return (module, path).

    Used by LLMFSRoot at startup to re-load every persisted agent without
    needing a live MetaAgent instance.
    """
    if agents_dir is None:
        agents_dir = get_agents_dir()
    path = agents_dir / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(str(path))
    module = _import_from_path(name, path)
    if not hasattr(module, "create"):
        raise ValueError(
            f"{path.name} is missing a module-level `create(name)` factory"
        )
    return module, path


def list_saved_modules(agents_dir: Optional[Path] = None) -> List[str]:
    """Return names (without .py) of every persisted agent on disk."""
    if agents_dir is None:
        agents_dir = get_agents_dir()
    try:
        return sorted(
            p.stem for p in agents_dir.glob("*.py")
            if not p.stem.startswith("_")
        )
    except Exception:
        return []


def _extract_python_block(text: str) -> Optional[str]:
    """
    Naive fallback extractor for the most recent ```python``` block.
    Used when the supplementary output isn't populated for some reason.
    """
    import re
    m = list(re.finditer(r"```python\s*\n(.*?)```", text, re.DOTALL))
    if not m:
        return None
    return m[-1].group(1)
    