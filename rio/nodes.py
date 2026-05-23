"""
nodes.py — User-created node directories for the rio filesystem.
================================================================

Exposes /n/<workspace>/nodes/ as a directory of dynamically-created
node directories, plus a `ctl` file for creating/destroying them.

The operator (apps/operator.py) creates nodes by writing to nodes/ctl:

    echo 'new text foo'    > /n/<m>/nodes/ctl   → creates nodes/text_foo/
    echo 'new debug bar'   > /n/<m>/nodes/ctl   → creates nodes/debug_bar/
    echo 'new media baz'   > /n/<m>/nodes/ctl   → creates nodes/media_baz/
    echo 'new bash sh1'    > /n/<m>/nodes/ctl   → creates nodes/bash_sh1/
    echo 'new python py1'  > /n/<m>/nodes/ctl   → creates nodes/python_py1/
    echo 'delete text_foo' > /n/<m>/nodes/ctl   → removes nodes/text_foo/

Each kind has a predictable file set so the operator's NodeView
subclasses can subscribe to specific ports without further negotiation:

    text_<id>/   ctl, in, OUT
    debug_<id>/  ctl, in_0, in_1, ..., in_<N-1>   (default N=4)
    media_<id>/  ctl, in, OUT                     (binary passthrough)
    bash_<id>/   ctl, cmd, in, OUT, ERR
    python_<id>/ ctl, code, in, OUT, ERR

Files are bytes-stored:
  - Lowercase names (in, cmd, code, ctl) are plain read/write buffers.
  - Uppercase names (OUT, ERR) are blocking-read files (the llmfs
    convention: cat blocks until new data arrives, like STDERR).

The operator routes data through them via the existing /n/<m>/routes
infrastructure. For bash/python nodes the server is just a buffer
holder — the operator subscribes to `cmd`/`code` and `in`, runs the
command in its own process (see op_nodes.py: _run_bash_pipe /
_eval_python_pipe), and writes the result back to OUT/ERR. This keeps
the server free of subprocess management and resource limits while
still letting routes wire data into stdin / `input`.

Lifecycle:
  - NodesDir is added to RioRoot like TerminalsDir is.
  - On `new`, NodesDir constructs the right Node<Kind>Dir and adds it.
  - On `delete`, NodesDir removes the dir and the node's stored bytes
    are freed when its files are GC'd.

The whole module is one file because everything here is small. Roughly
350 lines for the full thing — comparable to TerminalDir + TerminalsDir
in your existing code.
"""

from __future__ import annotations

import asyncio
import re
from typing import Dict, List, Optional, Set

from core.files import (
    SyntheticDir, SyntheticFile, CtlFile, CtlHandler,
)
from core.types import FidState


# ─── Shared file primitives ────────────────────────────────────────────


class BufferFile(SyntheticFile):
    """A bytes buffer with blocking-on-rearm read semantics.

    Reads:
      - First read: returns current bytes immediately (or empty if
        never written).
      - After reader hits EOF: marks content consumed.
      - Next read at offset 0: BLOCKS until something writes the
        buffer. This is the push channel — operator processes that
        want change notifications open + read + close in a loop;
        each open blocks server-side until a writer arrives.

    Writes overwrite the buffer at offset 0 and splice at offset > 0,
    just like the previous version. Every write wakes any blocked
    readers.

    A shell `cat in` once still returns immediately and EOFs, because
    one cat does open → read-to-EOF → close. The blocking is only on
    a second consecutive open after the previous content was consumed.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._data: bytearray = bytearray()
        self._lock = asyncio.Lock()
        # Starts SET — first read returns immediately (with empty bytes
        # if never written, current bytes if written before the first
        # reader showed up).
        self._changed = asyncio.Event()
        self._changed.set()
        self._content_consumed = False

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        # If we're starting fresh at offset 0 and the previous content
        # was already drained, rearm and wait for the next write.
        if offset == 0 and self._content_consumed:
            async with self._lock:
                if self._content_consumed:
                    self._content_consumed = False
                    self._changed.clear()

        await self._changed.wait()

        async with self._lock:
            chunk = bytes(self._data[offset:offset + count])
            if offset + len(chunk) >= len(self._data):
                # Reader hit EOF — mark consumed so next read-at-0 rearms.
                self._content_consumed = True
            return chunk

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        async with self._lock:
            if offset == 0:
                self._data = bytearray(data)
            else:
                if offset + len(data) > len(self._data):
                    self._data.extend(
                        b"\0" * (offset + len(data) - len(self._data)))
                self._data[offset:offset + len(data)] = data
            # Wake any blocked readers.
            self._content_consumed = False
        self._changed.set()
        return len(data)

    async def get_bytes(self) -> bytes:
        """For local consumers (the kind ctl handlers) — snapshot copy."""
        async with self._lock:
            return bytes(self._data)

    async def set_bytes(self, data: bytes) -> None:
        async with self._lock:
            self._data = bytearray(data)
            self._content_consumed = False
        self._changed.set()


class BlockingOutputFile(SyntheticFile):
    """Uppercase-named blocking-read output file.

    Modeled on StderrFile in filesystem.py:
      1. WAITING: read() blocks until post()/mark_ready()
      2. READY: read() returns content
      3. CONSUMED: read() returns b"" (EOF — cat exits)
      4. Next read at offset 0: rearms, blocks at step 1 again

    This is what enables the `while true; do cat OUTPUT > somewhere`
    routing loops your routes infrastructure already uses.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._chunks: List[bytes] = []
        self._content_ready = asyncio.Event()
        self._content_consumed = False
        self._lock = asyncio.Lock()

    async def post(self, data: bytes) -> None:
        """Push new data and mark ready. Replaces previous unread
        content (last-generation-wins, like an agent's OUTPUT)."""
        async with self._lock:
            self._chunks = [data]
            self._content_consumed = False
        self._content_ready.set()

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        # Rearm on read-at-0 after previous content was fully consumed
        if offset == 0 and self._content_consumed:
            async with self._lock:
                if self._content_consumed:
                    self._content_consumed = False
                    self._content_ready.clear()
                    self._chunks.clear()

        await self._content_ready.wait()

        async with self._lock:
            content = b"".join(self._chunks)
            chunk = content[offset:offset + count]
            if offset + len(chunk) >= len(content):
                self._content_consumed = True
            return chunk

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        # Allow direct writes (e.g. for nodes that don't have an
        # executor — bytes get echoed through). The post() path is for
        # programmatic pushes; write() handles shell-side writes.
        await self.post(data)
        return len(data)


# ─── Per-kind node directories ─────────────────────────────────────────


class _NodeDirBase(SyntheticDir):
    """Common behavior for all kinds: holds a ctl file and a registry
    of named ports for the parent NodesDir to introspect."""

    kind: str = "generic"

    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.node_id = node_id
        # Port name → file instance (BufferFile or BlockingOutputFile)
        self.ports: Dict[str, SyntheticFile] = {}
        # ctl handler — subclasses may override execute() for kind-
        # specific commands like 'run', 'clear'.
        self.add(CtlFile("ctl", _NodeCtlHandler(self)))

    def _add_buffer(self, name: str) -> BufferFile:
        f = BufferFile(name)
        self.ports[name] = f
        self.add(f)
        return f

    def _add_output(self, name: str) -> BlockingOutputFile:
        f = BlockingOutputFile(name)
        self.ports[name] = f
        self.add(f)
        return f

    async def ctl_status(self) -> bytes:
        """Override in subclasses to expose node-specific status. By
        default, list the ports."""
        lines = [f"kind {self.kind}", f"id {self.node_id}"]
        lines.extend(f"port {n}" for n in self.ports)
        return ("\n".join(lines) + "\n").encode()

    async def ctl_command(self, cmd: str, arg: str) -> Optional[str]:
        """Override in subclasses to handle node-specific ctl writes.
        Return a status string or raise ValueError on bad command."""
        if cmd == "clear":
            for f in self.ports.values():
                if isinstance(f, BufferFile):
                    await f.set_bytes(b"")
            return "cleared"
        raise ValueError(f"unknown command: {cmd}")


class _NodeCtlHandler(CtlHandler):
    """Shared ctl handler for all node kinds. Delegates to the node's
    ctl_command() / ctl_status() methods, so each kind can extend
    without us multiplying handler classes."""

    def __init__(self, node: _NodeDirBase):
        self.node = node

    async def execute(self, command: str) -> Optional[str]:
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1] if len(parts) > 1 else ""
        if not cmd:
            return None
        return await self.node.ctl_command(cmd, arg)

    async def get_status(self) -> bytes:
        return await self.node.ctl_status()


class TextNodeDir(_NodeDirBase):
    """A text node: stores bytes in `in`, mirrors them to `OUT` on write."""
    kind = "text"

    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.in_file = self._add_buffer("in")
        self.out_file = self._add_output("OUT")
        # Mirror in → OUT on write. We hook the in_file's write by
        # subclassing or by wrapping; simplest is to override write.
        self._wrap_in_write()

    def _wrap_in_write(self):
        original_write = self.in_file.write
        out_file = self.out_file
        async def write_and_mirror(fid, offset, data):
            n = await original_write(fid, offset, data)
            # Push the new full buffer through OUT, so subscribers see
            # the latest text in one shot.
            await out_file.post(bytes(await self.in_file.get_bytes()))
            return n
        self.in_file.write = write_and_mirror


class DebugNodeDir(_NodeDirBase):
    """A debug log sink: N input ports, no outputs. Default N=4."""
    kind = "debug"

    def __init__(self, node_id: str, n_inputs: int = 4):
        super().__init__(node_id)
        for i in range(n_inputs):
            self._add_buffer(f"in_{i}")


class MediaNodeDir(_NodeDirBase):
    """A media preview node: binary passthrough from `in` to `OUT`."""
    kind = "media"

    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.in_file = self._add_buffer("in")
        self.out_file = self._add_output("OUT")
        self._wrap_passthrough()

    def _wrap_passthrough(self):
        original_write = self.in_file.write
        out_file = self.out_file
        in_file = self.in_file
        async def write_and_passthrough(fid, offset, data):
            n = await original_write(fid, offset, data)
            await out_file.post(bytes(await in_file.get_bytes()))
            return n
        self.in_file.write = write_and_passthrough


class BashNodeDir(_NodeDirBase):
    """Bash command node. Ports:
        cmd  — command text (also overrides the operator's editor when written)
        in   — bytes to be piped to the subprocess's stdin
        OUT  — subprocess stdout
        ERR  — subprocess stderr

    Execution lives client-side in the operator (op_nodes.py:
    _run_bash_pipe). The server's job here is just to hold buffers and
    let routes flow data through them — writing to `cmd` or `in` is a
    no-op as far as the server is concerned, the operator subscribes
    and runs the command itself.
    """
    kind = "bash"

    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.cmd_file = self._add_buffer("cmd")
        # `in` is the stdin port. Lowercase basename → POLL mode on the
        # operator side, which is what we want for bytes-based input.
        # Mirrors TextNodeDir's `in` buffer.
        self.in_file = self._add_buffer("in")
        self.out_file = self._add_output("OUT")
        self.err_file = self._add_output("ERR")


class PythonNodeDir(_NodeDirBase):
    """Python expression node. Ports:
        code — Python source (also overrides the operator's editor)
        in   — bytes bound to `input` in the eval namespace
        OUT  — expression result (bytes/str/repr)
        ERR  — traceback / error text

    Evaluation lives client-side (op_nodes.py: _eval_python_pipe). The
    server just holds buffers — writes to `code` or `in` are stored
    and made available for subscribers; the operator picks them up
    and evaluates.
    """
    kind = "python"

    def __init__(self, node_id: str):
        super().__init__(node_id)
        self.code_file = self._add_buffer("code")
        self.in_file = self._add_buffer("in")
        self.out_file = self._add_output("OUT")
        self.err_file = self._add_output("ERR")


# ─── NodesDir + ctl ────────────────────────────────────────────────────


_KIND_CLASSES = {
    "text":   TextNodeDir,
    "debug":  DebugNodeDir,
    "media":  MediaNodeDir,
    "bash":   BashNodeDir,
    "python": PythonNodeDir,
}

# Valid node-id pattern. Restrictive to keep filesystem semantics sane.
_VALID_ID_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")


class NodesDir(SyntheticDir):
    """The /n/<workspace>/nodes/ directory.

    Holds a ctl file and a registry of user-created node directories.
    Mirrors TerminalsDir's pattern: a container that handles dynamic
    creation/removal via its ctl.

    Optionally takes a `routes_manager` so node deletion can also sweep
    any routes whose endpoints live inside the deleted node's
    directory. This is defense in depth — the operator UI also clears
    these routes before sending `delete <id>` — but it covers the case
    where a script or non-UI client deletes a node directly. Without
    the sweep, RoutesManager.attachments retains orphan Plan9Attachment
    instances trying to cat now-vanished port files.
    """

    def __init__(self, routes_manager: Optional[object] = None):
        super().__init__("nodes")
        self._nodes: Dict[str, _NodeDirBase] = {}
        self._routes_manager = routes_manager
        self.add(CtlFile("ctl", NodesCtlHandler(self)))

    def list_node_ids(self) -> List[str]:
        return sorted(self._nodes.keys())

    def get_node(self, node_id: str) -> Optional[_NodeDirBase]:
        return self._nodes.get(node_id)

    def create_node(self, kind: str, node_id: str) -> _NodeDirBase:
        """Create a new node of the given kind. Raises ValueError on
        bad kind, duplicate id, or invalid id."""
        if kind not in _KIND_CLASSES:
            raise ValueError(
                f"unknown kind {kind!r}; "
                f"valid: {sorted(_KIND_CLASSES)}")
        if not _VALID_ID_RE.match(node_id):
            raise ValueError(
                f"invalid id {node_id!r}; "
                f"must match [A-Za-z0-9_][A-Za-z0-9_.-]{{0,63}}")
        if node_id in self._nodes:
            raise ValueError(f"node {node_id!r} already exists")
        cls = _KIND_CLASSES[kind]
        node = cls(node_id)
        self._nodes[node_id] = node
        self.add(node)
        return node

    def remove_node(self, node_id: str) -> bool:
        """Remove a node by id. Returns True if removed, False if not
        found.

        Also sweeps the routes manager (if wired in) for any routes
        whose source or destination references a path under this
        node's directory. Without that sweep, deleting a wired node
        leaves orphan Plan9Attachment loops in RoutesManager that
        keep trying to cat vanished port files until they error out.
        """
        if node_id not in self._nodes:
            return False
        node = self._nodes[node_id]

        # Sweep dependent routes BEFORE the directory is removed so the
        # path-prefix match is unambiguous. The route file's listener
        # will fire `_notify('remove', ...)` for each one, which the
        # operator picks up to drop its UI connections.
        self._sweep_routes_for_node(node)

        del self._nodes[node_id]
        self.remove(node_id)
        return True

    def _sweep_routes_for_node(self, node: "_NodeDirBase") -> None:
        """Remove any route whose source or destination falls inside
        `node`'s directory. No-op if no routes_manager was wired in."""
        rm = self._routes_manager
        if rm is None:
            return
        # We don't have a stable absolute path on the node (the synthetic
        # FS doesn't expose one), so match by the suffix
        # `/nodes/<node_id>/`. This is correct because all routes are
        # written with absolute mount paths and node ids are unique
        # within the nodes directory.
        suffix = f"/nodes/{node.node_id}/"
        # list_routes() returns (src, dst, running) tuples — we only
        # need the sources, since remove_route() is keyed by source.
        stale_sources = [
            src for (src, dst, _running) in rm.list_routes()
            if suffix in src or suffix in dst
        ]
        for src in stale_sources:
            try:
                rm.remove_route(src)
            except Exception:
                # Best-effort cleanup; don't fail the delete if a
                # route's Plan9Attachment chokes on its own teardown.
                pass


class NodesCtlHandler(CtlHandler):
    """Handles ctl commands for the nodes directory.

    Commands:
        new <kind> <id>     Create a new node of the given kind.
        delete <id>         Remove an existing node by id.
        list                Read-only: emitted via get_status().
    """

    def __init__(self, nodes_dir: NodesDir):
        self.nodes_dir = nodes_dir

    async def execute(self, command: str) -> Optional[str]:
        parts = command.split()
        if not parts:
            return None
        cmd = parts[0].lower()

        if cmd == "new":
            if len(parts) < 3:
                raise ValueError("Usage: new <kind> <id>")
            kind = parts[1].lower()
            node_id = parts[2]
            self.nodes_dir.create_node(kind, node_id)
            return f"created {kind} {node_id}"

        if cmd == "delete":
            if len(parts) != 2:
                raise ValueError("Usage: delete <id>")
            node_id = parts[1]
            if self.nodes_dir.remove_node(node_id):
                return f"deleted {node_id}"
            raise ValueError(f"no such node: {node_id}")

        raise ValueError(f"unknown command: {cmd}")

    async def get_status(self) -> bytes:
        ids = self.nodes_dir.list_node_ids()
        if not ids:
            return b"(no nodes)\n"
        lines = []
        for nid in ids:
            node = self.nodes_dir.get_node(nid)
            kind = node.kind if node else "?"
            lines.append(f"{kind} {nid}")
        return ("\n".join(lines) + "\n").encode()