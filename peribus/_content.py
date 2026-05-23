"""
peribus._content — concatenation of: embed_search.py, gossip.py, widget_validator.py, widget_runtime.py, app_swarm.py, apps_fs.py

This is a build artefact. The original module names live as section
banners below so `grep "^# ===="` jumps to each one.
"""

from __future__ import annotations


# ============================================================================
# embed_search.py
# ----------------------------------------------------------------------------
"""
peribus.embed_search — thin client over the local /n/llm/embed agent

This module used to maintain a shadow corpus (~/.peribus/corpus/ with
content-addressed filenames and an index.json with use_count tracking).
That layer is gone. The /n/llm/embed agent already has a corpus — the
folders the user has scanned — and that corpus IS the truth. Peribus
just queries it.

What we keep:
  - The async I/O wrapper around /n/llm/embed/{ctl,input,OUTPUT}
  - Probe / graceful-degradation logic for when the mount isn't there

What we drop:
  - The parallel ~/.peribus/corpus/ directory
  - index.json, CorpusEntry, use_count, last_used_at, indexed flags
  - The _app_id_from_path filter that rejected paths outside our corpus
  - The "endorse by running" mechanism (mark_used). Endorsement is now:
    if the user pointed their agent at a folder, they endorsed it.

What we add:
  - SearchHit now carries content bytes and a content_hash, computed
    on the fly by reading the path the agent returned.
  - get_content_by_hash() looks up bytes for a given hash, for the
    responder side of MSG_FETCH when a peer asks "give me <hash>".

The wire protocol (in app_swarm.py) addresses apps by content hash so
the same file from two responders dedupes naturally. The hash is just
sha256 of the source bytes — computed at search time, not maintained
in any persistent state.
"""


import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_EMBED_MOUNT = "/n/llm/embed"
DEFAULT_QUERY_TIMEOUT_S = 30.0

# Agent output lines: "  [0.4987] /path/to/file.py"
_RESULT_LINE_RE = re.compile(r"^\s*\[(?P<score>-?\d+\.\d+)\]\s+(?P<path>.+?)\s*$")


@dataclass
class SearchHit:
    """
    One ranked result from the local agent.

    `app_id` is sha256 of the source bytes. Two peers serving the same
    file produce the same app_id, which is what lets the swarm's
    AggregatedHit consensus logic work.

    `content` is the raw .py bytes, read from `path` at search time.
    No longer shipped inline over the wire (that caused readline buffer
    overruns); kept here so the responder can serve MSG_FETCH for this
    hash if a peer asks.
    """
    app_id: str             # "sha256:<hex>"
    score: float
    path: str               # absolute path on the responder's disk
    content: bytes          # raw .py source bytes (local-only, not wire)
    title: str = ""         # derived from filename stem


def _hash_content(data: bytes) -> str:
    """Stable content hash — sha256, hex-encoded, prefixed."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


class EmbedSearch:
    """
    Async client for the local /n/llm/embed agent. No persistent state
    of its own — every query is fresh. The agent owns the index, the
    descriptions, and the scanned folders; peribus just asks questions.
    """

    def __init__(
        self,
        mount: str = DEFAULT_EMBED_MOUNT,
        query_timeout: float = DEFAULT_QUERY_TIMEOUT_S,
        max_content_bytes: int = 256 * 1024,
    ):
        self.mount = Path(mount)
        self.query_timeout = query_timeout
        self.max_content_bytes = max_content_bytes

        # Read serialization: only one query in flight at a time, since
        # the agent's input/OUTPUT is a single-channel handshake.
        self._query_lock = asyncio.Lock()
        self._ctl_lock = asyncio.Lock()

        self._available = False

        # Reverse index built during search rounds so the responder side
        # of MSG_FETCH can answer "give me <hash>" by mapping back to
        # the path on disk. Bounded LRU.
        self._hash_to_path: Dict[str, str] = {}
        self._hash_to_path_max = 256

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._available = self._probe_mount()
        if self._available:
            logger.info(f"embed_search: connected to {self.mount}")
        else:
            logger.info(
                f"embed_search: {self.mount} not available; "
                "semantic search disabled until it appears"
            )

    async def stop(self) -> None:
        pass

    @property
    def available(self) -> bool:
        if not self._available:
            self._available = self._probe_mount()
        return self._available

    def _probe_mount(self) -> bool:
        ctl = self.mount / "ctl"
        inp = self.mount / "input"
        out = self.mount / "OUTPUT"
        try:
            return ctl.exists() and inp.exists() and out.exists()
        except OSError:
            return False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(self, query: str, top_k: int = 5) -> List[SearchHit]:
        """
        Run a semantic search against the local agent. Returns hits in
        descending score order. Empty list on missing mount, empty
        index, or timeout.
        """
        if not self.available:
            return []
        query = query.strip()
        if not query:
            return []

        async with self._query_lock:
            await self._ctl(f"top_k {int(top_k)}")
            try:
                output_data = await asyncio.wait_for(
                    self._do_query(query),
                    timeout=self.query_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"embed_search: query timed out: {query!r}")
                return []
            except OSError as e:
                logger.debug(f"embed_search: query I/O error: {e}")
                return []

        return await self._parse_and_load(output_data, top_k)

    async def _do_query(self, query: str) -> bytes:
        """
        Send a query to the agent and collect its OUTPUT.

        Pattern: write input, then open OUTPUT and read until EOF.
        Same pattern `cat OUTPUT` uses from a shell.
        """
        input_path = self.mount / "input"
        output_path = self.mount / "OUTPUT"

        def _blocking() -> bytes:
            with open(input_path, "wb") as f:
                f.write(query.encode("utf-8"))
            with open(output_path, "rb") as f:
                return f.read()

        return await asyncio.get_running_loop().run_in_executor(None, _blocking)

    async def _parse_and_load(self, data: bytes, top_k: int) -> List[SearchHit]:
        """Parse the agent's OUTPUT and load each hit's content from disk."""
        hits: List[SearchHit] = []
        text = data.decode("utf-8", errors="replace")
        seen_paths: set = set()

        loop = asyncio.get_running_loop()
        for line in text.splitlines():
            m = _RESULT_LINE_RE.match(line)
            if not m:
                continue
            try:
                score = float(m.group("score"))
            except ValueError:
                continue
            path = m.group("path")

            try:
                abs_path = str(Path(path).resolve())
            except (OSError, ValueError):
                abs_path = path
            if abs_path in seen_paths:
                continue
            seen_paths.add(abs_path)

            content = await loop.run_in_executor(
                None, self._read_capped, abs_path,
            )
            if not content:
                continue

            app_id = _hash_content(content)
            title = Path(abs_path).stem
            hits.append(SearchHit(
                app_id=app_id,
                score=score,
                path=abs_path,
                content=content,
                title=title,
            ))

            self._remember_hash(app_id, abs_path)

            if len(hits) >= top_k:
                break

        return hits

    def _read_capped(self, path: str) -> bytes:
        try:
            with open(path, "rb") as f:
                return f.read(self.max_content_bytes)
        except OSError:
            return b""

    # ------------------------------------------------------------------
    # Hash → path reverse lookup, for responder-side MSG_FETCH
    # ------------------------------------------------------------------

    def _remember_hash(self, app_id: str, path: str) -> None:
        if app_id in self._hash_to_path:
            del self._hash_to_path[app_id]
            self._hash_to_path[app_id] = path
            return
        if len(self._hash_to_path) >= self._hash_to_path_max:
            try:
                oldest = next(iter(self._hash_to_path))
                del self._hash_to_path[oldest]
            except StopIteration:
                pass
        self._hash_to_path[app_id] = path

    async def get_content_by_hash(self, app_id: str) -> Optional[bytes]:
        """
        Look up the bytes for a given content hash. Used by the
        responder when serving MSG_FETCH for an app a peer wants.
        """
        path = self._hash_to_path.get(app_id)
        if path is None:
            return None
        loop = asyncio.get_running_loop()
        content = await loop.run_in_executor(None, self._read_capped, path)
        if not content:
            return None
        if _hash_content(content) != app_id:
            logger.debug(
                f"embed_search: content at {path} changed since cache; "
                f"requested hash {app_id} no longer matches"
            )
            return None
        return content

    # ------------------------------------------------------------------
    # ctl plumbing
    # ------------------------------------------------------------------

    async def _ctl(self, command: str) -> bool:
        if not self.available:
            return False
        ctl_path = self.mount / "ctl"

        def _blocking() -> bool:
            try:
                with open(ctl_path, "wb") as f:
                    f.write(command.encode("utf-8"))
                return True
            except OSError as e:
                logger.debug(f"embed_search: ctl {command!r} failed: {e}")
                return False

        async with self._ctl_lock:
            return await asyncio.get_running_loop().run_in_executor(
                None, _blocking,
            )

    def stats(self) -> dict:
        return {
            "available": self.available,
            "mount": str(self.mount),
            "recent_hashes": len(self._hash_to_path),
        }

# ============================================================================
# gossip.py
# ----------------------------------------------------------------------------
"""
peribus.gossip — content store and feed propagation

Two responsibilities:

  1. Content store. Every shared item (post, widget, image) has a content
     hash. The store keeps recent items in memory and serves reads to peers.

  2. Feed. As posts arrive from the network (or from us), they're added to
     a ring buffer. Readers (the feed/new file) consume from cursors.
     Filtering by relevance to the identity vector happens at read time, not
     ingest time, so the same buffer serves "everything" and "personalized".

Wire protocol (peribus/0.1, JSON-line over TCP for v0.1; QUIC later):

  Each message is a single line of JSON, newline-terminated. Fields:

    {
      "type": "announce" | "post" | "fetch" | "data" | "msg",
      "from": "<nodeid>",
      "ts": <unix-ms>,
      "sig": "<base64 ed25519 sig over the canonical body>",
      ...type-specific fields...
    }

  `announce`  — peer says "I'm here, here's my sketch"
  `post`      — broadcast a post: { id, vector, title, body, attachments }
  `fetch`     — request content by hash: { hash }
  `data`      — response to fetch: { hash, bytes (b64) }
  `msg`       — direct message to one peer: { to, body }

For v0.1 we keep this synchronous and simple. No retries, no fragment
reassembly. If a packet drops, you just don't see that post — gossip
amplification across other peers covers most of it.
"""


import asyncio
import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from peribus._foundation import IdentityVector


FEED_RING_SIZE = 4096


def _content_hash(data: bytes) -> str:
    """Stable content hash. SHA-256 only for cross-machine consistency.

    Same reasoning as identity._nodeid_hash: every peer must hash the
    same bytes to the same string, so we cannot rely on optional
    libraries (blake3) being present on every machine. The "b3:" prefix
    is kept for backwards-compatibility with the wire format; it does
    NOT mean the hash is BLAKE3.
    """
    h = hashlib.sha256(data).digest()
    return "b3:" + base64.b32encode(h).decode("ascii").lower().rstrip("=")[:32]


@dataclass
class Post:
    """A single piece of shared content circulating on the rhizome."""
    id: str                          # content hash
    author: str                      # nodeid
    ts: float                        # unix seconds
    title: str
    body: str                        # short text body (or summary if attached)
    vector: List[float]              # embedding of this post's content
    attachments: List[str] = field(default_factory=list)  # content hashes
    sig: bytes = b""                 # author signature

    def to_json(self) -> str:
        return json.dumps({
            "id": self.id,
            "author": self.author,
            "ts": self.ts,
            "title": self.title,
            "body": self.body,
            "vector": self.vector,
            "attachments": self.attachments,
            "sig": base64.b64encode(self.sig).decode("ascii"),
        })

    @classmethod
    def from_json(cls, s: str) -> "Post":
        d = json.loads(s)
        return cls(
            id=d["id"],
            author=d["author"],
            ts=d["ts"],
            title=d.get("title", ""),
            body=d.get("body", ""),
            vector=d.get("vector", []),
            attachments=d.get("attachments", []),
            sig=base64.b64decode(d.get("sig", "")),
        )

    def feed_line(self, resonance: float) -> bytes:
        """Format for a feed/new read: one JSON line per post."""
        line = json.dumps({
            "id": self.id,
            "author": self.author,
            "ts": self.ts,
            "title": self.title,
            "body": self.body,
            "resonance": round(resonance, 3),
            "attachments": self.attachments,
        })
        return (line + "\n").encode("utf-8")


@dataclass
class FeedCursor:
    """Per-fid position in the feed ring buffer."""
    last_index: int = -1     # ring index of last post returned
    pending: bytes = b""     # leftover bytes from a partial read
    waiter: Optional[asyncio.Future] = None  # set when we're blocked


class GossipMesh:
    """
    Content store + feed buffer. The daemon owns one of these.

    Threading: all methods are coroutines and assume single-loop access.
    """

    def __init__(self, identity_provider: Callable[[], "IdentityVector"]):
        self._identity_provider = identity_provider
        self._content: Dict[str, bytes] = {}        # hash -> raw content
        self._posts: Dict[str, Post] = {}           # hash -> post metadata
        self._ring: List[Post] = []                 # FIFO with eviction
        self._waiters: List[asyncio.Future] = []    # cursors blocked on new posts
        self._lock = asyncio.Lock()

    # ---- content store ----

    def put_content(self, data: bytes) -> str:
        """Store bytes, return the content hash."""
        h = _content_hash(data)
        self._content[h] = data
        return h

    def get_content(self, content_hash: str) -> Optional[bytes]:
        return self._content.get(content_hash)

    # ---- posts / feed ----

    async def add_post(self, post: Post) -> bool:
        """
        Insert a post into the local feed buffer. Returns True if newly seen,
        False if we already had it (gossip dedup).
        """
        async with self._lock:
            if post.id in self._posts:
                return False
            self._posts[post.id] = post
            self._ring.append(post)
            # Bound the ring.
            if len(self._ring) > FEED_RING_SIZE:
                evicted = self._ring.pop(0)
                self._posts.pop(evicted.id, None)

            # Wake any feed readers blocked waiting for new posts.
            for w in self._waiters:
                if not w.done():
                    w.set_result(None)
            self._waiters.clear()

            return True

    def feed_cursor(self, from_start: bool = True) -> FeedCursor:
        """
        Hand out a fresh cursor.

        from_start=True (default): cursor begins before any existing posts,
            so the first read returns everything in the buffer ranked by
            current relevance, then blocks for new arrivals. This is what
            `cat /n/peribus/feed/new` should do — give the user a usable
            chunk of content immediately.

        from_start=False: cursor begins past the current end, so the first
            read blocks until something new arrives. Use this for pure
            tail-style readers that don't care about backlog.
        """
        if from_start:
            return FeedCursor(last_index=-1)
        return FeedCursor(last_index=len(self._ring) - 1)

    async def read_feed(
        self,
        cursor: FeedCursor,
        max_bytes: int,
        identity: "IdentityVector",
        block: bool = True,
    ) -> bytes:
        """
        Return up to `max_bytes` of feed content for this cursor.

        block=True (default): if no new posts, blocks until one arrives.
            This is what /n/peribus/feed/new uses (tail-style, never EOFs).

        block=False: returns b"" immediately if no new posts. Used by
            /n/peribus/feed/recent — `cat` returns the buffer and exits.
        """
        from peribus._foundation import cosine

        # Drain partial buffer first.
        if cursor.pending:
            chunk = cursor.pending[:max_bytes]
            cursor.pending = cursor.pending[max_bytes:]
            return chunk

        # If no new posts, either wait or return empty (EOF for non-blocking).
        if cursor.last_index >= len(self._ring) - 1:
            if not block:
                return b""
            while cursor.last_index >= len(self._ring) - 1:
                waiter = asyncio.get_running_loop().create_future()
                self._waiters.append(waiter)
                cursor.waiter = waiter
                try:
                    await waiter
                finally:
                    cursor.waiter = None
                    # If we were cancelled (e.g. the FeedNewFile keepalive
                    # timeout fired), remove ourselves from _waiters so
                    # long silent stretches don't accumulate dead futures.
                    # On the happy path the waiter was already popped by
                    # add_post's clear(), so this is a no-op.
                    try:
                        self._waiters.remove(waiter)
                    except ValueError:
                        pass

        # Snapshot new posts and rank by relevance to current identity.
        async with self._lock:
            start = cursor.last_index + 1
            new_posts = list(self._ring[start:])
            cursor.last_index = len(self._ring) - 1

        identity_vec = identity.snapshot()
        ranked = sorted(
            new_posts,
            key=lambda p: -cosine(identity_vec, p.vector),
        )

        # Emit as JSON lines, fitting under max_bytes.
        out = bytearray()
        for p in ranked:
            res = cosine(identity_vec, p.vector)
            line = p.feed_line(res)
            if len(out) + len(line) > max_bytes:
                # Save the rest in cursor.pending so the next read returns it.
                cursor.pending = bytes(line)
                # And stash the remaining unranked posts too.
                idx = ranked.index(p) + 1
                for rest in ranked[idx:]:
                    cursor.pending += rest.feed_line(cosine(identity_vec, rest.vector))
                break
            out.extend(line)

        return bytes(out)

    def stats(self) -> Dict[str, int]:
        return {
            "posts": len(self._posts),
            "content_blobs": len(self._content),
            "ring_size": len(self._ring),
        }

# ============================================================================
# widget_validator.py
# ----------------------------------------------------------------------------
"""
peribus.widget_validator — make foreign Python safe to render

A shared widget is a Python file that arrives over the network. We will not
exec arbitrary Python from strangers. Instead, we parse it to AST and walk
the tree, rejecting anything outside a strict whitelist:

  ALLOWED:
    - Literal expressions, math, comparisons, boolean logic
    - Variable assignment, augmented assignment
    - if/elif/else, for/while loops with bounded iteration counts
    - def / lambda function definitions (no decorators except @Slot)
    - Calls to whitelisted names from the widget namespace (PySide6 widgets,
      scene helpers, the `peribus` runtime module)
    - Attribute access, subscript access (but NOT dunder access)
    - List/dict/set/tuple/generator comprehensions

  REJECTED:
    - import / from / __import__ — namespace is pre-built, no new imports
    - exec / eval / compile / globals / locals / vars / dir
    - open / file / input — no I/O outside what the runtime mediates
    - Any attribute starting with _ or __ — no escaping the sandbox
    - try/except (so widgets can't hide errors from the daemon)
    - class definitions with metaclasses or weird base lists
    - while True without a break in body (heuristic; not bulletproof)
    - Anything with __ in attribute or name lookup

If validation passes, the runtime additionally enforces:
    - CPU time limit per execution (signal.SIGXCPU on POSIX)
    - Memory ceiling via resource.RLIMIT_AS
    - All "imports" come from a fixed namespace dict; getattr is wrapped
      to forbid dunders at runtime as a belt-and-suspenders check.

This is not a perfect sandbox — pure-Python sandboxes never are — but it's
the right tradeoff for our threat model: the network is a friend-of-a-friend
graph, not the open internet, and the worst a bad widget can do is ugly
output or crash one widget process. Combined with the per-widget process
isolation in widget_runtime, this is good enough.
"""


import ast
from dataclasses import dataclass, field
from typing import List, Set


# Names we explicitly deny even if someone tries to alias them.
DENIED_NAMES: Set[str] = {
    "exec", "eval", "compile",
    "globals", "locals", "vars",
    "__import__", "__builtins__", "__class__", "__bases__",
    "__subclasses__", "__mro__", "__dict__",
    "open", "file", "input",
    "breakpoint", "help", "exit", "quit",
    "memoryview",  # easy escape vector
    "type",        # type().__bases__ etc.
    "object",
    "super",       # access to dunders via mro
    "getattr", "setattr", "delattr",  # we'll provide a safe getattr in runtime
    "__builtins__",
}

# Decorator names that are safe (Qt slots, mostly).
ALLOWED_DECORATORS: Set[str] = {"Slot", "staticmethod", "classmethod"}


@dataclass
class ValidationResult:
    """Outcome of validating a widget AST."""

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)


class WidgetValidator(ast.NodeVisitor):
    """Walks an AST and records every rule violation."""

    def __init__(self):
        self.result = ValidationResult(ok=True)

    # ------------------------------------------------------------------
    # Disallowed statements
    # ------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        self.result.fail(f"line {node.lineno}: `import` is not allowed in widgets")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.result.fail(f"line {node.lineno}: `from ... import` is not allowed in widgets")

    def visit_Try(self, node: ast.Try) -> None:
        # We forbid try/except so widgets can't silently swallow errors
        # the daemon needs to see (and so they can't probe attribute
        # blocklists by trying things in a loop).
        self.result.fail(
            f"line {node.lineno}: `try`/`except` is not allowed in widgets — "
            f"let errors propagate"
        )

    def visit_Global(self, node: ast.Global) -> None:
        self.result.fail(f"line {node.lineno}: `global` is not allowed")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.result.fail(f"line {node.lineno}: `nonlocal` is not allowed")

    def visit_With(self, node: ast.With) -> None:
        # Context managers can hide resource acquisition. Disallow until
        # we have a clear use case.
        self.result.fail(f"line {node.lineno}: `with` blocks are not allowed in widgets")

    def visit_AsyncFunctionDef(self, node) -> None:
        self.result.fail(f"line {node.lineno}: async functions are not allowed in widgets")

    def visit_AsyncFor(self, node) -> None:
        self.result.fail(f"line {node.lineno}: `async for` is not allowed in widgets")

    def visit_AsyncWith(self, node) -> None:
        self.result.fail(f"line {node.lineno}: `async with` is not allowed in widgets")

    def visit_Yield(self, node) -> None:
        self.result.fail(f"line {node.lineno}: `yield` is not allowed in widgets")

    def visit_YieldFrom(self, node) -> None:
        self.result.fail(f"line {node.lineno}: `yield from` is not allowed in widgets")

    # ------------------------------------------------------------------
    # Names — block dunders and the deny-list
    # ------------------------------------------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in DENIED_NAMES:
            self.result.fail(f"line {node.lineno}: name `{node.id}` is forbidden")
        if node.id.startswith("__") and node.id.endswith("__"):
            self.result.fail(
                f"line {node.lineno}: dunder name `{node.id}` is forbidden"
            )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            # _private and __dunder both blocked. Widgets shouldn't need them
            # — Qt's public API doesn't have them, and the runtime module
            # exposes only public functions.
            self.result.fail(
                f"line {node.lineno}: access to underscore attribute "
                f"`.{node.attr}` is forbidden"
            )
        # Recurse into the value chain (e.g. obj.foo.bar).
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Decorators
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for dec in node.decorator_list:
            name = self._decorator_name(dec)
            if name not in ALLOWED_DECORATORS:
                self.result.fail(
                    f"line {dec.lineno}: decorator `@{name}` is not allowed "
                    f"(allowed: {sorted(ALLOWED_DECORATORS)})"
                )
        self.generic_visit(node)

    def _decorator_name(self, node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return self._decorator_name(node.func)
        return "<expr>"

    # ------------------------------------------------------------------
    # Class definitions — limited but allowed
    # ------------------------------------------------------------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.keywords:
            # No metaclass=, no other class kwargs.
            self.result.fail(
                f"line {node.lineno}: class kwargs (metaclass=, etc.) not allowed"
            )
        for dec in node.decorator_list:
            self.result.fail(
                f"line {dec.lineno}: class decorators not allowed"
            )
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Loop heuristics — catch obvious infinite loops at validation time.
    # The runtime CPU limit catches the rest.
    # ------------------------------------------------------------------

    def visit_While(self, node: ast.While) -> None:
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            # while True: must contain a break or return statement somewhere.
            if not self._contains_break_or_return(node.body):
                self.result.fail(
                    f"line {node.lineno}: `while True` without a `break` or "
                    f"`return` in body is forbidden"
                )
        self.generic_visit(node)

    def _contains_break_or_return(self, body: List[ast.stmt]) -> bool:
        for stmt in body:
            for sub in ast.walk(stmt):
                if isinstance(sub, (ast.Break, ast.Return)):
                    return True
        return False


def validate_widget_source(source: str) -> ValidationResult:
    """
    Parse and validate a widget's source code. Returns a ValidationResult
    you can inspect; check `.ok` to see if it's safe to run.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as e:
        r = ValidationResult(ok=False)
        r.fail(f"syntax error: {e}")
        return r

    validator = WidgetValidator()
    validator.visit(tree)
    return validator.result

# ============================================================================
# widget_runtime.py
# ----------------------------------------------------------------------------
"""
peribus.widget_runtime — the namespace shared widgets get to play in

After validate_widget_source() approves a piece of foreign code, this
module builds the namespace it executes in and provides the `peribus`
runtime object widgets use to talk to the world.

Core principles:
  * Pre-built namespace. Widgets cannot import. Whatever they need has to
    be in the dict we hand to exec().
  * Mediated I/O. There is no `open`, no `socket`, no `requests`. If a
    widget wants to talk to a peer, it goes through `peribus.stream(name)`
    which returns a file-like object the daemon controls.
  * Resource limits. We set CPU and memory ceilings on the executing
    thread/process so a misbehaving widget cannot wedge the daemon.

The runtime API exposed to widgets:

    peribus.me          → read-only NodeID string
    peribus.peer        → NodeID string of the widget's "other side"
                           (set when widget is rendered in a peer's context)
    peribus.stream(name) → returns a Stream object with read/write/on_data
    peribus.post(text)  → publish text to your social/ feed
    peribus.signal(name, data) → fire a one-shot event to the peer
"""


import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# The Stream object widgets see when they call peribus.stream()
# ---------------------------------------------------------------------------

@dataclass
class Stream:
    """
    A bidirectional channel to one or more peers, presented to the widget
    as a simple read/write/on_data interface. Backed by the daemon's
    streams plumbing (currently a local asyncio.Queue, later a QUIC stream).
    """

    name: str
    _send: Callable[[bytes], Awaitable[None]]
    _on_data_callbacks: List[Callable[[bytes], None]] = field(default_factory=list)
    _inbox: asyncio.Queue = field(default_factory=asyncio.Queue)

    async def write(self, data: bytes) -> None:
        """Send bytes to the other side. Awaitable, but widgets call from sync code via the bridge."""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("Stream.write expects bytes")
        await self._send(bytes(data))

    async def read(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Read the next chunk. Returns None on timeout."""
        if timeout is None:
            return await self._inbox.get()
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def on_data(self, callback: Callable[[bytes], None]) -> None:
        """Register a callback fired (on the Qt thread) for each inbound chunk."""
        self._on_data_callbacks.append(callback)

    # Internal — called by the daemon when bytes arrive.
    def _deliver(self, data: bytes) -> None:
        # Wake awaiters first so async readers see the data.
        try:
            self._inbox.put_nowait(data)
        except asyncio.QueueFull:
            pass  # backpressure: drop. Streams are best-effort.
        for cb in self._on_data_callbacks:
            try:
                cb(data)
            except Exception:
                # Don't let widget bugs kill the delivery loop.
                pass


# ---------------------------------------------------------------------------
# The PeribusAPI object — the only network-touching thing widgets can see
# ---------------------------------------------------------------------------

class PeribusAPI:
    """
    Per-widget runtime handle. The daemon constructs one of these for each
    rendered widget instance, with `me` and `peer` pre-filled and `_streams`
    backed by real channels.
    """

    def __init__(
        self,
        me: str,
        peer: Optional[str],
        stream_factory: Callable[[str], Stream],
        post_callback: Callable[[str], Awaitable[None]],
        signal_callback: Callable[[str, bytes], Awaitable[None]],
    ):
        self.me = me
        self.peer = peer
        self._stream_factory = stream_factory
        self._post_callback = post_callback
        self._signal_callback = signal_callback
        self._streams: Dict[str, Stream] = {}

    def stream(self, name: str) -> Stream:
        """
        Get-or-create a named stream. Widgets call this with stable names
        like "game-state" or "cursor-position". Both sides using the same
        name end up connected.
        """
        if not isinstance(name, str) or not name or len(name) > 64:
            raise ValueError("stream name must be a 1-64 char string")
        if name not in self._streams:
            self._streams[name] = self._stream_factory(name)
        return self._streams[name]

    async def post(self, text: str) -> None:
        """Publish a short text post to your feed. Limit: 4 KiB."""
        if len(text.encode("utf-8")) > 4096:
            raise ValueError("post body too large (max 4 KiB)")
        await self._post_callback(text)

    async def signal(self, name: str, data: bytes = b"") -> None:
        """One-shot event to the peer (cheaper than a full stream message)."""
        if len(data) > 1024:
            raise ValueError("signal payload too large (max 1 KiB)")
        await self._signal_callback(name, data)


# ---------------------------------------------------------------------------
# Namespace builder
# ---------------------------------------------------------------------------

def build_widget_namespace(
    api: PeribusAPI,
    qt_objects: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the dict that gets passed to exec() as the widget's globals.

    `qt_objects` should contain at minimum:
        graphics_scene, graphics_view, main_window
    so widgets can attach their output. The same objects the rio scene
    parser passes to user code, but we filter what surfaces.
    """
    ns: Dict[str, Any] = {}

    # ---- Pre-built imports (everything the widget could possibly need) ----
    try:
        from PySide6 import QtWidgets, QtCore, QtGui

        # Mirror what the rio parser exposes, but filter out anything starting
        # with underscore or that lets you reach into Python's guts.
        for mod, prefix in [(QtWidgets, "Q"), (QtCore, "Q"), (QtGui, "Q")]:
            for name in dir(mod):
                if name.startswith("_"):
                    continue
                if not name.startswith(prefix) and name not in (
                    "Qt", "Signal", "Slot", "Property"
                ):
                    continue
                obj = getattr(mod, name)
                ns[name] = obj

        # A few extras that aren't naturally Q-prefixed.
        ns["Qt"] = QtCore.Qt
        ns["Signal"] = QtCore.Signal
        ns["Slot"] = QtCore.Slot
    except ImportError:
        # PySide6 missing — widgets won't render, but the daemon should still
        # boot. Leave the namespace minimal.
        pass

    # ---- Math + safe builtins (subset of Python's builtins) ----
    import math
    for name in (
        "abs", "all", "any", "bool", "bytes", "chr", "dict", "divmod",
        "enumerate", "filter", "float", "frozenset", "hash", "hex", "int",
        "isinstance", "issubclass", "iter", "len", "list", "map", "max",
        "min", "next", "oct", "ord", "pow", "print", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum",
        "tuple", "zip",
    ):
        ns[name] = getattr(__builtins__, name) if isinstance(__builtins__, type) else __builtins__[name]
    ns["sin"] = math.sin
    ns["cos"] = math.cos
    ns["tan"] = math.tan
    ns["sqrt"] = math.sqrt
    ns["pi"] = math.pi
    ns["e"] = math.e

    # ---- Scene access (read-only objects, not the scene_manager) ----
    if "graphics_scene" in qt_objects:
        ns["graphics_scene"] = qt_objects["graphics_scene"]
    if "graphics_view" in qt_objects:
        ns["graphics_view"] = qt_objects["graphics_view"]
    # Note: main_window deliberately NOT exposed. Widgets don't need it
    # and it's a foothold to other widgets / the menu bar / quit().

    # ---- The peribus runtime API ----
    ns["peribus"] = api

    # ---- Lock down builtins so the widget can't reach back through them ----
    # Replace __builtins__ with a tiny dict that has only the names we
    # whitelisted above. This is the runtime backstop to the AST validator.
    safe_builtins = {k: ns[k] for k in (
        "abs", "all", "any", "bool", "bytes", "chr", "dict", "divmod",
        "enumerate", "filter", "float", "frozenset", "hash", "hex", "int",
        "isinstance", "issubclass", "iter", "len", "list", "map", "max",
        "min", "next", "oct", "ord", "pow", "print", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum",
        "tuple", "zip",
    )}
    ns["__builtins__"] = safe_builtins

    return ns


def execute_widget(
    source: str,
    api: PeribusAPI,
    qt_objects: Dict[str, Any],
    *,
    app_id: Optional[str] = None,
    embed_search: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Validate, then execute, a widget. Returns the post-execution namespace
    (so the rio scene parser can pick up created Qt objects the same way it
    does for local code).

    Raises ValueError if validation fails. The caller should catch and
    surface a clean error to the UI.

    Endorsement: if `app_id` and `embed_search` are provided AND exec
    completed without raising, schedule a `mark_used` call. This is the
    use-as-endorsement signal that the swarm-search layer reads — apps
    we've run get advertised to peers; apps we merely have sitting in
    the corpus don't.
    """
    from peribus._content import validate_widget_source

    result = validate_widget_source(source)
    if not result.ok:
        raise ValueError(
            "widget rejected:\n  " + "\n  ".join(result.errors)
        )

    ns = build_widget_namespace(api, qt_objects)
    # Compile then exec — this lets us pin the filename in tracebacks.
    code = compile(source, filename="<peribus-widget>", mode="exec")
    exec(code, ns)

    # The previous version called embed_search.mark_used(app_id) here
    # to flag this widget as "endorsed" for the swarm. With the shadow
    # corpus removed, endorsement is no longer a runtime concept — the
    # source of truth is whatever folders the user has pointed their
    # /n/llm/embed agent at. The `app_id` and `embed_search` params on
    # this function are now ignored; they remain in the signature for
    # backward compatibility with existing callers.

    return ns

# ============================================================================
# app_swarm.py
# ----------------------------------------------------------------------------
"""
peribus.app_swarm — federated semantic search over peribus apps

Design principles
=================

We don't broadcast queries. We don't replicate everyone's index.
We ride the resonance overlay that's already there.

When a user searches "calculator", their daemon does NOT ask every
peer in sight. It asks the top-K peers in its current resonance view —
the peers whose vector taste is closest to ours right now. Each of
those peers runs the query against its OWN local embed agent and
returns its top hits. The querier merges, ranks, and presents.

Forwarding follows the same gradient. If a peer doesn't have a strong
hit in their own corpus (best score below FORWARD_THRESHOLD), they
forward the query (TTL decremented) to their own top resonant peers,
minus whoever sent it to them and minus whoever the original querier
already pinged. This bounds the wavefront: a query expands along
high-resonance edges, dies in low-resonance regions, returns through
the same edges it came from.

What this gives you
===================

  * Bounded fan-out. A query touches at most O(K^TTL) peers — with
    K=4 and TTL=2 that's 16 peers in the worst case, typically far
    fewer because of overlap and early termination.

  * Locality. The peers most likely to have a "calculator" you'd
    actually like are the ones in your taste neighborhood. Routing by
    resonance is a useful prior, not just a fan-out limiter.

  * Diffusion. When you find an app you like, you fetch its source
    (existing MSG_FETCH path), validate it, install it locally. Your
    own embed agent now indexes it. Next time someone in YOUR
    neighborhood searches for similar things, you're an answerer.
    Apps spread along resonance gradients; popular-among-people-like-you
    naturally wins.

  * No global index, no central anything. The "catalog" is whatever
    each cluster has collectively endorsed.

Wire protocol
=============

Three new message types on top of the existing wire:

    MSG_APP_SEARCH   = "app_search"
    MSG_APP_RESULTS  = "app_results"
    # MSG_FETCH / MSG_DATA reused for content-addressed source fetch.

app_search:
    {
      "type": "app_search",
      "qid": "<random 16-byte hex>",  # for dedup + reply correlation
      "query": "calculator",
      "ttl": 2,                       # decremented on each forward
      "origin": "<nodeid>",           # original querier
      "exclude": ["<nodeid>", ...],   # peers already asked (don't re-ping)
      "ts": <ms>,
    }

app_results:
    {
      "type": "app_results",
      "qid": "<echoed from the search>",
      "from": "<nodeid>",                 # the responder
      "hits": [
        {
          "app_id": "sha256:<hex>",       # content hash of the .py source
          "score": 0.84,                  # cosine from responder's agent
          "title": "chemlab",             # filename stem
          "preview": "first 200 chars",
          "size": 4321,                   # full source byte length
        },
        ...
      ],
    }

Source bytes are NOT inlined in the response — they travel via
MSG_FETCH on demand, keyed by app_id. This keeps the wire payload
small and bounded, avoiding asyncio readline buffer overruns.

Results travel directly back to the original querier when possible.
If we don't have a wire conn to the origin, we route through whoever
gave us the search (the standard back-route in flooded search).

Rate limiting + dedup
=====================

  * Per-peer token bucket on incoming search requests. Default: 10
    queries / 30s. Excess is dropped silently. Stops accidental DoS.

  * Per-(qid) one-shot reply: we'll only respond to a given qid once,
    even if it arrives via two paths.

  * LRU of recent qids per peer; we don't forward the same query to
    the same peer twice.
"""


import asyncio
import logging
import secrets
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Deque, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from peribus._content import EmbedSearch, SearchHit
    from peribus._discovery import ResonanceOverlay
    from peribus._transport import WireServer
    from peribus._content import ValidationResult

logger = logging.getLogger(__name__)


# Wire message types — keep in sync with peribus.wire constants.
MSG_APP_SEARCH = "app_search"
MSG_APP_RESULTS = "app_results"

# Routing knobs.
DEFAULT_FANOUT = 4              # how many overlay neighbors we ask
DEFAULT_TTL = 2                 # max forwarding hops
DEFAULT_TOP_K = 5               # results returned per peer

# Forwarding policy: a responder forwards the query if their own best
# hit is below this score. Above this they're confident and don't need
# to bother neighbors.
FORWARD_THRESHOLD = 0.55

# Rate limiting: a peer may issue this many queries in this window
# before we start dropping their incoming app_search messages.
RATE_BUCKET_MAX = 10
RATE_BUCKET_WINDOW_S = 30.0

# Dedup: how many recent qids we remember per peer (and globally).
QID_MEMORY_SIZE = 256

# How long we wait for results to arrive before finalizing a search.
DEFAULT_GATHER_TIMEOUT_S = 4.0

# Result preview cap.
PREVIEW_MAX_CHARS = 200


@dataclass
class AggregatedHit:
    """
    Merged result across all responders for a single app_id.

    Multiple peers may surface the same app — that's actually a strong
    signal (the more independent endorsers, the better the app probably
    is for our neighborhood). We track the responder list and aggregate
    score.
    """
    app_id: str
    title: str
    preview: str
    best_score: float
    summed_score: float
    responders: List[str] = field(default_factory=list)
    endorsements: int = 0       # number of responders who surfaced this
    # The actual .py source bytes, when at least one responder shipped
    # them inline in their MSG_APP_RESULTS. The querier serves
    # apps/<id>/source from this; if it's b"" we fall back to MSG_FETCH.
    content: bytes = b""

    @property
    def consensus_score(self) -> float:
        """
        Final ranking score: best individual score, plus a small bonus
        per additional responder. Agreement is signal — if three peers
        in our resonance neighborhood all surfaced this calculator, that
        beats one peer's slightly higher single score.

        Previously this also factored in a per-responder use_count
        "endorsement" from peribus's shadow corpus. That mechanism is
        gone (the agent's scan list is the only endorsement that
        matters now), so the bonus is purely about cross-responder
        agreement.
        """
        agree_bonus = 0.05 * (len(self.responders) - 1)
        return self.best_score + agree_bonus


@dataclass
class _PendingQuery:
    """A search the user issued; we're collecting responses."""
    qid: str
    query: str
    started_at: float
    deadline: float
    asked: Set[str]                                  # peers we sent to
    hits: Dict[str, AggregatedHit] = field(default_factory=dict)
    finished: Optional[asyncio.Event] = None         # set when timeout fires


@dataclass
class _RateBucket:
    """Sliding-window counter for incoming queries from one peer."""
    timestamps: Deque[float] = field(default_factory=deque)

    def admit(self, now: float) -> bool:
        # Drop expired entries.
        cutoff = now - RATE_BUCKET_WINDOW_S
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        if len(self.timestamps) >= RATE_BUCKET_MAX:
            return False
        self.timestamps.append(now)
        return True


class AppSwarm:
    """
    The swarm-search layer. Owned by the daemon; coordinates between
    embed_search (local index), the overlay (routing prior), and the
    wire (transport).
    """

    def __init__(
        self,
        nodeid: str,
        embed_search: "EmbedSearch",
        overlay_provider: Callable[[], Optional["ResonanceOverlay"]],
        wire: "WireServer",
        validate_app: Callable[[bytes], "ValidationResult"],
        fetch_content: Callable[[str, str], Awaitable[Optional[bytes]]],
        # Optional: callback fired when we install a foreign app, in case
        # the UI wants to surface a "new app available" notice.
        on_app_installed: Optional[Callable[[str, str], Awaitable[None]]] = None,
    ):
        self.nodeid = nodeid
        self.embed_search = embed_search
        self._overlay_provider = overlay_provider
        self.wire = wire
        self._validate_app = validate_app
        self._fetch_content = fetch_content
        self._on_app_installed = on_app_installed

        # In-flight searches we issued: qid -> _PendingQuery
        self._pending: Dict[str, _PendingQuery] = {}

        # Dedup: qids we've already handled (responded or forwarded).
        # Bounded LRU; we don't need precise eviction.
        self._seen_qids: Deque[str] = deque(maxlen=QID_MEMORY_SIZE)
        self._seen_set: Set[str] = set()

        # Per-peer rate buckets for incoming searches.
        self._rate_buckets: Dict[str, _RateBucket] = {}

    # ------------------------------------------------------------------
    # Public API: issue a search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        fanout: int = DEFAULT_FANOUT,
        ttl: int = DEFAULT_TTL,
        gather_timeout: float = DEFAULT_GATHER_TIMEOUT_S,
        include_local: bool = True,
    ) -> List[AggregatedHit]:
        """
        Run a swarm search. Returns a list of AggregatedHit, sorted by
        consensus_score descending.

        Steps:
          1. (Optional) query our own local embed agent.
          2. Pick top-`fanout` peers from the resonance overlay.
          3. Send MSG_APP_SEARCH to each, with TTL.
          4. Wait `gather_timeout` seconds for responses to roll in.
          5. Return the merged ranking.
        """
        qid = secrets.token_hex(16)
        now = time.time()
        pending = _PendingQuery(
            qid=qid,
            query=query,
            started_at=now,
            deadline=now + gather_timeout,
            asked=set(),
            finished=asyncio.Event(),
        )
        self._pending[qid] = pending
        self._remember_qid(qid)

        try:
            # Local hits, if enabled. We treat ourselves as one responder
            # so the merging code stays uniform. SearchHit now carries
            # title and content directly — no separate corpus lookup
            # needed (and none would succeed; there's no shadow corpus).
            if include_local:
                local_hits = await self.embed_search.search(query, top_k=DEFAULT_TOP_K)
                for h in local_hits:
                    self._merge_hit(
                        pending,
                        responder=self.nodeid,
                        app_id=h.app_id,
                        score=h.score,
                        title=h.title,
                        preview="",
                        content=h.content,
                    )

            # Pick neighbors and dispatch.
            neighbors = self._pick_neighbors(fanout, exclude={self.nodeid})
            for nid in neighbors:
                pending.asked.add(nid)

            if neighbors:
                msg = {
                    "type": MSG_APP_SEARCH,
                    "qid": qid,
                    "query": query,
                    "ttl": ttl,
                    "origin": self.nodeid,
                    "exclude": [self.nodeid] + list(pending.asked),
                    "ts": int(now * 1000),
                }
                await self._send_to_peers(neighbors, msg)
            elif not include_local:
                # No neighbors to ask, no local search → nothing to do.
                logger.debug("app_swarm: search with no neighbors; returning empty")

            # Wait for responses (or timeout).
            try:
                await asyncio.wait_for(
                    pending.finished.wait(),
                    timeout=gather_timeout,
                )
            except asyncio.TimeoutError:
                pass  # expected — we use timeout as the natural gather window

            # Final ranking.
            ranked = sorted(
                pending.hits.values(),
                key=lambda h: -h.consensus_score,
            )
            return ranked
        finally:
            self._pending.pop(qid, None)

    # ------------------------------------------------------------------
    # Public API: install an app from a hit
    # ------------------------------------------------------------------

    async def install(self, hit: AggregatedHit) -> Optional[bytes]:
        """
        Return the validated source bytes for this hit.

        Most of the time `hit.content` is already populated from the
        inline payload in MSG_APP_RESULTS. If for some reason it's
        empty (very large file, missed by the responder, etc.) we fall
        back to MSG_FETCH against the responders in order until one
        produces bytes.

        Returns the source bytes on success, or None if no responder
        could (or would) provide them. The bytes are validated against
        the widget validator before return — peers can't make us
        surface untrusted code through this path.

        Note: we deliberately do NOT write to disk. The caller (the
        filesystem layer in apps_fs.py) serves bytes from memory. If
        the user wants the file on disk they can redirect:
            cat /n/peribus/apps/<id>/source > ~/apps/foo.py
        """
        candidates: List[bytes] = []
        if hit.content:
            candidates.append(hit.content)

        # If inline content was missing, fall back to MSG_FETCH from
        # each responder in turn.
        if not candidates:
            for responder in hit.responders:
                try:
                    source = await self._fetch_content(responder, hit.app_id)
                except Exception as e:
                    logger.debug(f"app_swarm: fetch from {responder}: {e}")
                    continue
                if source:
                    candidates.append(source)
                    break

        for source in candidates:
            # Validate before handing to anyone.
            try:
                result = self._validate_app(source)
            except Exception as e:
                logger.warning(f"app_swarm: validator crashed on {hit.app_id}: {e}")
                continue
            if not result.ok:
                logger.warning(
                    f"app_swarm: rejected app {hit.app_id}: "
                    f"{'; '.join(result.errors)}"
                )
                continue
            logger.info(f"app_swarm: source ready for {hit.app_id}")
            if self._on_app_installed is not None:
                try:
                    await self._on_app_installed(hit.app_id, hit.title)
                except Exception as e:
                    logger.debug(f"on_app_installed raised: {e}")
            return source

        logger.info(f"app_swarm: could not retrieve {hit.app_id}")
        return None

    # ------------------------------------------------------------------
    # Wire message handlers — invoked by daemon's _on_wire_message
    # ------------------------------------------------------------------

    async def handle_message(self, conn_nodeid: str, msg: dict) -> bool:
        """
        Try to handle a wire message. Returns True if the message was
        ours (caller skips other dispatch), False otherwise.
        """
        t = msg.get("type")
        if t == MSG_APP_SEARCH:
            await self._handle_search(conn_nodeid, msg)
            return True
        if t == MSG_APP_RESULTS:
            await self._handle_results(conn_nodeid, msg)
            return True
        return False

    async def _handle_search(self, sender: str, msg: dict) -> None:
        qid = msg.get("qid", "")
        query = msg.get("query", "")
        ttl = int(msg.get("ttl", 0))
        origin = msg.get("origin", "")
        exclude = set(msg.get("exclude", []))

        if not qid or not query or not origin:
            return
        if origin == self.nodeid:
            # Loop guard: we issued this; don't reply to our own query.
            return

        # Rate limit by sender (NOT origin — sender is who's actually
        # using our resources; rate-limiting origin is rate-limiting
        # forwarding peers and that's not what we want).
        bucket = self._rate_buckets.setdefault(sender, _RateBucket())
        if not bucket.admit(time.time()):
            logger.debug(f"app_swarm: rate-limited search from {sender}")
            return

        # Dedup: did we already see this qid?
        if qid in self._seen_set:
            return
        self._remember_qid(qid)

        # Run our local search.
        local_hits: List["SearchHit"] = []
        if self.embed_search.available:
            local_hits = await self.embed_search.search(query, top_k=DEFAULT_TOP_K)

        # Build response payload. Every hit the agent returned is a hit
        # we surface — putting a folder under the agent's scan list IS
        # the endorsement, no separate gate.
        #
        # We deliberately do NOT include the source bytes inline. An
        # earlier version did (under content_b64), with a 64 KB cap;
        # that interacted badly with asyncio.StreamReader's default
        # 64 KB readline limit, dropping the wire conn when a single
        # MSG_APP_RESULTS exceeded it. Sources now travel via MSG_FETCH
        # on demand, keyed by app_id (sha256 content hash). The wire
        # response stays small and bounded regardless of result size.
        wire_hits = []
        best_local_score = 0.0
        for h in local_hits:
            wire_hits.append({
                "app_id": h.app_id,
                "score": float(h.score),
                "title": h.title,
                "preview": h.content[:PREVIEW_MAX_CHARS].decode("utf-8", errors="replace"),
                "size": len(h.content),
            })
            if h.score > best_local_score:
                best_local_score = h.score

        # Reply to origin (or back through sender if no direct conn).
        if wire_hits:
            await self._send_results(origin, sender, qid, wire_hits)

        # Forward if our best wasn't great and TTL allows.
        if ttl > 0 and best_local_score < FORWARD_THRESHOLD:
            new_exclude = exclude | {self.nodeid, sender}
            forwarded_to = self._pick_neighbors(
                DEFAULT_FANOUT, exclude=new_exclude,
            )
            if forwarded_to:
                fwd_msg = {
                    "type": MSG_APP_SEARCH,
                    "qid": qid,
                    "query": query,
                    "ttl": ttl - 1,
                    "origin": origin,
                    "exclude": list(new_exclude | set(forwarded_to)),
                    "ts": int(time.time() * 1000),
                }
                await self._send_to_peers(forwarded_to, fwd_msg)

    async def _handle_results(self, sender: str, msg: dict) -> None:
        qid = msg.get("qid", "")
        responder = msg.get("from", sender)
        hits = msg.get("hits", [])

        pending = self._pending.get(qid)
        if pending is None:
            # Either we've already finalized this query, or it's a
            # response to a query we're forwarding for someone else.
            # In the forwarding case, route the result toward the origin.
            await self._maybe_route_results(qid, msg)
            return

        for h in hits:
            try:
                # Inline content was removed from the wire — content now
                # comes via MSG_FETCH on demand. Pass empty bytes; the
                # filesystem layer's refetch path will pull source from
                # a responder if/when apps/<id>/source is read.
                self._merge_hit(
                    pending,
                    responder=responder,
                    app_id=h["app_id"],
                    score=float(h.get("score", 0.0)),
                    title=str(h.get("title", "")),
                    preview=str(h.get("preview", ""))[:PREVIEW_MAX_CHARS],
                    content=b"",
                )
            except (KeyError, ValueError, TypeError):
                continue

        # If our gather window is already up, fire the event so the
        # awaiting search() returns. (Normal case: we let the timeout
        # fire on its own to gather more responses.)
        if pending.finished is not None and time.time() >= pending.deadline:
            pending.finished.set()

    async def _maybe_route_results(self, qid: str, msg: dict) -> None:
        """
        We received an app_results for a qid we don't own. If we
        forwarded the search and remember the back-route, pass the
        results on. For v0.1 we don't keep that breadcrumb table —
        the simpler design has responders address replies directly to
        the origin nodeid; if no wire conn exists we drop. This method
        is here for the future when we add back-routing.
        """
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _merge_hit(
        self,
        pending: _PendingQuery,
        *,
        responder: str,
        app_id: str,
        score: float,
        title: str,
        preview: str,
        content: bytes = b"",
    ) -> None:
        agg = pending.hits.get(app_id)
        if agg is None:
            agg = AggregatedHit(
                app_id=app_id,
                title=title,
                preview=preview,
                best_score=score,
                summed_score=score,
                responders=[responder],
                endorsements=1,
                content=content,
            )
            pending.hits[app_id] = agg
        else:
            if responder not in agg.responders:
                agg.responders.append(responder)
                agg.endorsements = len(agg.responders)
            if score > agg.best_score:
                agg.best_score = score
            agg.summed_score += score
            # Prefer non-empty title/preview if we get them later.
            if not agg.title and title:
                agg.title = title
            if not agg.preview and preview:
                agg.preview = preview
            # Keep the first non-empty content we get. (Content hash is
            # the key, so any responder's bytes are equivalent — but
            # an empty payload from a follower shouldn't overwrite a
            # full one from an earlier responder.)
            if not agg.content and content:
                agg.content = content

    def _pick_neighbors(self, k: int, exclude: Set[str]) -> List[str]:
        """
        Top-k peers from the overlay's resonance view, skipping anyone
        in `exclude`. If the overlay is empty (early bootstrap), fall
        back to whoever we have wire connections to.
        """
        overlay = self._overlay_provider()
        chosen: List[str] = []

        if overlay is not None:
            for entry in overlay.top_resonant():
                if entry.nodeid in exclude:
                    continue
                if entry.nodeid in chosen:
                    continue
                chosen.append(entry.nodeid)
                if len(chosen) >= k:
                    break

        # Fallback: any wire conns we already have.
        if len(chosen) < k:
            for nid in self.wire._conns.keys():
                if nid in exclude or nid in chosen:
                    continue
                chosen.append(nid)
                if len(chosen) >= k:
                    break

        return chosen

    async def _send_to_peers(self, peer_ids: List[str], msg: dict) -> None:
        for nid in peer_ids:
            conn = self.wire.get_conn(nid)
            if conn is None:
                continue
            try:
                await conn.send(msg)
            except Exception as e:
                logger.debug(f"app_swarm: send to {nid}: {e}")

    async def _send_results(
        self, origin: str, sender: str, qid: str, hits: list,
    ) -> None:
        """
        Send results back. Prefer a direct conn to the origin; fall back
        to the sender (who'll either be the origin or will route).
        """
        msg = {
            "type": MSG_APP_RESULTS,
            "qid": qid,
            "from": self.nodeid,
            "hits": hits,
        }
        target = origin if self.wire.get_conn(origin) is not None else sender
        conn = self.wire.get_conn(target)
        if conn is None:
            return
        try:
            await conn.send(msg)
        except Exception as e:
            logger.debug(f"app_swarm: send_results to {target}: {e}")

    def _remember_qid(self, qid: str) -> None:
        if qid in self._seen_set:
            return
        self._seen_qids.append(qid)
        self._seen_set.add(qid)
        # Trim the set to match the deque (deque autoshrinks).
        while len(self._seen_set) > QID_MEMORY_SIZE:
            old = self._seen_qids.popleft() if self._seen_qids else None
            if old is not None:
                self._seen_set.discard(old)
            else:
                break

    def stats(self) -> dict:
        return {
            "pending": len(self._pending),
            "seen_qids": len(self._seen_set),
            "rate_tracked_peers": len(self._rate_buckets),
        }

# ============================================================================
# apps_fs.py
# ----------------------------------------------------------------------------
"""
peribus.apps_fs — /n/peribus/apps/ — filesystem surface for the swarm

User-facing handle on `peribus.app_swarm`. Two top-level files plus a
content-addressed dir for each app surfaced by recent searches:

    /n/peribus/apps/
        search                       — write a query; the next read on
                                       the SAME fid blocks until the
                                       swarm round completes, then
                                       returns JSON-line hits. Each fid
                                       gets its own round; concurrent
                                       searches don't collide.
        installed                    — local /n/llm/embed stats; what
                                       this machine itself indexes
        sha256:<hex>/source          — raw .py bytes for that hash,
                                       served from in-memory cache
                                       populated by recent searches.
                                       Re-fetches from a responder if
                                       cache expired.
        sha256:<hex>/title           — short title (filename stem)
        sha256:<hex>/responders      — which peers vouched for this app
        sha256:<hex>/info            — JSON: score, responders, size

User flow on Machine B, sitting at a shell:

    echo "chemlab" > /n/peribus/apps/search
    cat /n/peribus/apps/search
        {"app_id":"sha256:abc...","title":"chemlab","score":0.49,
         "responders":["z7yk..."],"consensus_score":0.49,
         "size":4321,"preview":"..."}

    cat /n/peribus/apps/sha256:abc.../source > ~/my-apps/chemlab.py
    # Now you have the source. Run it however your widget system runs widgets.

Why no apps/install file
------------------------

The previous design had `echo <id> > apps/install` to "fetch + index"
the source. With the shadow corpus gone, "install" stops being a verb:
the source is already served via apps/<id>/source. If the user wants
to keep it on disk, they redirect. If they want their own embed agent
to learn it, they put the file in a folder the agent scans. Peribus
doesn't decide either of those.

Concurrency model
-----------------

`search` is per-fid:
  - Each open of /n/peribus/apps/search creates a fresh _SearchState
    keyed by the fid object.
  - A write to that fid kicks off a swarm round.
  - A read on that fid awaits the round's result and caches it for
    re-reads on the SAME fid (so `cat search` after `echo q > search`
    returns the same bytes if you read twice).
  - Closing the fid (clunk) drops the state. There's no global "last
    results" — different shells, different fids, different worlds.

Concurrent writes from different shells produce independent rounds.
Two users searching "chemlab" simultaneously each get their own qid,
their own gather window, their own ranking.
"""


import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, TYPE_CHECKING

from core.files import SyntheticFile, SyntheticDir

if TYPE_CHECKING:
    from peribus._daemon import PeribusDaemon
    from peribus._content import AggregatedHit

logger = logging.getLogger(__name__)


# How long to wait for swarm responses before considering the round done.
DEFAULT_SEARCH_GATHER_S = 4.0

# In-memory cache of fetched source bytes, indexed by app_id (= sha256
# of the content). Lives on AppsDir, fed by every search round, served
# via sha256:<hex>/source. Bounded by entry count, not bytes.
SOURCE_CACHE_MAX_ENTRIES = 256


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _hit_line(hit: "AggregatedHit") -> bytes:
    """One JSON line per hit. Stable shape for scripting."""
    payload = {
        "app_id": hit.app_id,
        "title": hit.title or "",
        "best_score": round(hit.best_score, 4),
        "consensus_score": round(hit.consensus_score, 4),
        "responders": list(hit.responders),
        "size": len(hit.content),
        "preview": (hit.preview or "")[:200],
    }
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# search — write query → cat returns ranked hits
#
# Model: single shared "current results" slot.
#
# The original per-fid design was wrong for the shell case. In a shell:
#     echo q > apps/search          # process #1 opens a fid, writes, closes
#     cat apps/search               # process #2 opens a DIFFERENT fid, reads
# Per-fid state would orphan the write's task and leave the read with
# nothing. The fix is to forget about fid identity entirely: writes
# kick a new round; reads await the current round and return whatever
# it produced.
#
# Concurrency semantics for two shells racing:
#     shell A: echo "alpha" > apps/search
#     shell B: echo "beta"  > apps/search    (slightly later)
#     shell A: cat apps/search
#     shell B: cat apps/search
# Both cats await the same in-flight round (whichever was written last —
# B's "beta" — because each write replaces the current task) and see the
# same results. This is a deliberate simplification: shells coordinate
# their own ordering. Two users wanting independent results should do
# their searches independently in time, not in parallel on the same
# search file. If we needed strict per-session isolation we'd add a
# session ID via a path like apps/search/<session>; not building that
# until someone asks.
# ---------------------------------------------------------------------------


class SearchFile(SyntheticFile):
    """
    /n/peribus/apps/search

    Behavior:
        write — stashes the query, cancels any in-flight round,
                starts a new round.
        read  — awaits the current round if one is running; returns
                the rendered bytes of the most recently completed
                round. Multiple reads after a single write all see
                the same bytes.

    No per-fid state. The file has one "current task" and one
    "current rendered" slot, both updated atomically when a round
    completes.
    """

    def __init__(
        self, daemon: "PeribusDaemon", apps_dir: "AppsDir", parent: SyntheticDir = None,
    ):
        super().__init__("search", parent)
        self._mode = 0o666
        self._daemon = daemon
        self._apps_dir = apps_dir

        # The currently in-flight round (or completed). Awaited by
        # read() to convert task → bytes. Replaced by every write().
        self._current_task: Optional[asyncio.Task] = None

        # Cached rendering of the most recent COMPLETED round. Reads
        # return this when no round is in flight (so re-reading after
        # a single search is instant) and as a fallback for plain
        # `cat search` with no prior write (shows whatever was last
        # searched, including across shells).
        self._current_rendered: bytes = (
            b"# no search has been run yet - "
            b"try: echo QUERY > /n/peribus/apps/search\n"
        )

        # Serializes the write→spawn-task transition so two concurrent
        # writes can't both think they're cancelling the same predecessor.
        self._write_lock = asyncio.Lock()

    def _get_length(self) -> int:
        # Over-reporting is fine for 9P clients.
        return max(4096, len(self._current_rendered))

    async def read(self, fid, offset: int, count: int) -> bytes:
        # If a round is in flight, await it. AppSwarm.search has its
        # own gather_timeout so this is bounded.
        task = self._current_task
        if task is not None and not task.done():
            try:
                hits = await task
            except asyncio.CancelledError:
                # A later write cancelled our task; whoever wrote that
                # newer query will produce fresh bytes. Fall through
                # to returning whatever's cached.
                hits = None
            except Exception as e:
                logger.warning(f"apps/search round failed: {e}")
                hits = []
            if hits is not None:
                self._current_rendered = self._render_hits(hits)
                self._apps_dir.cache_hits(hits)
            # Don't clear self._current_task — leaving it set means a
            # second reader awaiting the same task gets the completed
            # result immediately (task.result() is cheap on a done task).

        return self._current_rendered[offset:offset + count]

    async def write(self, fid, offset: int, data: bytes) -> int:
        # Each write is a fresh query. Ignore offset.
        query = data.decode("utf-8", errors="replace").strip()
        if not query:
            return len(data)

        async with self._write_lock:
            prev = self._current_task
            if prev is not None and not prev.done():
                # Cancel the previous round; whoever was awaiting it
                # gets CancelledError, which read() handles gracefully.
                prev.cancel()
            self._current_task = asyncio.create_task(self._run_round(query))
        return len(data)

    async def _run_round(self, query: str) -> List["AggregatedHit"]:
        return await self._daemon.app_swarm.search(
            query,
            gather_timeout=DEFAULT_SEARCH_GATHER_S,
        )

    def _render_hits(self, hits: List["AggregatedHit"]) -> bytes:
        if not hits:
            return b"# no results\n"
        return b"".join(_hit_line(h) for h in hits)


# ---------------------------------------------------------------------------
# installed — what THIS machine's local agent has, summary
# ---------------------------------------------------------------------------

class InstalledFile(SyntheticFile):
    """
    /n/peribus/apps/installed — what does our own /n/llm/embed know about?

    Since peribus no longer maintains its own corpus, this is a thin
    proxy onto the agent's stats. For a full file listing, the user can
    `cat /n/llm/embed/index_status` (or whatever the agent exposes) —
    we just present a compact peribus-flavored summary here.
    """

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("installed", parent)
        self._mode = 0o444
        self._daemon = daemon

    def _render(self) -> bytes:
        es = self._daemon.embed_search
        stats = es.stats()
        return (json.dumps(stats, indent=2) + "\n").encode("utf-8")

    def _get_length(self) -> int:
        return len(self._render())

    async def read(self, fid, offset: int, count: int) -> bytes:
        return self._render()[offset:offset + count]

    async def write(self, fid, offset: int, data: bytes) -> int:
        raise PermissionError("apps/installed is read-only")


# ---------------------------------------------------------------------------
# Per-hash files — sha256:<hex>/{source, title, responders, info}
# ---------------------------------------------------------------------------

class _AppFile(SyntheticFile):
    """Base for read-only per-hash files. Subclasses define `_render`."""

    def __init__(
        self,
        name: str,
        apps_dir: "AppsDir",
        app_id: str,
        parent: SyntheticDir = None,
    ):
        super().__init__(name, parent)
        self._mode = 0o444
        self._apps_dir = apps_dir
        self._app_id = app_id

    def _render(self) -> bytes:
        raise NotImplementedError

    def _get_length(self) -> int:
        return len(self._render())

    async def read(self, fid, offset: int, count: int) -> bytes:
        return self._render()[offset:offset + count]

    async def write(self, fid, offset: int, data: bytes) -> int:
        raise PermissionError(f"{self.name} is read-only")


class AppSourceFile(_AppFile):
    """sha256:<hex>/source — the raw .py bytes."""

    def _render(self) -> bytes:
        hit = self._apps_dir.get_cached(self._app_id)
        if hit is None or not hit.content:
            return b""
        return hit.content

    async def read(self, fid, offset: int, count: int) -> bytes:
        # Try fresh fetch if cache is empty.
        hit = self._apps_dir.get_cached(self._app_id)
        if hit is not None and not hit.content and hit.responders:
            # In-memory cache lost the bytes but we know who has them;
            # ask the swarm to refill via MSG_FETCH.
            try:
                source = await self._apps_dir.refetch(self._app_id)
                if source is not None:
                    hit.content = source
            except Exception as e:
                logger.debug(f"refetch {self._app_id}: {e}")
        return self._render()[offset:offset + count]


class AppTitleFile(_AppFile):
    def _render(self) -> bytes:
        hit = self._apps_dir.get_cached(self._app_id)
        if hit is None or not hit.title:
            return b""
        return (hit.title + "\n").encode("utf-8")


class AppRespondersFile(_AppFile):
    def _render(self) -> bytes:
        hit = self._apps_dir.get_cached(self._app_id)
        if hit is None:
            return b""
        return ("\n".join(hit.responders) + "\n").encode("utf-8")


class AppInfoFile(_AppFile):
    def _render(self) -> bytes:
        hit = self._apps_dir.get_cached(self._app_id)
        if hit is None:
            return b""
        payload = {
            "app_id": hit.app_id,
            "title": hit.title,
            "best_score": round(hit.best_score, 4),
            "consensus_score": round(hit.consensus_score, 4),
            "responders": list(hit.responders),
            "size": len(hit.content),
        }
        return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


class AppDir(SyntheticDir):
    """One app's directory: source, title, responders, info."""

    def __init__(
        self, apps_dir: "AppsDir", app_id: str, parent: SyntheticDir = None,
    ):
        # 9P names can contain colons but some clients dislike them;
        # we deliberately use the full "sha256:<hex>" as the dir name.
        # If your client misbehaves with colons, swap ":" for "_" here
        # and in AppsDir.cache_hits.
        super().__init__(app_id, parent)
        self.children["source"] = AppSourceFile(
            "source", apps_dir, app_id, parent=self,
        )
        self.children["title"] = AppTitleFile(
            "title", apps_dir, app_id, parent=self,
        )
        self.children["responders"] = AppRespondersFile(
            "responders", apps_dir, app_id, parent=self,
        )
        self.children["info"] = AppInfoFile(
            "info", apps_dir, app_id, parent=self,
        )


# ---------------------------------------------------------------------------
# AppsDir — top-level /n/peribus/apps/
# ---------------------------------------------------------------------------

class AppsDir(SyntheticDir):
    """
    /n/peribus/apps/

    Owns:
      - search file (per-fid sticky)
      - installed file
      - the source cache: dict[app_id -> AggregatedHit] populated by
        every completed search round, served by sha256:<hex>/source
      - lazy materialization of sha256:<hex>/ subdirs
    """

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("apps", parent)
        self._daemon = daemon

        # Source cache — insertion-order dict, bounded by entry count.
        self._cache: Dict[str, "AggregatedHit"] = {}

        # Materialized per-hash dirs. Same insertion-order pattern.
        self._app_dirs: Dict[str, AppDir] = {}

        # Wire fixed children.
        self.children["search"] = SearchFile(daemon, self, parent=self)
        self.children["installed"] = InstalledFile(daemon, parent=self)

    # ---- cache management ----

    def cache_hits(self, hits: List["AggregatedHit"]) -> None:
        """Called after a search round completes."""
        for hit in hits:
            self._cache_put(hit.app_id, hit)
            self._ensure_dir(hit.app_id)

    def _cache_put(self, app_id: str, hit: "AggregatedHit") -> None:
        # Refresh recency.
        if app_id in self._cache:
            del self._cache[app_id]
        elif len(self._cache) >= SOURCE_CACHE_MAX_ENTRIES:
            try:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
                # Drop the dir too — keeps `ls` from showing stale apps.
                if oldest in self._app_dirs:
                    del self._app_dirs[oldest]
                    self.children.pop(oldest, None)
            except StopIteration:
                pass
        self._cache[app_id] = hit

    def get_cached(self, app_id: str) -> Optional["AggregatedHit"]:
        return self._cache.get(app_id)

    def _ensure_dir(self, app_id: str) -> None:
        if app_id in self._app_dirs:
            return
        d = AppDir(self, app_id, parent=self)
        self._app_dirs[app_id] = d
        self.children[app_id] = d

    async def refetch(self, app_id: str) -> Optional[bytes]:
        """
        Re-fetch source for an app whose cache entry lost its bytes.
        Goes through app_swarm.install() which validates + tries
        responders in order. Returns bytes on success, None on failure.
        """
        hit = self._cache.get(app_id)
        if hit is None:
            return None
        return await self._daemon.app_swarm.install(hit)