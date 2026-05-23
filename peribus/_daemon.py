"""
peribus._daemon — concatenation of: filesystem.py, daemon.py, feed_bridge.py

This is a build artefact. The original module names live as section
banners below so `grep "^# ===="` jumps to each one.
"""

from __future__ import annotations


# ============================================================================
# filesystem.py
# ----------------------------------------------------------------------------
"""
peribus.filesystem — the /n/peribus synthetic tree

Built on the same SyntheticFile / SyntheticDir base classes as everything
else in the rio stack. The base classes own qid construction (from id(self))
and stat() construction; subclasses override read/write and _get_length.

The shape:

    /n/peribus/
        ctl                    — write commands here (start, stop, dial, follow, …)
        identity/
            nodeid             — your NodeID (read-only)
            vector             — your identity vector (binary float32)
            summary            — human-readable description (read/write)
        nodes/                 — discovered peers (dynamic)
            <nodeid>/          — populated when daemon discovers a peer
                signal         — strength, latency, last_seen
                social/        — their public posts
                inbox          — write here to send them a DM
                from           — DMs we've received from this peer (snapshot)
        feed/
            new                — tail-able stream of fresh posts (blocks)
            recent             — snapshot of the post buffer (EOFs)
        inbox/                 — direct messages received from peers
            new                — tail-able JSON-line stream (blocks)
            recent             — snapshot, EOFs once drained
            send               — write "<nodeid> <body>" to send a DM
        share/                 — clone-style draft directory for publishing
            clone              — read returns "<n>", and share/<n>/ is born
            <n>/               — one in-progress draft (numbered)
                body           — text body (writable; appends compose)
                kind           — "post" | "reply"
                reply_to       — parent post hash, if this is a reply
                attach         — write paths (one per line) to queue files
                ctl            — write "publish" to commit, "discard" to drop
                result         — read after publish: returns post hash. Blocks.
        streams/               — per-widget channels (populated by widgets)
        apps/                  — swarm-search surface (federated semantic search)
            search                  — write a query (per-fid sticky),
                                       read JSON-line ranked hits
            installed               — JSON: local /n/llm/embed stats
            sha256:<hex>/source     — raw .py bytes from a hit
            sha256:<hex>/title      — filename stem
            sha256:<hex>/responders — which peers vouched
            sha256:<hex>/info       — JSON: score, responders, size

The daemon owns the tree and is reachable from every node, so reads can
return live state.
"""


from typing import Optional, TYPE_CHECKING

import asyncio

from core.files import SyntheticFile, SyntheticDir

if TYPE_CHECKING:
    from peribus._daemon import PeribusDaemon


# How long a blocking read on feed/new or inbox/new will sit waiting
# before returning b"" as a keepalive. A 0-byte read tells the 9P
# client "I'm still here, just nothing new" — the tailer's existing
# "if not chunk: reconnect" branch handles it gracefully without an
# error log. Set well below the 9P transport's request timeout (which
# is what was producing the EIO storm on quiet feeds).
_BLOCKING_READ_KEEPALIVE_S = 30.0


# ---------------------------------------------------------------------------
# Mixins to satisfy SyntheticFile's abstract read/write contract.
#
# Read-only files raise EPERM-style errors on write attempts; write-only
# files do the same on read. The 9P server surfaces the exception as an
# Rerror so the client gets a clean "permission denied" instead of a
# crashed server.
# ---------------------------------------------------------------------------


class _ReadOnly:
    """Mixin: refuses writes. Combine before SyntheticFile in the MRO."""

    async def write(self, fid, offset: int, data: bytes) -> int:
        raise PermissionError(f"{self.name}: read-only file")


class _WriteOnly:
    """Mixin: refuses reads. Combine before SyntheticFile in the MRO."""

    async def read(self, fid, offset: int, count: int) -> bytes:
        raise PermissionError(f"{self.name}: write-only file")


# ---------------------------------------------------------------------------
# File subclasses
#
# Convention from core/files.py:
#   - __init__(self, name, parent=None) — base sets up qid / _mode / _mtime
#   - touch() bumps mtime and qid version when content changes
#   - _get_length() reports the file's size in bytes
#   - async read(fid, offset, count) returns bytes
#   - async write(fid, offset, data) returns int (bytes accepted)
# ---------------------------------------------------------------------------


class CtlFile(SyntheticFile):
    """
    /n/peribus/ctl — write commands, read status.

    Commands (one per line, write to the file):
        start                — bring up discovery + gossip
        stop                 — tear down network, keep filesystem mounted
        follow <nodeid>      — bias identity vector toward this peer
        attract <text>       — push current identity toward `text`
        share <path>         — publish a local file to the rhizome

    Reading the file returns the current daemon status as a few lines.
    """

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("ctl", parent)
        self._mode = 0o666
        self._daemon = daemon

    def _get_length(self) -> int:
        # Best-effort — actual length isn't known until we render status.
        # 9P clients tolerate over-reporting fine.
        return 1024

    async def read(self, fid, offset: int, count: int) -> bytes:
        status = await self._daemon.status_text()
        data = status.encode("utf-8")
        return data[offset:offset + count]

    async def write(self, fid, offset: int, data: bytes) -> int:
        # Commands are line-based and small. We ignore offset; control files
        # always interpret the full write as a fresh command batch.
        text = data.decode("utf-8", errors="replace").strip()
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            await self._daemon.handle_ctl(line)
        self.touch()
        return len(data)


class InfoFile(_ReadOnly, SyntheticFile):
    """A read-only file whose contents come from a named info-provider on
    the daemon. Used for things like /n/peribus/bootstrap whose value is
    owned by an extension layer (the global-discovery glue) rather than
    by the daemon itself.

    If the named info isn't currently registered, the file reads as empty.
    This means the file always exists (so scripts can `cat` it without an
    ENOENT race), but reads zero bytes when nothing is set.
    """

    def __init__(
        self,
        name: str,
        info_key: str,
        daemon: "PeribusDaemon",
        parent: SyntheticDir = None,
    ):
        super().__init__(name, parent)
        self._daemon = daemon
        self._info_key = info_key

    def _value(self) -> bytes:
        v = self._daemon.get_info(self._info_key)
        if not v:
            return b""
        return (v + "\n").encode("utf-8")

    def _get_length(self) -> int:
        return len(self._value())

    async def read(self, fid, offset: int, count: int) -> bytes:
        data = self._value()
        return data[offset:offset + count]


class NodeIdFile(_ReadOnly, SyntheticFile):
    """/n/peribus/identity/nodeid — read-only NodeID string."""

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("nodeid", parent)
        self._mode = 0o444
        self._daemon = daemon

    def _get_length(self) -> int:
        return len(self._daemon.identity.nodeid) + 1

    async def read(self, fid, offset: int, count: int) -> bytes:
        nodeid = self._daemon.identity.nodeid + "\n"
        data = nodeid.encode("utf-8")
        return data[offset:offset + count]


class IdentityVectorFile(_ReadOnly, SyntheticFile):
    """/n/peribus/identity/vector — binary float32 vector dump."""

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("vector", parent)
        self._mode = 0o444
        self._daemon = daemon

    def _get_length(self) -> int:
        # 4 bytes per float, dim known to embedder.
        return 4 * self._daemon.identity_vector.dim

    async def read(self, fid, offset: int, count: int) -> bytes:
        from peribus._foundation import pack_vector
        vec = self._daemon.identity_vector.snapshot()
        data = pack_vector(vec)
        return data[offset:offset + count]


class SummaryFile(SyntheticFile):
    """/n/peribus/identity/summary — what the daemon thinks you're about."""

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("summary", parent)
        self._mode = 0o644
        self._daemon = daemon

    def _get_length(self) -> int:
        return len(self._daemon.summary.encode("utf-8")) + 1

    async def read(self, fid, offset: int, count: int) -> bytes:
        text = self._daemon.summary + "\n"
        data = text.encode("utf-8")
        return data[offset:offset + count]

    async def write(self, fid, offset: int, data: bytes) -> int:
        # Summary is user-editable; the daemon uses it as a strong signal
        # for the identity vector.
        text = data.decode("utf-8", errors="replace").strip()
        await self._daemon.set_summary(text)
        self.touch()
        return len(data)


class FeedNewFile(_ReadOnly, SyntheticFile):
    """
    /n/peribus/feed/new — tail-able feed of fresh posts.

    Each read returns one or more JSON lines (one per post), ranked by
    relevance to the current identity vector. Reads block when there's
    nothing new — the feed bridge tails this file the same way `tail -f`
    works on a regular log.

    For one-shot reads (`cat`-style) use feed/recent instead.
    """

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("new", parent)
        self._mode = 0o444
        self._daemon = daemon

    def _get_length(self) -> int:
        # Streaming file — length is meaningless. Report 0; tail-style
        # readers don't care, they just keep reading.
        return 0

    async def read(self, fid, offset: int, count: int) -> bytes:
        # Each fid keeps its own cursor into the feed buffer.
        if not hasattr(fid, "_feed_cursor"):
            fid._feed_cursor = self._daemon.gossip.feed_cursor()

        # Block until at least one post arrives — but cap the wait at
        # _BLOCKING_READ_KEEPALIVE_S so the 9P transport doesn't time
        # out and surface EIO to clients on quiet feeds. Returning b""
        # is a valid 9P Rread payload meaning "no data right now"; the
        # tailer's existing reconnect path handles it cleanly.
        try:
            return await asyncio.wait_for(
                self._daemon.gossip.read_feed(
                    fid._feed_cursor,
                    max_bytes=count,
                    identity=self._daemon.identity_vector,
                    block=True,
                ),
                timeout=_BLOCKING_READ_KEEPALIVE_S,
            )
        except asyncio.TimeoutError:
            return b""


class FeedRecentFile(_ReadOnly, SyntheticFile):
    """
    /n/peribus/feed/recent — snapshot of the current feed buffer.

    Same content as feed/new, but returns EOF (empty bytes) once the
    current buffer is exhausted instead of blocking. Use this for
    one-shot reads:

        cat /n/peribus/feed/recent           # dumps current feed and exits

    For live tailing, use feed/new (cat will block until next post arrives).
    """

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("recent", parent)
        self._mode = 0o444
        self._daemon = daemon

    def _get_length(self) -> int:
        return 0

    async def read(self, fid, offset: int, count: int) -> bytes:
        if not hasattr(fid, "_feed_cursor"):
            fid._feed_cursor = self._daemon.gossip.feed_cursor()
        return await self._daemon.gossip.read_feed(
            fid._feed_cursor,
            max_bytes=count,
            identity=self._daemon.identity_vector,
            block=False,
        )


class InboxFile(_ReadOnly, SyntheticFile):
    """
    /n/peribus/inbox/new — tail-able stream of received DMs.

    Each line is one JSON message: {"from": "...", "ts": ..., "body": "..."}.
    Reads block when there's no new mail, exactly like feed/new — perfect
    for `tail -f`-style consumers and live notification widgets.

    For a one-shot snapshot use inbox/recent.
    """

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("new", parent)
        self._mode = 0o444
        self._daemon = daemon

    def _get_length(self) -> int:
        # Streaming file — length is meaningless. Same convention as feed/new.
        return 0

    async def read(self, fid, offset: int, count: int) -> bytes:
        # Each fid carries its own cursor.
        if not hasattr(fid, "_inbox_cursor"):
            fid._inbox_cursor = self._daemon.inbox.cursor(from_start=True)
        # Same keepalive pattern as FeedNewFile.read() — return b"" on
        # timeout so the 9P transport never sees a request stalled past
        # its limit and surfaces EIO to the client.
        try:
            return await asyncio.wait_for(
                self._daemon.inbox.read(
                    fid._inbox_cursor, max_bytes=count, block=True,
                ),
                timeout=_BLOCKING_READ_KEEPALIVE_S,
            )
        except asyncio.TimeoutError:
            return b""


class InboxRecentFile(_ReadOnly, SyntheticFile):
    """
    /n/peribus/inbox/recent — snapshot of received DMs.

    Returns the current inbox buffer and EOFs when drained. The blocking
    counterpart is /n/peribus/inbox/new (see InboxFile).
    """

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("recent", parent)
        self._mode = 0o444
        self._daemon = daemon

    def _get_length(self) -> int:
        return 0

    async def read(self, fid, offset: int, count: int) -> bytes:
        if not hasattr(fid, "_inbox_cursor"):
            fid._inbox_cursor = self._daemon.inbox.cursor(from_start=True)
        return await self._daemon.inbox.read(
            fid._inbox_cursor, max_bytes=count, block=False,
        )


class InboxSendFile(_WriteOnly, SyntheticFile):
    """
    /n/peribus/inbox/send — write "<nodeid> <body>" to send a DM.

    Each write is parsed as a single message: the part up to the first
    whitespace is the recipient nodeid, and the rest is the message body
    (stripped). Trailing newline is fine. Multi-line bodies are supported
    if you write the whole thing in one syscall.

    Examples:
        echo "napr447jevdw5i44sovdoiflqq hi from me" > /n/peribus/inbox/send
        printf '%s %s' nodeid "$(cat note.txt)" > /n/peribus/inbox/send

    Equivalent to writing to /n/peribus/nodes/<nodeid>/inbox, but lets
    you address messages without first walking into a peer's directory
    (useful for short-lived agents and for senders who only have the
    nodeid as a string).
    """

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("send", parent)
        self._mode = 0o222   # write-only
        self._daemon = daemon

    def _get_length(self) -> int:
        return 0

    async def write(self, fid, offset: int, data: bytes) -> int:
        text = data.decode("utf-8", errors="replace")
        # Split on the first whitespace run only. We don't want to treat
        # body content as more arguments — the rest is one message.
        head, sep, body = text.partition(" ")
        if not sep:
            head, sep, body = text.partition("\t")
        if not sep:
            head, sep, body = text.partition("\n")
        nodeid = head.strip()
        body = body.strip()
        if not nodeid or not body:
            # Treat as no-op rather than raising — write semantics on a
            # control file should be forgiving for partial echoes.
            return len(data)
        await self._daemon.send_message(nodeid, body)
        return len(data)


class PeerFromFile(_ReadOnly, SyntheticFile):
    """
    /n/peribus/nodes/<nodeid>/from — DMs received specifically from this peer.

    Snapshot read; returns the per-peer message history as JSON lines and
    EOFs. (For a live tail of all DMs, use /n/peribus/inbox.)
    """

    def __init__(self, peer_nodeid: str, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("from", parent)
        self._mode = 0o444
        self._peer = peer_nodeid
        self._daemon = daemon

    def _get_length(self) -> int:
        return 0

    async def read(self, fid, offset: int, count: int) -> bytes:
        msgs = self._daemon.inbox.from_peer(self._peer)
        # Render as JSON lines, then slice by offset/count for paged reads.
        if not hasattr(fid, "_from_buffer"):
            buf = bytearray()
            for m in msgs:
                buf.extend(m.to_json_line())
            fid._from_buffer = bytes(buf)
        return fid._from_buffer[offset:offset + count]


class StaticFile(_ReadOnly, SyntheticFile):
    """A simple file with fixed bytes (used for materialized peer posts)."""

    def __init__(self, name: str, content: bytes, parent: SyntheticDir = None):
        super().__init__(name, parent)
        self._mode = 0o444
        self._content = content

    def _get_length(self) -> int:
        return len(self._content)

    async def read(self, fid, offset: int, count: int) -> bytes:
        return self._content[offset:offset + count]


class SharedItemFile(SyntheticFile):
    """
    A "raw" share file. DEPRECATED — kept for backwards compatibility with
    older clients (and ad-hoc shell uses) that drop a complete JSON envelope
    directly into share/<name>. New clients should use share/clone instead,
    which gives them a draft directory with separate body/kind/reply_to/
    attach files that compose cleanly under shell pipelines.

    When the fid is clunked, the buffered bytes are passed straight to the
    daemon's legacy publish_share path. That path treats whatever you wrote
    as the post payload — JSON envelope or raw bytes, the receiver-side
    widget figures it out.

    Removing this class entirely is a future cleanup; while it remains, it
    serves as the wire-compat shim for any tool that still does
    `cp foo.txt /n/peribus/share/foo.txt`.
    """

    def __init__(self, name: str, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__(name, parent)
        self._mode = 0o644
        self._daemon = daemon
        self._buffer = bytearray()
        self._published = False

    def _get_length(self) -> int:
        return len(self._buffer)

    async def read(self, fid, offset: int, count: int) -> bytes:
        return bytes(self._buffer[offset:offset + count])

    async def write(self, fid, offset: int, data: bytes) -> int:
        end = offset + len(data)
        if end > len(self._buffer):
            self._buffer.extend(b"\x00" * (end - len(self._buffer)))
        self._buffer[offset:end] = data
        self.touch()
        return len(data)

    async def clunk(self, fid):
        if self._published or not self._buffer:
            return
        self._published = True
        try:
            await self._daemon.publish_share(self.name, bytes(self._buffer))
        except Exception:
            import logging, traceback
            logging.getLogger(__name__).warning(
                "publish_share failed for %s:\n%s", self.name, traceback.format_exc()
            )
            raise


# ---------------------------------------------------------------------------
# Clone-style draft directory
#
# The new publishing surface. Replaces the "write a JSON envelope to
# share/<name>" model with a per-attribute file layout that composes
# under shell pipelines:
#
#     n=$(cat /n/peribus/share/clone)
#     (cat file1.py; echo; cat file2.py) > /n/peribus/share/$n/body
#     echo b3:abc123 > /n/peribus/share/$n/reply_to
#     echo ~/Pictures/cat.png > /n/peribus/share/$n/attach
#     echo publish > /n/peribus/share/$n/ctl
#     cat /n/peribus/share/$n/result
#
# The daemon doesn't have to grow new state to support this — the draft
# dir is local-only scaffolding that, on `publish`, produces exactly the
# same shape of payload publish_share already accepts. What changes is
# the user-facing contract: the syntax is structural, not textual.
#
# Lifetime: drafts are kept around after publish so `result` stays
# readable for late readers, but a daemon-side reaper (DraftReaper) GCs
# drafts older than _DRAFT_TTL_S whose state is finalized (published or
# discarded). In-progress drafts are kept indefinitely.
# ---------------------------------------------------------------------------


class _DraftAttrFile(SyntheticFile):
    """
    Single-line attribute file backed by a string on the parent draft.

    Reads return the current value (with a trailing newline if non-empty
    and not already terminated). Writes replace the value, or append when
    `append=True`. Truncate-to-zero on open is honored for the
    overwrite-style `echo … > body` case via clear_on_next_write.
    """

    def __init__(
        self,
        name: str,
        draft: "DraftDir",
        attr: str,
        *,
        append: bool = False,
        validator=None,
        parent: SyntheticDir = None,
    ):
        super().__init__(name, parent)
        self._mode = 0o666
        self._draft = draft
        self._attr = attr
        self._append = append
        self._validator = validator

    def _value(self) -> str:
        return getattr(self._draft, self._attr) or ""

    def _bytes(self) -> bytes:
        v = self._value()
        if not v:
            return b""
        if not v.endswith("\n"):
            v = v + "\n"
        return v.encode("utf-8")

    def _get_length(self) -> int:
        return len(self._bytes())

    async def read(self, fid, offset: int, count: int) -> bytes:
        data = self._bytes()
        return data[offset:offset + count]

    async def write(self, fid, offset: int, data: bytes) -> int:
        if self._draft.is_finalized():
            raise PermissionError(
                f"{self.name}: draft already {self._draft.state}"
            )

        chunk = data.decode("utf-8", errors="replace")

        if self._append:
            # Multi-write append: the body file uses this so that
            # `(cat a; cat b) > body` composes. Each successive write
            # extends the value. Offset is honored as a hint but not
            # enforced — append-mode files don't usefully support
            # random writes.
            current = self._value()
            new_value = current + chunk
        else:
            # Overwrite mode: every write replaces the value. Offsets
            # past 0 are tolerated for simple shells but treated as a
            # full replace (consistent with the line-based control
            # files elsewhere in this tree).
            new_value = chunk

        if self._validator is not None:
            try:
                new_value = self._validator(new_value)
            except ValueError as e:
                raise PermissionError(f"{self.name}: {e}")

        setattr(self._draft, self._attr, new_value)
        self._draft.touch()
        self.touch()
        return len(data)


class _DraftBodyFile(_DraftAttrFile):
    """
    Body file with truncate-on-create semantics.

    Open-with-truncate (mode O_TRUNC, the standard `echo … > body` shell
    primitive) clears the buffer; subsequent writes from the same fid
    append. This is what makes `echo "hello" > body` overwrite cleanly
    while `cat a >> body; cat b >> body` accumulates. The 9P server
    surfaces O_TRUNC by calling `truncate(0)` on the file before the
    first write; we hook that to reset the underlying string.
    """

    def __init__(self, draft: "DraftDir", parent: SyntheticDir = None):
        super().__init__("body", draft, "body", append=True, parent=parent)

    async def truncate(self, length: int) -> None:
        if self._draft.is_finalized():
            raise PermissionError(f"body: draft already {self._draft.state}")
        if length == 0:
            self._draft.body = ""
        else:
            cur = self._draft.body or ""
            if length < len(cur):
                self._draft.body = cur[:length]
            elif length > len(cur):
                self._draft.body = cur + "\x00" * (length - len(cur))
        self._draft.touch()
        self.touch()


class _DraftAttachFile(SyntheticFile):
    """
    `attach` — line-oriented queue of paths to attach.

    Each newline-terminated chunk is one attachment path. Multiple writes
    accumulate. Reading returns the current queue, one path per line, in
    insertion order. Truncate-to-zero clears the queue.
    """

    def __init__(self, draft: "DraftDir", parent: SyntheticDir = None):
        super().__init__("attach", parent)
        self._mode = 0o666
        self._draft = draft

    def _bytes(self) -> bytes:
        if not self._draft.attachments:
            return b""
        return ("\n".join(self._draft.attachments) + "\n").encode("utf-8")

    def _get_length(self) -> int:
        return len(self._bytes())

    async def read(self, fid, offset: int, count: int) -> bytes:
        data = self._bytes()
        return data[offset:offset + count]

    async def write(self, fid, offset: int, data: bytes) -> int:
        if self._draft.is_finalized():
            raise PermissionError(
                f"attach: draft already {self._draft.state}"
            )

        # `attach` is a list of paths, one per line. We deliberately
        # require strict UTF-8 here (no errors="replace") because the
        # most common mistake is `cat binary-file > attach` — which
        # streams the file's bytes into this control file instead of
        # the path. Without this guard, those random bytes get split
        # on stray \n characters, become garbage "paths", and produce
        # cryptic publish-time errors (e.g. expanduser blowing up on
        # a binary fragment that happens to start with ~).
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise PermissionError(
                "attach: input is not valid UTF-8 — `attach` takes file "
                "PATHS, one per line, not file contents. Use "
                "`echo /path/to/file > attach` (or "
                "`echo /path/to/file >> attach` to add more)."
            )

        # Also catch the "user piped a path with embedded NULs or
        # control bytes" case before it reaches publish_draft. NUL is
        # never legal in a POSIX path; treating it as a soft signal
        # that the user meant `cat`, not `echo`.
        if "\x00" in text:
            raise PermissionError(
                "attach: path contains NUL bytes — did you mean "
                "`echo /path/to/file > attach`?"
            )

        added = 0
        for line in text.splitlines():
            p = line.strip()
            if p:
                self._draft.attachments.append(p)
                added += 1
        if added:
            self._draft.touch()
            self.touch()
        return len(data)

    async def truncate(self, length: int) -> None:
        if self._draft.is_finalized():
            raise PermissionError(f"attach: draft already {self._draft.state}")
        if length == 0:
            self._draft.attachments.clear()
            self._draft.touch()
            self.touch()


class _DraftCtlFile(SyntheticFile):
    """
    `ctl` — write "publish" to commit the draft, "discard" to drop it.

    The write blocks until the action completes. For "publish" this means
    the daemon has stored the post, signed it, and broadcast it; the post
    hash is then available via the sibling `result` file. For "discard"
    the draft is marked dropped; subsequent attribute writes raise.

    Reading `ctl` returns the draft's current state ("draft", "publishing",
    "published", "discarded", or "error: <msg>") on one line.
    """

    def __init__(self, draft: "DraftDir", parent: SyntheticDir = None):
        super().__init__("ctl", parent)
        self._mode = 0o666
        self._draft = draft

    def _bytes(self) -> bytes:
        return (self._draft.state + "\n").encode("utf-8")

    def _get_length(self) -> int:
        return len(self._bytes())

    async def read(self, fid, offset: int, count: int) -> bytes:
        data = self._bytes()
        return data[offset:offset + count]

    async def write(self, fid, offset: int, data: bytes) -> int:
        cmd = data.decode("utf-8", errors="replace").strip().lower()
        if cmd == "publish":
            await self._draft.publish()
        elif cmd == "discard":
            await self._draft.discard()
        else:
            raise PermissionError(
                f"ctl: unknown command {cmd!r} (expected 'publish' or 'discard')"
            )
        self.touch()
        return len(data)


class _DraftResultFile(_ReadOnly, SyntheticFile):
    """
    `result` — read the draft's outcome.

    Reading blocks until the draft is finalized (published or discarded).
    Then returns either:
      - the post hash + newline, on success
      - "error: <message>\\n" on failure
      - "discarded\\n" if the draft was dropped

    Once the draft finalizes, all subsequent reads return immediately
    with the cached result, so this works for late readers too.
    """

    def __init__(self, draft: "DraftDir", parent: SyntheticDir = None):
        super().__init__("result", parent)
        self._mode = 0o444
        self._draft = draft

    def _get_length(self) -> int:
        # Length is unknown until publish completes. Over-report — 9P
        # clients tolerate this and just stop reading at EOF.
        return 256

    async def read(self, fid, offset: int, count: int) -> bytes:
        # Wait for finalization on the first read of this fid.
        await self._draft.wait_finalized()
        result = self._draft.result_text()
        data = result.encode("utf-8")
        return data[offset:offset + count]


class DraftDir(SyntheticDir):
    """
    One in-progress draft. share/<n>/ on the synthetic tree.

    Holds the user-authored attributes (body, kind, reply_to, attachments)
    and orchestrates the publish step that funnels them into the daemon's
    existing publish path. State machine:

        draft  →  publishing  →  published     (success)
        draft  →  publishing  →  error:<msg>   (publish raised)
        draft  →  discarded                    (user cancelled)

    Once finalized, attribute writes raise PermissionError. The directory
    itself stays browsable so `result` is still readable, until DraftReaper
    sweeps it.
    """

    _VALID_KINDS = ("post", "reply")

    def __init__(
        self,
        n: int,
        daemon: "PeribusDaemon",
        parent: SyntheticDir = None,
    ):
        super().__init__(str(n), parent)
        self._daemon = daemon
        self.n = n
        self.created_at = 0.0  # set below via time.time()

        # Attributes the user fills in.
        self.body: str = ""
        self.kind: str = "post"  # default — flipped to "reply" automatically
                                  # if reply_to is set and kind wasn't
        self.reply_to: str = ""
        self.attachments: list[str] = []

        # Lifecycle.
        self.state: str = "draft"
        self._post_hash: Optional[str] = None
        self._error: Optional[str] = None
        self._final_event = None  # asyncio.Event, lazy-init in publish()
        self.finalized_at: Optional[float] = None

        # Wire up children.
        self.children["body"] = _DraftBodyFile(self, parent=self)
        self.children["kind"] = _DraftAttrFile(
            "kind", self, "kind",
            validator=self._validate_kind, parent=self,
        )
        self.children["reply_to"] = _DraftAttrFile(
            "reply_to", self, "reply_to",
            validator=self._validate_reply_to, parent=self,
        )
        self.children["attach"] = _DraftAttachFile(self, parent=self)
        self.children["ctl"] = _DraftCtlFile(self, parent=self)
        self.children["result"] = _DraftResultFile(self, parent=self)

        import time as _time
        self.created_at = _time.time()

    # ---- attribute validation -----

    def _validate_kind(self, value: str) -> str:
        v = value.strip().lower()
        if not v:
            return ""
        if v not in self._VALID_KINDS:
            raise ValueError(
                f"kind must be one of {self._VALID_KINDS}, got {v!r}"
            )
        return v

    def _validate_reply_to(self, value: str) -> str:
        v = value.strip()
        # Empty is fine — clears the reply target.
        if not v:
            return ""
        # The hash format on the wire is "b3:..." (see gossip._content_hash).
        # Be liberal here: accept any non-whitespace token; the daemon's
        # publish path will fail loudly if it's truly malformed.
        if any(ws in v for ws in (" ", "\t", "\n")):
            raise ValueError("reply_to must be a single hash token")
        return v

    # ---- lifecycle -----

    def is_finalized(self) -> bool:
        return self.state in ("published", "discarded") or self.state.startswith("error:")

    def _ensure_event(self):
        import asyncio as _asyncio
        if self._final_event is None:
            self._final_event = _asyncio.Event()
            if self.is_finalized():
                self._final_event.set()
        return self._final_event

    async def wait_finalized(self) -> None:
        await self._ensure_event().wait()

    def result_text(self) -> str:
        if self.state == "published" and self._post_hash:
            return self._post_hash + "\n"
        if self.state == "discarded":
            return "discarded\n"
        if self._error:
            return f"error: {self._error}\n"
        # Shouldn't be reached if wait_finalized was awaited.
        return self.state + "\n"

    async def discard(self) -> None:
        if self.is_finalized():
            return  # idempotent — discarding an already-discarded draft is a no-op
        self.state = "discarded"
        import time as _time
        self.finalized_at = _time.time()
        self._ensure_event().set()
        self.touch()

    async def publish(self) -> None:
        """Materialize this draft into a real post via the daemon."""
        if self.is_finalized():
            raise PermissionError(f"draft already {self.state}")
        if self.state == "publishing":
            raise PermissionError("publish already in progress")

        self.state = "publishing"
        self.touch()
        try:
            # Default kind inference: if reply_to is set and kind is the
            # default "post", treat it as a reply automatically. This is
            # convenience — explicit `echo reply > kind` still wins.
            kind = self.kind or "post"
            if kind == "post" and self.reply_to:
                kind = "reply"

            h = await self._daemon.publish_draft(
                body=self.body,
                kind=kind,
                reply_to=self.reply_to,
                attachment_paths=list(self.attachments),
            )
            self._post_hash = h
            self.state = "published"
        except Exception as e:
            import logging, traceback
            logging.getLogger(__name__).warning(
                "publish_draft failed for share/%d:\n%s",
                self.n, traceback.format_exc(),
            )
            self._error = str(e) or e.__class__.__name__
            self.state = f"error: {self._error}"
            # Don't re-raise: the user's `ctl` write succeeded as a
            # control action; the failure is observable via result.
        finally:
            import time as _time
            self.finalized_at = _time.time()
            self._ensure_event().set()
            self.touch()


class _CloneFile(_ReadOnly, SyntheticFile):
    """
    `share/clone` — read-once allocator for new draft directories.

    Each read returns a fresh decimal number. Concurrently, share/<n>/
    is materialized in the parent ShareDir's children. Reads are
    idempotent per-fid: a single `cat` on clone returns one number even
    if the kernel issues multiple read syscalls under the hood.
    """

    def __init__(self, share_dir: "ShareDir", parent: SyntheticDir = None):
        super().__init__("clone", parent)
        self._mode = 0o444
        self._share_dir = share_dir

    def _get_length(self) -> int:
        # Indeterminate; over-report so cat doesn't EOF early.
        return 64

    async def read(self, fid, offset: int, count: int) -> bytes:
        # Allocate at most once per fid. Subsequent reads on the same
        # fid just walk the cached buffer (so `cat` works cleanly even
        # if it issues two read syscalls).
        if not hasattr(fid, "_clone_buffer"):
            n = self._share_dir.allocate_draft()
            fid._clone_buffer = (str(n) + "\n").encode("utf-8")
        return fid._clone_buffer[offset:offset + count]


class ShareDir(SyntheticDir):
    """
    /n/peribus/share — clone-style draft directory.

    Layout:
        clone        — read to allocate a new draft. Returns "<n>\\n".
        <n>/         — one draft (see DraftDir).
        <legacy>     — DEPRECATED: a JSON envelope dropped here as a
                       single file is still publish_share()'d on clunk,
                       so old clients keep working during the transition.

    The legacy single-file path lives as a fallback in `create()`. Any
    name that's a pure decimal integer is reserved for clone allocation,
    so legacy creators will never collide with new drafts.
    """

    _DRAFT_TTL_S = 600.0  # 10 minutes after finalize, drafts get reaped

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("share", parent)
        self._daemon = daemon
        self._next_n: int = 1
        self.children["clone"] = _CloneFile(self, parent=self)

    def allocate_draft(self) -> int:
        """Mint a new draft directory and return its number."""
        # Skip past any number whose slot is already populated (shouldn't
        # happen since we own the counter, but be defensive).
        while str(self._next_n) in self.children:
            self._next_n += 1
        n = self._next_n
        self._next_n += 1
        d = DraftDir(n, self._daemon, parent=self)
        self.children[str(n)] = d
        self.touch()
        return n

    def reap_old_drafts(self, now: float) -> int:
        """
        Remove finalized drafts whose finalize time is older than the TTL.
        Returns the count removed. Called periodically by DraftReaper.
        """
        cutoff = now - self._DRAFT_TTL_S
        victims = []
        for name, child in self.children.items():
            if not isinstance(child, DraftDir):
                continue
            if not child.is_finalized():
                continue
            if child.finalized_at is None:
                continue
            if child.finalized_at < cutoff:
                victims.append(name)
        for name in victims:
            del self.children[name]
        if victims:
            self.touch()
        return len(victims)

    async def lookup(self, name: str) -> Optional[SyntheticFile]:
        return self.children.get(name)

    async def create(self, fid_state, name: str, perm: int, mode: int):
        """
        Legacy raw-envelope creation: a client writes a JSON envelope to
        share/<name> and the daemon publishes the bytes on clunk. Kept
        for backwards compatibility. New clients should read share/clone
        and write into the resulting share/<n>/ subtree instead.

        Reserved names: pure decimal integers (collision with clone
        allocations), and "clone" itself.
        """
        from core.types import FidState

        if name == "clone" or name.isdigit():
            raise PermissionError(
                f"share/{name}: reserved name (use share/clone to allocate a draft)"
            )
        if name in self.children:
            raise FileExistsError(f"{name}: already exists in share/")

        f = SharedItemFile(name, self._daemon, parent=self)
        self.children[name] = f
        self.touch()

        return FidState(
            fid=fid_state.fid,
            path=f.path,
            qid=f.qid,
            file=f,
            opened=True,
            mode=mode,
        )


class PeerSignalFile(_ReadOnly, SyntheticFile):
    """Per-peer signal file: strength, latency, last_seen, resonance."""

    def __init__(self, peer_nodeid: str, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("signal", parent)
        self._mode = 0o444
        self._peer = peer_nodeid
        self._daemon = daemon

    def _get_length(self) -> int:
        return 128

    async def read(self, fid, offset: int, count: int) -> bytes:
        info = self._daemon.peer_signal(self._peer)
        text = (
            f"strength: {info.get('strength', 0.0):.2f}\n"
            f"latency_ms: {info.get('latency_ms', -1)}\n"
            f"last_seen: {info.get('last_seen', 0)}\n"
            f"resonance: {info.get('resonance', 0.0):.3f}\n"
        )
        data = text.encode("utf-8")
        return data[offset:offset + count]


class PeerInboxFile(_WriteOnly, SyntheticFile):
    """Write here to send the peer a message."""

    def __init__(self, peer_nodeid: str, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("inbox", parent)
        self._mode = 0o222  # write-only
        self._peer = peer_nodeid
        self._daemon = daemon

    def _get_length(self) -> int:
        return 0

    async def write(self, fid, offset: int, data: bytes) -> int:
        text = data.decode("utf-8", errors="replace")
        await self._daemon.send_message(self._peer, text)
        return len(data)


# ---------------------------------------------------------------------------
# Directory subclasses — base SyntheticDir already wires up qid/mode/children.
# ---------------------------------------------------------------------------


class PeerSocialDir(SyntheticDir):
    """A peer's public posts, lazily fetched on first lookup of each name."""

    def __init__(self, peer_nodeid: str, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("social", parent)
        self._peer = peer_nodeid
        self._daemon = daemon

    async def lookup(self, name: str) -> Optional[SyntheticFile]:
        if name in self.children:
            return self.children[name]
        post = await self._daemon.fetch_peer_post(self._peer, name)
        if post is None:
            return None
        f = StaticFile(name, post, parent=self)
        self.children[name] = f
        return f


class PeerDir(SyntheticDir):
    """Container for one peer's exposed surface — signal, social, inbox, from."""

    def __init__(self, peer_nodeid: str, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__(peer_nodeid, parent)
        self._peer = peer_nodeid
        self._daemon = daemon

        # Build child files. They thread `parent=self` so paths render right.
        self.children["signal"] = PeerSignalFile(peer_nodeid, daemon, parent=self)
        self.children["social"] = PeerSocialDir(peer_nodeid, daemon, parent=self)
        self.children["inbox"] = PeerInboxFile(peer_nodeid, daemon, parent=self)
        # Read this to see DMs we've received from this specific peer.
        # (For all DMs across all peers, use /n/peribus/inbox/new or recent.)
        self.children["from"] = PeerFromFile(peer_nodeid, daemon, parent=self)


class NodesDir(SyntheticDir):
    """/n/peribus/nodes/ — dynamic, populated as peers are discovered."""

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("nodes", parent)
        self._daemon = daemon

    async def lookup(self, name: str) -> Optional[SyntheticFile]:
        if name in self.children:
            return self.children[name]
        # Recognize our own NodeID. This makes the local case symmetric
        # with the remote case: a post we just published lists *us* as
        # author, and the receiver-side render path walks
        # nodes/<author>/social/<hash>. Without this, locally-published
        # posts can never resolve their own attachments — the chip
        # shows but the bytes never load. (PeerSocialDir's lookup
        # ultimately calls fetch_peer_post, whose local-content fast
        # path returns the bytes from gossip._content regardless of
        # whether the author is us or a peer.)
        own_nodeid = getattr(self._daemon.identity, "nodeid", None)
        if name == own_nodeid:
            d = PeerDir(name, self._daemon, parent=self)
            self.children[name] = d
            return d
        # On-demand: if the daemon knows this peer, materialize a PeerDir.
        if self._daemon.knows_peer(name):
            d = PeerDir(name, self._daemon, parent=self)
            self.children[name] = d
            return d
        return None

    def add_peer(self, nodeid: str) -> "PeerDir":
        """Daemon calls this when a peer is discovered, so it shows in `ls`."""
        if nodeid in self.children:
            return self.children[nodeid]
        d = PeerDir(nodeid, self._daemon, parent=self)
        self.children[nodeid] = d
        self.touch()
        return d

    def remove_peer(self, nodeid: str) -> None:
        """Peer dropped — withers the corresponding directory."""
        if nodeid in self.children:
            del self.children[nodeid]
            self.touch()


# ---------------------------------------------------------------------------
# Content-addressed surface
# ---------------------------------------------------------------------------


class CasDir(SyntheticDir):
    """
    /n/peribus/cas/ — content-addressed access to the gossip blob store.

    Walking cas/<hash> returns the raw bytes if they're in the local
    daemon's content cache. This is useful in three places:

      1. As a fallback for attachment resolution: a receiver-side
         renderer can ask for a blob by hash without having to know
         which peer authored it. The widget's _ATTACHMENT_PATH_TEMPLATES
         already lists `{root}/cas/{hash}` as one of its candidates;
         this directory makes that template real.

      2. For locally-published posts: until cas/ existed, walking
         nodes/<self_nodeid>/social/<hash> was the only way to reach
         the bytes, and that path also works now (NodesDir was patched
         to recognize the self-NodeID). cas/ gives a peer-agnostic
         alternative — handy for blobs that arrived via gossip with
         no obvious "author" relationship.

      3. For debugging and ad-hoc tools: `cat /n/peribus/cas/<hash>`
         is a one-liner for any blob the daemon has seen. Nice to have.

    Listing cas/ returns nothing — the content store can be huge and
    we don't want a stat() to enumerate every blob. Lookups are
    direct: cas/<hash> either resolves or doesn't.
    """

    def __init__(self, daemon: "PeribusDaemon", parent: SyntheticDir = None):
        super().__init__("cas", parent)
        self._daemon = daemon

    async def lookup(self, name: str) -> Optional[SyntheticFile]:
        # Don't cache blob files in self.children — the content store
        # is large, and these are leaf reads. A fresh StaticFile per
        # lookup is fine; the Server9P fid table holds it as long as
        # the client keeps the fid open.
        data = self._daemon.gossip.get_content(name)
        if data is None:
            return None
        return StaticFile(name, data, parent=self)


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------

def build_peribus_tree(daemon: "PeribusDaemon") -> SyntheticDir:
    """
    Construct the /n/peribus root and all its child dirs/files. Returns
    the root, which gets handed to Server9P.

    The root is named "peribus" and has no parent — it gets attached at
    the mount point by whatever client mounts it (typically /n/peribus).
    """
    root = SyntheticDir("peribus", parent=None)

    # ctl
    root.children["ctl"] = CtlFile(daemon, parent=root)

    # bootstrap — printable NODEID@host:port for someone joining our DHT.
    # Empty until the global-discovery glue registers a "bootstrap" info
    # provider. Lives at the root rather than under identity/ because it's
    # a network address, not an identity attribute.
    root.children["bootstrap"] = InfoFile(
        "bootstrap", "bootstrap", daemon, parent=root,
    )

    # identity/
    identity = SyntheticDir("identity", parent=root)
    identity.children["nodeid"] = NodeIdFile(daemon, parent=identity)
    identity.children["vector"] = IdentityVectorFile(daemon, parent=identity)
    identity.children["summary"] = SummaryFile(daemon, parent=identity)
    root.children["identity"] = identity

    # nodes/  — daemon hangs onto this so it can add/remove peers
    nodes = NodesDir(daemon, parent=root)
    root.children["nodes"] = nodes
    daemon.nodes_dir = nodes

    # feed/
    feed = SyntheticDir("feed", parent=root)
    feed.children["new"] = FeedNewFile(daemon, parent=feed)
    feed.children["recent"] = FeedRecentFile(daemon, parent=feed)
    root.children["feed"] = feed

    # inbox/  — direct messages received from peers
    #   inbox/new      blocking tail, JSON-line per message
    #   inbox/recent   snapshot, EOFs once drained
    #   inbox/send     write "<nodeid> <body>" to send a DM (one per write)
    inbox = SyntheticDir("inbox", parent=root)
    inbox.children["new"] = InboxFile(daemon, parent=inbox)
    inbox.children["recent"] = InboxRecentFile(daemon, parent=inbox)
    inbox.children["send"] = InboxSendFile(daemon, parent=inbox)
    root.children["inbox"] = inbox

    # share/
    root.children["share"] = ShareDir(daemon, parent=root)

    # cas/  — content-addressed access to the gossip blob store.
    # `cat /n/peribus/cas/<hash>` returns the raw bytes if locally
    # cached. Useful as an attachment-resolution fallback and for
    # tooling that wants to fetch a blob without knowing its author.
    root.children["cas"] = CasDir(daemon, parent=root)

    # streams/  — populated by widget runtime when widgets request streams
    streams = SyntheticDir("streams", parent=root)
    root.children["streams"] = streams
    daemon.streams_dir = streams

    # apps/  — filesystem surface for the swarm-search layer
    #
    #   echo "calculator" > apps/search   # fan out, gather, block
    #   cat apps/search                   # ranked hits as JSON lines
    #   echo "<app_id>" > apps/install    # fetch + validate + index
    #   cat apps/install                  # "ok" / "error: ..."
    #   cat apps/installed                # local corpus listing
    #   cat apps/<app_id>/source          # the validated .py source
    #
    # Same shape on every machine — Machine B searches, swarm fans out
    # to Machine A's local /n/llm/embed, hits return, source flows back
    # via MSG_FETCH, B's local embedder indexes the new app.
    from peribus._content import AppsDir
    root.children["apps"] = AppsDir(daemon, parent=root)

    return root

# ============================================================================
# daemon.py
# ----------------------------------------------------------------------------
"""
peribus.daemon — peribusd, the spider that ties the web together

The daemon owns:
  * The cryptographic identity (Identity)
  * The local embedder + identity vector
  * The synthetic /n/peribus filesystem (a SyntheticDir tree)
  * A discovery backend (mDNS now, libp2p later)
  * A wire server speaking peribus/0.1 with peers
  * The gossip mesh (content store + feed buffer)

It exposes the synthetic root via Server9P (the same 9P implementation rio
uses for its scene), so mounting /n/peribus is no different from mounting
/n/llm or /n/<machine>.

The daemon is a single asyncio loop. All callbacks from discovery (which
runs zeroconf threads) are marshaled back via run_coroutine_threadsafe.
"""


import asyncio
import base64
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from peribus._foundation import Identity, nodeid_from_pubkey, verify_signature
from peribus._foundation import (
    Embedder,
    HashEmbedder,
    IdentityVector,
    cosine,
    load_embedder,
    make_sketch,
    pack_vector,
)
from core.files import SyntheticDir
from peribus._daemon import build_peribus_tree, NodesDir
from peribus._discovery import Discovery, MdnsDiscovery, LocalAnnouncement, PeerInfo
from peribus._content import GossipMesh, Post, _content_hash
from peribus._transport import (
    WireServer,
    WireConn,
    MSG_HELLO,
    MSG_ANNOUNCE,
    MSG_POST,
    MSG_FETCH,
    MSG_DATA,
    MSG_MSG,
    MSG_PING,
    MSG_PONG,
)
from peribus._content import EmbedSearch
from peribus._content import AppSwarm

logger = logging.getLogger(__name__)


@dataclass
class PeerState:
    """What we know about a peer beyond what discovery gave us."""
    info: PeerInfo
    last_seen: float
    resonance: float = 0.0       # cosine to our identity at last announce
    inbox: List["Message"] = None  # messages we've received from them

    def __post_init__(self):
        if self.inbox is None:
            self.inbox = []


@dataclass
class Message:
    """A direct message received from a peer."""
    sender: str          # nodeid of the author
    body: str            # the message text
    ts: float            # unix seconds when we received it

    def to_json_line(self) -> bytes:
        """Serialize as one JSON line, suitable for tail-style consumption."""
        import json
        line = json.dumps({
            "from": self.sender,
            "ts": self.ts,
            "body": self.body,
        })
        return (line + "\n").encode("utf-8")


class InboxStore:
    """
    Daemon-level inbox for received DMs.

    Two views of the same data:
      * a flat list (newest last), exposed via /n/peribus/inbox
      * per-peer lists, exposed via /n/peribus/nodes/<nodeid>/from

    Reads on /inbox block when there's nothing new (like feed/new); reads
    on /inbox/recent return the buffer immediately (like feed/recent).
    """

    MAX_MESSAGES = 4096   # flat ring cap

    def __init__(self):
        self._messages: List[Message] = []
        # Asyncio Futures held by readers blocked waiting for new mail.
        self._waiters: List[asyncio.Future] = []
        self._lock = asyncio.Lock()

    async def add(self, msg: Message) -> None:
        """Append a message and wake any waiting readers."""
        async with self._lock:
            self._messages.append(msg)
            if len(self._messages) > self.MAX_MESSAGES:
                self._messages.pop(0)
            for w in self._waiters:
                if not w.done():
                    w.set_result(None)
            self._waiters.clear()

    def cursor(self, from_start: bool = True) -> "InboxCursor":
        """Hand out a fresh cursor.

        from_start=True: cursor begins before all existing messages, so
            the first read returns whatever's in the buffer right now,
            then blocks. This is what /inbox uses.
        from_start=False: cursor begins past the end; the first read
            blocks until something new arrives.
        """
        return InboxCursor(
            last_index=-1 if from_start else len(self._messages) - 1
        )

    async def read(
        self, cursor: "InboxCursor", max_bytes: int, block: bool = True,
    ) -> bytes:
        """Return up to max_bytes from this cursor. Blocks if nothing new
        and block=True; returns b"" otherwise (EOF for `cat`)."""
        if cursor.pending:
            chunk = cursor.pending[:max_bytes]
            cursor.pending = cursor.pending[max_bytes:]
            return chunk

        if cursor.last_index >= len(self._messages) - 1:
            if not block:
                return b""
            while cursor.last_index >= len(self._messages) - 1:
                waiter = asyncio.get_running_loop().create_future()
                self._waiters.append(waiter)
                try:
                    await waiter
                finally:
                    pass

        async with self._lock:
            start = cursor.last_index + 1
            new_msgs = list(self._messages[start:])
            cursor.last_index = len(self._messages) - 1

        out = bytearray()
        for m in new_msgs:
            line = m.to_json_line()
            if len(out) + len(line) > max_bytes:
                # Stash overflow for next read.
                cursor.pending = bytes(line)
                idx = new_msgs.index(m) + 1
                for rest in new_msgs[idx:]:
                    cursor.pending += rest.to_json_line()
                break
            out.extend(line)
        return bytes(out)

    def from_peer(self, nodeid: str) -> List[Message]:
        """All messages we've received from a given nodeid (newest last)."""
        return [m for m in self._messages if m.sender == nodeid]

    def stats(self) -> Dict[str, int]:
        return {
            "total": len(self._messages),
            "senders": len({m.sender for m in self._messages}),
        }


@dataclass
class InboxCursor:
    """Per-fid position in the inbox flat ring."""
    last_index: int = -1
    pending: bytes = b""


class PeribusDaemon:
    """The whole show."""

    def __init__(
        self,
        listen_port: int = 5660,
        llm_mount: Optional[str] = "/n/llm",
        identity_dir: Optional[Path] = None,
    ):
        self.listen_port = listen_port

        # --- identity ---
        self.identity: Identity = Identity.load_or_create(identity_dir)

        # --- embedder + identity vector ---
        self.embedder: Embedder = load_embedder(llm_mount)
        self.identity_vector = IdentityVector(dim=self.embedder.dim)
        self.summary: str = ""

        # --- filesystem ---
        self.nodes_dir: Optional[NodesDir] = None     # set by build_peribus_tree
        self.streams_dir: Optional[SyntheticDir] = None  # ditto
        self.fs_root: SyntheticDir = build_peribus_tree(self)

        # --- network ---
        self.discovery: Optional[Discovery] = None
        self.wire = WireServer(
            listen_port=listen_port,
            on_message=self._on_wire_message,
            on_disconnect=self._on_wire_disconnect,
        )

        # --- gossip ---
        self.gossip = GossipMesh(lambda: self.identity_vector)

        # --- semantic search via /n/llm/embed (per-user agent) ---
        # The embed agent runs as a 9p mount; peribus is a thin client.
        # No shadow corpus — the agent's `scan` folders ARE the corpus.
        # Degrades to no-op if /n/llm/embed is absent.
        self.embed_search = EmbedSearch(
            mount=(llm_mount.rstrip("/") + "/embed") if llm_mount else "/n/llm/embed",
        )

        # --- swarm-search layer ---
        # Federates semantic search across resonance-overlay neighbors.
        # overlay_provider is a closure because the overlay only exists
        # once discovery has been swapped in (DhtDiscovery owns it); the
        # closure picks it up lazily at search time.
        from peribus._content import validate_widget_source as _validate_widget
        self.app_swarm = AppSwarm(
            nodeid=self.identity.nodeid,
            embed_search=self.embed_search,
            overlay_provider=lambda: getattr(self.discovery, "overlay", None),
            wire=self.wire,
            validate_app=lambda src: _validate_widget(
                src.decode("utf-8", errors="replace") if isinstance(src, (bytes, bytearray)) else src
            ),
            fetch_content=self._fetch_app_source,
        )

        # --- direct messages ---
        self.inbox = InboxStore()

        # --- peer tracking ---
        self.peers: Dict[str, PeerState] = {}

        # In-flight content fetches. Keyed by content hash; the value is
        # the asyncio.Future that will be resolved when the matching
        # MSG_DATA response arrives. See fetch_peer_post + the MSG_DATA
        # handler below.
        self._pending_fetches: Dict[str, asyncio.Future] = {}

        # --- background tasks ---
        self._tasks: List[asyncio.Task] = []
        self._running = False

        # --- extension hooks ---
        # Other layers (the global-discovery glue, future plugins) can register
        # additional ctl commands and additional pieces of status info without
        # having to know about each other or modify daemon.py. Patterns:
        #
        #   daemon.register_ctl("connect", async_handler_taking_arg_string)
        #   daemon.register_info("bootstrap", lambda: "url-string")
        #
        # The synthetic /n/peribus/bootstrap file (and any future ones) read
        # their content from these providers; the ctl file dispatches commands
        # by trying built-ins first, then extensions.
        self._ctl_extensions: Dict[str, "Callable[[str], Awaitable[None]]"] = {}
        self._info_providers: Dict[str, "Callable[[], str]"] = {}

        # Wire the swarm-search ctl commands. `search <query>` runs a
        # federated search over resonance-overlay neighbors and logs the
        # ranked hits. `install <app_id>` fetches an app's source from
        # whoever offers it, validates it, and adds it to our corpus.
        self.register_ctl("search", self._ctl_search)
        self.register_ctl("install", self._ctl_install)

    # ======================================================================
    # ctl extension handlers (swarm search)
    # ======================================================================

    async def _ctl_search(self, arg: str) -> None:
        """Handler for `search <query>` — federated semantic search."""
        query = arg.strip()
        if not query:
            logger.info("search: usage: search <query>")
            return
        results = await self.app_swarm.search(query)
        if not results:
            logger.info(f"search: no results for {query!r}")
            return
        logger.info(f"search: {len(results)} hits for {query!r}")
        for h in results[:10]:
            title = h.title or "(untitled)"
            logger.info(
                f"  [{h.consensus_score:.3f}] {h.app_id} "
                f"({len(h.responders)} responder(s), {len(h.content)} bytes) — {title}"
            )

    async def _ctl_install(self, arg: str) -> None:
        """
        Handler for `install <app_id>` — fetch + validate.

        Old behavior: write to peribus's own corpus. New behavior:
        there's no peribus corpus. This handler is kept for parity with
        the old CLI but now just logs the bytes' availability. The
        filesystem-side equivalent — `cat /n/peribus/apps/<id>/source`
        — is what users should reach for in practice.
        """
        app_id = arg.strip()
        if not app_id:
            logger.info("install: usage: install <app_id>")
            return

        # Search the swarm to find responders for this hash.
        results = await self.app_swarm.search(app_id, include_local=False)
        target = next((r for r in results if r.app_id == app_id), None)
        if target is None:
            logger.warning(f"install: no responder knows {app_id}; nothing to install")
            return
        source = await self.app_swarm.install(target)
        if source is None:
            logger.warning(f"install: {app_id} fetch/validation failed")
            return
        logger.info(
            f"install: {app_id} ready ({len(source)} bytes). "
            f"Read via /n/peribus/apps/{app_id}/source"
        )

    # ======================================================================
    # Lifecycle
    # ======================================================================

    async def start(self, with_discovery: bool = True) -> None:
        """Bring up the wire server and (optionally) start discovery."""
        if self._running:
            return
        self._running = True

        # Set our hello payload.
        self.wire.set_hello(self._hello_payload())
        await self.wire.start()

        # Bring up the embed-search client. Probes /n/llm/embed; logs a
        # one-liner whether the mount is available. Safe to start regardless.
        await self.embed_search.start()

        if with_discovery:
            self.discovery = MdnsDiscovery()
            self.discovery._our_nodeid = self.identity.nodeid
            self.discovery.on_peer_appeared = self._on_peer_appeared
            self.discovery.on_peer_disappeared = self._on_peer_disappeared
            await self.discovery.start()
            await self._refresh_announcement()

        # Periodic refresh of our advertised sketch, so peers see drift.
        self._tasks.append(asyncio.create_task(self._announcement_loop()))

        # Periodic reaping of finalized share/clone drafts. Cheap: walks
        # share/'s children once a minute and drops anything finalized
        # past the TTL. In-progress drafts are never reaped.
        self._tasks.append(asyncio.create_task(self._draft_reaper_loop()))

        # (Previous versions had a _decay_sweep_loop that called
        # embed_search.decay_sweep() to unindex/purge unused apps from
        # peribus's shadow corpus. The shadow corpus is gone, the agent
        # owns the corpus, the agent owns its own decay. Removed.)

        logger.info(f"peribusd up: nodeid={self.identity.nodeid} port={self.listen_port}")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

        if self.discovery is not None:
            await self.discovery.stop()
            self.discovery = None

        # Flush corpus index to disk so use counts survive restart.
        await self.embed_search.stop()

        await self.wire.stop()

    async def _announcement_loop(self) -> None:
        """Re-announce every 60s so peers see our current vector sketch."""
        try:
            while self._running:
                await asyncio.sleep(60)
                if self._running:
                    await self._refresh_announcement()
                    await self._send_announce_to_peers()
        except asyncio.CancelledError:
            pass

    async def _draft_reaper_loop(self) -> None:
        """
        Sweep finalized share/clone drafts whose TTL has expired.

        Drafts that are still in-progress (state == "draft" or "publishing")
        are never reaped — there's no liveness signal we can trust to
        decide a draft is truly abandoned, and a forgotten draft is
        cheap to keep around. Once a draft is published or discarded,
        the result file is the only thing readers care about, and they
        get _DRAFT_TTL_S seconds to consume it before the directory
        disappears.
        """
        try:
            while self._running:
                await asyncio.sleep(60)
                if not self._running:
                    return
                share_dir = None
                # The share dir is hung off the synthetic root, which the
                # daemon stores after build_peribus_tree.
                root = getattr(self, "fs_root", None)
                if root is not None:
                    share_dir = root.children.get("share")
                if share_dir is None or not hasattr(share_dir, "reap_old_drafts"):
                    continue
                try:
                    n = share_dir.reap_old_drafts(time.time())
                except Exception as e:
                    logger.warning(f"draft reaper: {e}")
                    continue
                if n:
                    logger.debug(f"draft reaper: removed {n} finalized draft(s)")
        except asyncio.CancelledError:
            pass

    async def _refresh_announcement(self) -> None:
        if self.discovery is None:
            return
        sketch = make_sketch(self.identity_vector.snapshot())
        await self.discovery.announce(LocalAnnouncement(
            nodeid=self.identity.nodeid,
            port=self.listen_port,
            pubkey=self.identity.public_key_bytes(),
            sketch=sketch,
        ))

    # ======================================================================
    # Filesystem callbacks (called from synthetic file read/write)
    # ======================================================================

    async def status_text(self) -> str:
        """Body of /n/peribus/ctl when read."""
        peer_lines = []
        for nid, ps in sorted(self.peers.items(), key=lambda kv: -kv[1].resonance):
            age = int(time.time() - ps.last_seen)
            peer_lines.append(
                f"  {nid}  resonance={ps.resonance:.2f}  age={age}s"
            )
        peers_block = "\n".join(peer_lines) or "  (no peers)"
        stats = self.gossip.stats()
        # Pull anything extension layers want to surface (e.g. dht stats,
        # bootstrap url). Each provider returns a single line or short block.
        extra_lines = []
        for name, provider in sorted(self._info_providers.items()):
            try:
                value = provider()
            except Exception:
                continue
            if value is None:
                continue
            value = str(value).rstrip()
            if value:
                extra_lines.append(f"{name}: {value}")
        extras_block = ("\n" + "\n".join(extra_lines)) if extra_lines else ""
        return (
            f"peribus/0.1\n"
            f"nodeid: {self.identity.nodeid}\n"
            f"port: {self.listen_port}\n"
            f"running: {self._running}\n"
            f"summary: {self.summary or '(empty)'}\n"
            f"peers: {len(self.peers)}\n{peers_block}\n"
            f"posts: {stats['posts']}"
            f"{extras_block}\n"
        )

    def register_ctl(self, command: str, handler) -> None:
        """Register an extension command for /n/peribus/ctl.

        The handler is `async def handler(arg: str) -> None`. It receives
        whatever followed the command on the line, with surrounding whitespace
        stripped. Built-in commands take precedence over extensions; you cannot
        shadow `start`, `stop`, `dial`, `forget`, `attract`, `follow`, `share`.
        """
        self._ctl_extensions[command.lower()] = handler

    def register_info(self, name: str, provider) -> None:
        """Register a one-shot info source.

        `provider()` is called synchronously when /n/peribus/ctl is read; its
        return value (a string or None) is appended to the status block as
        `name: value`. Used by the global-discovery glue to surface things
        like `bootstrap: NODEID@host:port` and `dht-peers: 27`.

        Files that publish a single info value (like /n/peribus/bootstrap)
        also pull from this same registry, looking up by `name`.
        """
        self._info_providers[name] = provider

    def get_info(self, name: str) -> Optional[str]:
        """Read an info value by name. Returns None if unset or if the
        provider raised. Used by synthetic info files in filesystem.py."""
        provider = self._info_providers.get(name)
        if provider is None:
            return None
        try:
            value = provider()
        except Exception:
            return None
        if value is None:
            return None
        return str(value).rstrip()

    async def handle_ctl(self, line: str) -> None:
        """Process a single command from /n/peribus/ctl."""
        parts = line.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "start":
            await self.start()
        elif cmd == "stop":
            await self.stop()
        elif cmd == "dial":
            # Manual peering for networks where mDNS isn't crossing.
            #
            # Usage: `echo "dial 192.168.1.42" > /n/peribus/ctl`
            #        `echo "dial 192.168.1.42:5660" > /n/peribus/ctl`
            #
            # We don't yet know the peer's NodeID — that comes back in
            # their hello. Use a placeholder keyed on the address; once
            # the hello is processed by the wire on_message hook, the
            # daemon will register the real peer state. The placeholder
            # only exists to give wire.dial somewhere to hang the conn.
            host, _, port_s = arg.strip().partition(":")
            host = host.strip()
            try:
                port = int(port_s) if port_s else 5660
            except ValueError:
                logger.warning(f"dial: bad port in {arg!r}")
                return
            if not host:
                logger.warning("dial: missing host")
                return
            placeholder = f"pending:{host}:{port}"
            logger.info(f"dial: connecting to {host}:{port}")
            conn = await self.wire.dial(placeholder, host, port)
            if conn is None:
                logger.warning(f"dial: could not reach {host}:{port}")
            else:
                logger.info(f"dial: handshake in flight with {host}:{port}")
        elif cmd == "forget":
            # forget <nodeid>  — drop a peer from our state and close conn
            nodeid = arg.strip()
            if not nodeid:
                logger.warning("forget: missing nodeid")
                return
            self.peers.pop(nodeid, None)
            if self.nodes_dir is not None:
                self.nodes_dir.remove_peer(nodeid)
            conn = self.wire.get_conn(nodeid)
            if conn is not None:
                await conn.close()
            logger.info(f"forget: dropped {nodeid}")
        elif cmd == "attract":
            # Treat the text as a fresh observation, weighted heavily.
            vec = await self.embedder.embed(arg)
            await self.identity_vector.observe(vec, weight=3.0)
            asyncio.create_task(self._refresh_announcement())
        elif cmd == "follow":
            # We don't track an explicit follow set; instead, briefly bias
            # identity toward the named peer's last-seen sketch.
            ps = self.peers.get(arg.strip())
            if ps and ps.info.sketch:
                # Sketch lives in low-d space; we can't add it to identity
                # directly. Instead, refetch their summary as text and
                # observe that. For v0.1, use the sketch as a coarse signal:
                # boost resonance estimate so they appear higher.
                ps.resonance = max(ps.resonance, 0.95)
        elif cmd == "share":
            # share <local-path>
            p = Path(arg.strip()).expanduser()
            if not p.exists():
                logger.warning(f"share: path does not exist: {p}")
            elif not p.is_file():
                logger.warning(f"share: not a regular file: {p}")
            else:
                try:
                    data = p.read_bytes()
                except OSError as e:
                    logger.warning(f"share: cannot read {p}: {e}")
                else:
                    h = await self.publish_share(p.name, data)
                    logger.info(f"share: published {p.name} as {h}")
        else:
            handler = self._ctl_extensions.get(cmd)
            if handler is not None:
                try:
                    await handler(arg.strip())
                except Exception as e:
                    logger.warning(f"ctl {cmd}: handler raised: {e}")
            else:
                logger.warning(f"unknown ctl command: {cmd}")

    async def set_summary(self, text: str) -> None:
        """Write to /n/peribus/identity/summary updates the identity vector strongly."""
        self.summary = text
        if text:
            vec = await self.embedder.embed(text)
            # Strong signal — user explicitly told us who they are.
            await self.identity_vector.observe(vec, weight=5.0)
            asyncio.create_task(self._refresh_announcement())

    async def publish_share(self, name: str, data: bytes) -> str:
        """Publish a shared item: store, embed, sign, broadcast."""
        # Store content.
        h = self.gossip.put_content(data)

        # Embed.
        # If it's text, embed directly. If binary (e.g. image), embed the
        # filename + a brief hex preview. Crude but workable for v0.1.
        try:
            text = data.decode("utf-8")
            preview = text[:512]
        except UnicodeDecodeError:
            preview = f"{name} (binary, {len(data)} bytes)"
        vec = await self.embedder.embed(name + " " + preview)

        # Build post.
        post = Post(
            id=h,
            author=self.identity.nodeid,
            ts=time.time(),
            title=name,
            body=preview[:280],
            vector=vec,
            attachments=[h],
        )
        # Sign canonical body.
        canonical = (
            f"{post.id}|{post.author}|{int(post.ts*1000)}|{post.title}".encode("utf-8")
        )
        post.sig = self.identity.sign(canonical)

        # Add to local feed.
        await self.gossip.add_post(post)

        # Boost identity toward our own work (you become what you make).
        await self.identity_vector.observe(vec, weight=2.0)

        # Broadcast to peers (fast — local writes to per-conn write buffers).
        await self.wire.broadcast(self._post_payload(post))

        # Schedule mDNS re-announce in the background. zeroconf's register
        # does ~6-10s of probing; awaiting it here would block the file
        # handle's clunk and cause client timeouts. We don't need callers
        # to wait for the new sketch to propagate.
        asyncio.create_task(self._refresh_announcement())
        return h

    async def publish_draft(
        self,
        *,
        body: str,
        kind: str = "post",
        reply_to: str = "",
        attachment_paths: Optional[List[str]] = None,
    ) -> str:
        """
        Publish a draft assembled via the share/clone interface.

        Funnels the structured draft attributes into the same gossip path
        publish_share uses, but without forcing the caller to JSON-encode
        an envelope themselves. Steps:

          1. For each attachment path: read the bytes, store in the
             content cache, get back a hash. The bytes are content-
             addressed so peers can MSG_FETCH them via
             nodes/<author>/social/<hash>.
          2. Embed the body text (plus the kind/reply_to context, which
             pushes replies toward the same vector neighborhood as their
             parent topic).
          3. Build a Post whose `body` is a JSON envelope describing
             kind, reply_to, body text, and attachment metadata. This is
             the same wire shape the widget produced before — receivers
             that already know how to unwrap envelope-style posts keep
             working unchanged.
          4. Sign, broadcast, return the hash.

        The envelope-in-body shape is a backwards-compat constraint: until
        Post grows first-class fields for kind/reply_to/etc, threading
        and structured posts ride inside body as JSON. A future cleanup
        moves these to top-level Post fields and adapts both sides.
        """
        import json as _json

        attachment_paths = list(attachment_paths or [])
        body_text = body or ""

        # ---- 1) Stage attachments ----
        # Each path becomes (filename, content_hash, bytes_size). The
        # bytes go into the gossip content store keyed by hash; peers
        # later fetch via MSG_FETCH using exactly those hashes.
        #
        # Strict by default: if any attachment can't be read, the
        # publish fails with a clear message rather than silently
        # producing a media-less post. Silent skipping was a bug —
        # users would see `result` return a hash and assume the file
        # had attached, then wonder why the receiver saw nothing.
        attachment_meta: List[dict] = []
        attachment_hashes: List[str] = []
        for raw in attachment_paths:
            # expanduser is brittle — it consults pwd for ~user lookups
            # and can raise RuntimeError("Could not determine home
            # directory") in environments where HOME is unset OR if the
            # raw string is malformed enough to look like a ~user path.
            # Try the expansion, but fall through to the raw path on
            # failure rather than crashing the whole publish.
            try:
                p = Path(raw).expanduser()
            except (RuntimeError, KeyError):
                p = Path(raw)
            if not p.exists():
                raise FileNotFoundError(
                    f"attachment not found: {raw}"
                )
            if not p.is_file():
                raise IsADirectoryError(
                    f"attachment is not a regular file: {raw}"
                )
            try:
                data = p.read_bytes()
            except OSError as e:
                raise OSError(
                    f"cannot read attachment {raw}: {e}"
                ) from e
            h = self.gossip.put_content(data)
            attachment_hashes.append(h)
            attachment_meta.append({
                "hash": h,
                "filename": p.name,
                "bytes": len(data),
                # ext sniff lets receivers pick the right renderer without
                # re-magic-sniffing every time.
                "ext": p.suffix.lstrip(".").lower(),
            })

        # ---- 2) Embed for routing ----
        # Embedding signal: body text + kind hint + each attachment
        # filename. If the post has no body (pure media), the filenames
        # are the only routing signal we have, so include them.
        signal_parts = []
        if body_text:
            signal_parts.append(body_text[:512])
        if kind and kind != "post":
            signal_parts.append(kind)
        for m in attachment_meta:
            signal_parts.append(m["filename"])
        signal = " ".join(signal_parts) or "(empty post)"
        vec = await self.embedder.embed(signal)

        # ---- 3) Build envelope-in-body ----
        # Receivers that understand the envelope schema parse this and
        # render it richly; older receivers see the raw envelope JSON in
        # the post body and fall back to plain text — graceful degradation.
        envelope: dict = {
            "type": "post",
            "body": body_text,
            "ts": time.time(),
        }
        if kind == "reply" and reply_to:
            envelope["reply_to"] = reply_to
            # The receiving widget likes a hint about the parent author
            # so it can render "↳ reply to @<short>" without an extra
            # lookup. We don't have it server-side — clients can supply
            # it via a future `reply_to_author` attribute file if needed.
        if attachment_meta:
            # Each attachment gets a media stub. The `type` field tells
            # receivers which renderer to dispatch to *before* they've
            # fetched the bytes — image, gif, audio, video, pdf, model3d
            # for binary kinds, python/html for source. We derive this
            # from the filename extension here so receivers don't have
            # to re-sniff every blob; the receiver-side code still falls
            # back to magic-byte sniffing if the hint is missing or
            # unrecognized.
            #
            # `_stripped: True` signals "the bytes aren't inline; fetch
            # them via attachments[i] hash." All non-tiny attachments are
            # stripped — there's no inline path here at all (unlike the
            # old wire envelope, which had the 80-byte inline branch).
            envelope["media"] = [
                {
                    "type": self._media_kind_for_ext(m["ext"]),
                    "filename": m["filename"],
                    "bytes": m["bytes"],
                    "format": m["ext"] or None,
                    "_stripped": True,
                }
                for m in attachment_meta
            ]

        envelope_json = _json.dumps(envelope, separators=(",", ":"))

        # The post's `body` field on the wire is capped to 280 bytes
        # (gossip's body cap). If our envelope busts that, we still ship
        # the post — the truncation is recoverable on the receiver
        # because attachments live in the (uncapped) attachments list.
        # The widget's caption-clip helper used to do this; doing it
        # here means clients don't have to.
        body_for_post = envelope_json[:280]

        # The post's content hash addresses the *envelope*. We also
        # store the envelope bytes themselves so peers walking
        # nodes/<author>/social/<hash> can fetch the post body.
        post_id = self.gossip.put_content(envelope_json.encode("utf-8"))

        post = Post(
            id=post_id,
            author=self.identity.nodeid,
            ts=envelope["ts"],
            title=self._draft_title(kind, body_text, attachment_meta),
            body=body_for_post,
            vector=vec,
            attachments=attachment_hashes,
        )

        canonical = (
            f"{post.id}|{post.author}|{int(post.ts*1000)}|{post.title}".encode("utf-8")
        )
        post.sig = self.identity.sign(canonical)

        # ---- 4) Local feed + broadcast ----
        await self.gossip.add_post(post)
        await self.identity_vector.observe(vec, weight=2.0)
        await self.wire.broadcast(self._post_payload(post))
        asyncio.create_task(self._refresh_announcement())
        return post.id

    @staticmethod
    def _draft_title(
        kind: str, body_text: str, attachment_meta: List[dict]
    ) -> str:
        """
        Pick a human-readable title for a draft-published post.

        Title is a short string surfaced in feed cards. Preference order:
            1. First non-empty line of the body, clipped to 60 chars.
            2. The first attachment's filename.
            3. A kind-derived placeholder ("(reply)" / "(post)").
        """
        if body_text:
            for line in body_text.splitlines():
                line = line.strip()
                if line:
                    return line[:60]
        if attachment_meta:
            return attachment_meta[0]["filename"]
        return f"({kind or 'post'})"

    # Maps a normalized filename extension (no leading dot, lowercase)
    # to the media kind tag the receiver-side InlineMediaWidget
    # dispatches on. Mirrors the widget's _SHARE_EXT_KIND — keep them
    # in sync. Anything not in this table gets "info" as a soft
    # fallback ("show whatever bytes you fetched as a card"); receivers
    # then re-sniff via magic bytes before deciding what to actually
    # render.
    _MEDIA_EXT_KIND = {
        # raster images
        "png": "image", "jpg": "image", "jpeg": "image",
        "webp": "image", "bmp": "image",
        # animated
        "gif": "gif",
        # audio
        "mp3": "audio", "wav": "audio", "ogg": "audio",
        "flac": "audio", "m4a": "audio",
        # video
        "mp4": "video", "mkv": "video", "webm": "video",
        "mov": "video", "avi": "video",
        # documents
        "pdf": "pdf",
        # 3D
        "obj": "model3d", "stl": "model3d", "glb": "model3d",
        "gltf": "model3d", "ply": "model3d",
        # source code / markup — text kinds. Receivers slurp the
        # bytes into the renderer's `code`/`content` field; without
        # this the chip would fall through to the generic info card.
        "py": "python",
        "html": "html", "htm": "html", "svg": "html",
    }

    @classmethod
    def _media_kind_for_ext(cls, ext: str) -> str:
        """Map a file extension to a renderer kind, or 'info' as fallback."""
        if not ext:
            return "info"
        return cls._MEDIA_EXT_KIND.get(ext.lower().lstrip("."), "info")

    async def _fetch_app_source(self, peer_nodeid: str, app_id: str) -> Optional[bytes]:
        """
        Shim used by AppSwarm to pull an app's source bytes from a peer.
        App IDs are content hashes, so this is the same content-addressed
        fetch path posts use for attachments — just renamed for clarity
        at the call site. Returns None on any failure (no conn, timeout,
        peer doesn't have it).
        """
        return await self.fetch_peer_post(peer_nodeid, app_id)

    async def fetch_peer_post(self, peer: str, name: str) -> Optional[bytes]:
        """
        Look up a post by content-hash filename in a peer's social/ dir.

        Names in social/ are content hashes ("b3:..."). Strategy:
          1. Local content store hit → return immediately
          2. Otherwise, send MSG_FETCH to the peer and await the MSG_DATA
             response. Times out after 5s if the peer doesn't have it or
             never replies. The data is verified by hash on receipt.
        """
        # Fast path: we already have it.
        local = self.gossip.get_content(name)
        if local is not None:
            return local

        # Need a wire connection to the peer.
        conn = self.wire.get_conn(peer)
        if conn is None:
            ps = self.peers.get(peer)
            if ps is None:
                return None
            conn = await self.wire.dial(peer, ps.info.host, ps.info.port)
            if conn is None:
                return None

        # Register a future under this hash so the MSG_DATA handler can
        # complete it. Multiple concurrent fetches for the same hash share
        # one future — once it resolves, all readers see the content.
        loop = asyncio.get_running_loop()
        existing = self._pending_fetches.get(name)
        if existing is not None:
            future = existing
        else:
            future = loop.create_future()
            self._pending_fetches[name] = future

        # Only one of the racers actually sends the MSG_FETCH.
        if existing is None:
            try:
                await conn.send({
                    "type": MSG_FETCH,
                    "from": self.identity.nodeid,
                    "hash": name,
                    "ts": int(time.time() * 1000),
                })
            except Exception as e:
                self._pending_fetches.pop(name, None)
                if not future.done():
                    future.set_exception(e)
                return None

        # Wait for the MSG_DATA response (or timeout / error).
        try:
            return await asyncio.wait_for(future, timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            return None
        finally:
            # Only clean up if we're the one holding the slot.
            if self._pending_fetches.get(name) is future:
                self._pending_fetches.pop(name, None)

    def peer_signal(self, peer: str) -> dict:
        ps = self.peers.get(peer)
        if not ps:
            return {}
        return {
            "strength": 1.0 if self.wire.get_conn(peer) else 0.0,
            "latency_ms": -1,  # measured during ping in a fuller version
            "last_seen": int(ps.last_seen),
            "resonance": ps.resonance,
        }

    def knows_peer(self, nodeid: str) -> bool:
        return nodeid in self.peers

    async def send_message(self, peer: str, text: str) -> None:
        conn = self.wire.get_conn(peer)
        if conn is None:
            # Try to dial.
            ps = self.peers.get(peer)
            if ps:
                conn = await self.wire.dial(peer, ps.info.host, ps.info.port)
        if conn is None:
            logger.warning(f"cannot send: no conn to {peer}")
            return
        await conn.send({
            "type": MSG_MSG,
            "from": self.identity.nodeid,
            "to": peer,
            "ts": int(time.time() * 1000),
            "body": text,
        })

    # ======================================================================
    # Discovery callbacks
    # ======================================================================

    async def _on_peer_appeared(self, info: PeerInfo) -> None:
        """A new peer was seen on the network."""
        if info.nodeid == self.identity.nodeid:
            return  # ourselves

        # Verify the advertised pubkey matches the claimed nodeid.
        if info.pubkey:
            expected = nodeid_from_pubkey(info.pubkey)
            if expected != info.nodeid:
                logger.warning(
                    f"mDNS peer {info.nodeid} pubkey mismatch (got {expected}); ignoring"
                )
                return

        # Compute resonance from the advertised sketch.
        from peribus._foundation import make_sketch
        my_sketch = make_sketch(self.identity_vector.snapshot())
        resonance = cosine(my_sketch, info.sketch) if info.sketch else 0.0

        # Track the peer.
        self.peers[info.nodeid] = PeerState(
            info=info,
            last_seen=time.time(),
            resonance=resonance,
        )

        # Materialize in /n/peribus/nodes/.
        if self.nodes_dir is not None:
            self.nodes_dir.add_peer(info.nodeid)

        logger.info(f"peer appeared: {info.nodeid} @ {info.host}:{info.port}  "
                    f"resonance={resonance:.2f}")

        # Optionally dial proactively if resonance is high enough.
        if resonance > 0.4:
            await self.wire.dial(info.nodeid, info.host, info.port)

    async def _on_peer_disappeared(self, nodeid: str) -> None:
        self.peers.pop(nodeid, None)
        if self.nodes_dir is not None:
            self.nodes_dir.remove_peer(nodeid)
        # Drop any wire conn.
        conn = self.wire.get_conn(nodeid)
        if conn is not None:
            await conn.close()
        logger.info(f"peer disappeared: {nodeid}")

    # ======================================================================
    # Wire callbacks
    # ======================================================================

    def _hello_payload(self) -> dict:
        return {
            "type": MSG_HELLO,
            "from": self.identity.nodeid,
            "pubkey": base64.b64encode(self.identity.public_key_bytes()).decode("ascii"),
            "port": self.listen_port,
            "sketch": make_sketch(self.identity_vector.snapshot()),
            "v": "peribus/0.1",
            "ts": int(time.time() * 1000),
        }

    def _post_payload(self, post: Post) -> dict:
        return {
            "type": MSG_POST,
            "from": self.identity.nodeid,
            "ts": int(time.time() * 1000),
            "post": {
                "id": post.id,
                "author": post.author,
                "ts": post.ts,
                "title": post.title,
                "body": post.body,
                "vector": post.vector,
                "attachments": post.attachments,
                "sig": base64.b64encode(post.sig).decode("ascii"),
            },
        }

    async def _send_announce_to_peers(self) -> None:
        msg = {
            "type": MSG_ANNOUNCE,
            "from": self.identity.nodeid,
            "ts": int(time.time() * 1000),
            "sketch": make_sketch(self.identity_vector.snapshot()),
            "summary": self.summary,
        }
        await self.wire.broadcast(msg)

    async def _on_wire_message(self, conn: WireConn, msg: dict) -> None:
        """Dispatch one peer message."""
        t = msg.get("type")

        # Try the swarm-search layer first. It owns app_search/app_results
        # and nothing else — if it doesn't claim the message we fall
        # through to the existing dispatch table below.
        try:
            if await self.app_swarm.handle_message(conn.nodeid, msg):
                return
        except Exception as e:
            logger.warning(f"app_swarm.handle_message raised: {e}")

        if t == MSG_HELLO:
            # Verify pubkey -> nodeid binding.
            try:
                pub = base64.b64decode(msg.get("pubkey", ""))
            except Exception:
                return
            claimed = msg.get("from")
            if pub and nodeid_from_pubkey(pub) != claimed:
                logger.warning(f"hello pubkey mismatch from {claimed}")
                await conn.close()
                return

            # Re-key the conn if it was opened under a placeholder
            # (manual dial to an IP, before we knew the NodeID).
            if conn.nodeid != claimed:
                self.wire.re_key(conn.nodeid, claimed)

            # Register the peer in our state if we haven't already.
            # This is what `_on_peer_appeared` does for mDNS-discovered
            # peers — we mirror it here so manually-dialed peers (and
            # peers who dialed us first) show up under /n/peribus/nodes.
            if claimed not in self.peers:
                # Best-effort: fish a host out of the conn's transport.
                host = ""
                try:
                    peername = conn.writer.get_extra_info("peername")
                    if peername:
                        host = peername[0]
                except Exception:
                    pass
                info = PeerInfo(
                    nodeid=claimed,
                    host=host,
                    port=int(msg.get("port", 5660)),
                    pubkey=pub,
                    sketch=list(msg.get("sketch", []) or []),
                    last_seen=time.time(),
                )
                self.peers[claimed] = PeerState(
                    info=info,
                    last_seen=time.time(),
                    resonance=0.0,
                )
                if self.nodes_dir is not None:
                    self.nodes_dir.add_peer(claimed)
                logger.info(
                    f"peer registered via hello: {claimed} @ {host}:{info.port}"
                )
            else:
                # Refresh last_seen on an already-known peer.
                self.peers[claimed].last_seen = time.time()
            return

        elif t == MSG_ANNOUNCE:
            sender = msg.get("from")
            ps = self.peers.get(sender)
            sketch = msg.get("sketch", [])
            if ps and sketch:
                my_sketch = make_sketch(self.identity_vector.snapshot())
                ps.resonance = cosine(my_sketch, sketch)
                ps.last_seen = time.time()

        elif t == MSG_POST:
            await self._handle_inbound_post(msg, conn)

        elif t == MSG_FETCH:
            h = msg.get("hash")
            data = None
            if h:
                # Try the gossip content store first (posts / attachments).
                data = self.gossip.get_content(h)
                # Then try the embed_search hash → path cache (app sources
                # surfaced by recent searches). Keyed by sha256:<hex>.
                if data is None and h.startswith("sha256:"):
                    data = await self.embed_search.get_content_by_hash(h)
            if data is not None:
                await conn.send({
                    "type": MSG_DATA,
                    "from": self.identity.nodeid,
                    "hash": h,
                    "bytes": base64.b64encode(data).decode("ascii"),
                })

        elif t == MSG_DATA:
            h = msg.get("hash")
            try:
                data = base64.b64decode(msg.get("bytes", ""))
            except Exception:
                return
            if not (h and data):
                return
            # Verify the hash. Two schemes coexist on the wire:
            #   - "b3:..." (legacy, sha256 base32) — gossip posts/attachments
            #   - "sha256:<hex>" (new) — app source from embed_search
            if h.startswith("sha256:"):
                expected = "sha256:" + hashlib.sha256(data).hexdigest()
            else:
                expected = _content_hash(data)
            if expected != h:
                return
            # Stash in gossip's blob store for cross-peer rebroadcast.
            self.gossip.put_content(data)
            # Wake any fetch_peer_post awaiter for this hash.
            pending = self._pending_fetches.pop(h, None)
            if pending is not None and not pending.done():
                pending.set_result(data)

        elif t == MSG_MSG:
            # Direct message from peer.
            sender = msg.get("from", "")
            body = msg.get("body", "")
            if not sender:
                return
            m = Message(sender=sender, body=body, ts=time.time())
            await self.inbox.add(m)
            # Per-peer index for /n/peribus/nodes/<nodeid>/from.
            ps = self.peers.get(sender)
            if ps is not None:
                ps.inbox.append(m)
            logger.info(f"msg from {sender}: {body[:80]}")

        elif t == MSG_PING:
            await conn.send({"type": MSG_PONG, "from": self.identity.nodeid})

    async def _handle_inbound_post(self, msg: dict, conn: WireConn) -> None:
        """Validate, store, and re-gossip a post."""
        p = msg.get("post") or {}
        try:
            post = Post(
                id=p["id"],
                author=p["author"],
                ts=float(p["ts"]),
                title=p.get("title", ""),
                body=p.get("body", ""),
                vector=p.get("vector", []),
                attachments=p.get("attachments", []),
                sig=base64.b64decode(p.get("sig", "")),
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"bad post: {e}")
            return

        # Verify signature against author's pubkey (we have it from hello).
        author_ps = self.peers.get(post.author)
        if author_ps and author_ps.info.pubkey:
            canonical = (
                f"{post.id}|{post.author}|{int(post.ts*1000)}|{post.title}".encode("utf-8")
            )
            if not verify_signature(author_ps.info.pubkey, canonical, post.sig):
                logger.warning(f"bad signature on post from {post.author}")
                return

        # Store + add to feed.
        newly_seen = await self.gossip.add_post(post)

        # Re-gossip if this is fresh — but not back to the sender.
        if newly_seen:
            for nid, peer_conn in list(self.wire._conns.items()):
                if nid == conn.nodeid or nid == post.author:
                    continue
                try:
                    await peer_conn.send(msg)
                except Exception:
                    pass

    async def _on_wire_disconnect(self, nodeid: str) -> None:
        # Don't remove from self.peers — discovery still considers them
        # "around" until mDNS times them out. Just no active wire conn.
        logger.info(f"wire disconnected from {nodeid}")

# ============================================================================
# feed_bridge.py
# ----------------------------------------------------------------------------
"""
peribus.feed_bridge — turn the feed into Qt cards on the rio canvas

Runs inside the rio process (where Qt lives). Tails the mounted file
/n/peribus/feed/new in a worker thread, parses each JSON line into a
Card, and renders the card as a Qt widget on the graphics scene.

The whole UX:

  1. User runs `start_feed_bridge(scene_manager, graphics_scene)` once
     (e.g. from rio's parse file, a context menu, or main.py startup).
  2. Existing posts in the buffer immediately render as cards in a column
     on the right edge of the scene. New posts arrive as cards from the
     top, pushing older ones down.
  3. Each card has actions:
       deepen → write `attract <topic>` to /n/peribus/ctl
                (drifts your identity vector toward this post)
       render → fetch the attached widget, validate, run in sandbox
       reply  → small text input, write to /n/peribus/nodes/<author>/inbox

Design choices worth knowing:

  * No follow button. The act of opening posts IS the follow — your
    identity vector drifts, the feed reorders accordingly.
  * Cards live on the rio QGraphicsScene at a fixed column. The user
    pans/zooms the canvas to see older ones. The scene IS the scroll
    surface — we don't wrap a QScrollArea on top.
  * The bridge runs entirely on rio's existing asyncio loop. No new
    threads beyond asyncio.to_thread for the blocking file reads.
  * Foreign widgets only run after passing the AST validator in
    peribus.widget_validator. The runtime in widget_runtime is separate
    from rio's own parser — shared widgets cannot reach rio internals.
"""


import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Set

logger = logging.getLogger(__name__)


# Module-level strong references for anything that must survive past the
# parser executor's namespace teardown.
#
# Why this matters: start_feed_bridge() is typically called from rio's parse
# file (e.g. /n/<workspace>/scene/parse). The parser creates a FeedBridge,
# its tailer creates an asyncio.Task via asyncio.create_task(...), and once
# the parser returns the only strong reference to that task is on the
# FeedTailer instance. If the user doesn't assign the FeedBridge to a stable
# global, Python eventually GCs the bridge -> the tailer -> and the task
# along with it, which silently cancels mid-read. That's the difference
# between "I see no cards" and "the daemon never gets read".
#
# asyncio's own docs are blunt about this: keep your own reference to a
# task or it can vanish under you. We do that here, so callers don't have
# to remember.
_LIVE_BRIDGES: "Set[FeedBridge]" = set()
_LIVE_TASKS: "Set[asyncio.Task]" = set()


def _say(msg: str) -> None:
    """
    Diagnostic print that bypasses sys.stdout/sys.stderr.

    Some embedding contexts (rio's parse executor, IDE consoles, capture
    fixtures) replace sys.stdout/sys.stderr. The raw os.write to fd 2 (the
    process's actual stderr file descriptor) survives all of that.
    """
    try:
        os.write(2, (msg + "\n").encode("utf-8"))
    except OSError:
        # Last-ditch fallback if fd 2 is itself broken.
        try:
            print(msg, file=sys.__stderr__, flush=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Card data model
# ---------------------------------------------------------------------------

@dataclass
class Card:
    """One feed post as it lives on the canvas."""
    post_id: str
    author: str
    title: str
    body: str
    resonance: float
    ts: float
    attachments: List[str] = field(default_factory=list)
    qt_item: Any = None    # QGraphicsProxyWidget after render

    @classmethod
    def from_feed_line(cls, line: str) -> Optional["Card"]:
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            return None
        return cls(
            post_id=d.get("id", ""),
            author=d.get("author", ""),
            title=d.get("title", ""),
            body=d.get("body", ""),
            resonance=float(d.get("resonance", 0.0)),
            ts=float(d.get("ts", time.time())),
            attachments=list(d.get("attachments", [])),
        )


# ---------------------------------------------------------------------------
# Tailer — async file reader, lives on the rio asyncio loop
# ---------------------------------------------------------------------------

class FeedTailer:
    """
    Tails /n/peribus/feed/new line-by-line.

    `feed/new` is the *blocking* feed file: reads return all current
    buffer content first, then block until new posts arrive. Perfect
    for a long-lived tailer.

    Each chunk read happens in a thread-pool worker so the main asyncio
    loop (and Qt event pump) stays responsive. Lines are split out and
    delivered to on_card on the main thread.
    """

    def __init__(self, feed_path: str, on_card):
        self.feed_path = feed_path
        self.on_card = on_card
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="peribus.feed_tailer")
        # Pin the task at module scope so the GC can't reach in and cancel
        # it after the parser's namespace tears down. We drop our pin in
        # the done-callback, after the task has actually finished.
        _LIVE_TASKS.add(self._task)
        self._task.add_done_callback(_LIVE_TASKS.discard)

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        if not Path(self.feed_path).exists():
            _say(f"[peribus.feed] feed not mounted at {self.feed_path}")
            return

        partial = b""

        # IMPORTANT: open with buffering=0 (raw I/O, no BufferedReader).
        #
        # Python's default mode "rb" wraps the fd in a BufferedReader. For
        # a streaming/blocking 9P-over-FUSE file, that's a disaster: the
        # daemon answers our 8192-byte Tread with however many bytes are
        # currently available (e.g. 230), the BufferedReader sees a short
        # read on what looks like a regular file, and reissues a fresh
        # underlying read to "fill the buffer" before returning anything
        # to us. The first read's bytes get stranded inside Python's
        # internal buffer, the second read blocks in the daemon, and we
        # never see any cards on the canvas — even though the wire shows
        # the data flowing.
        #
        # buffering=0 returns a raw FileIO, where read(n) does exactly one
        # syscall and returns whatever it returns, including short reads.
        # That's the contract a tail-style consumer needs.
        try:
            f = await asyncio.to_thread(open, self.feed_path, "rb", 0)
        except OSError as e:
            _say(f"[peribus.feed] feed open failed: {e}")
            return

        _say(f"[peribus.feed] tailer entered read loop on {self.feed_path}")

        async def _reopen() -> bool:
            """Close and reopen the feed file. Returns True on success."""
            nonlocal f
            try:
                await asyncio.to_thread(f.close)
            except Exception:
                pass
            try:
                f = await asyncio.to_thread(open, self.feed_path, "rb", 0)
                _say(f"[peribus.feed] reopened {self.feed_path}")
                return True
            except OSError as e:
                _say(f"[peribus.feed] reopen failed: {e}")
                return False

        try:
            while not self._stop.is_set():
                try:
                    chunk = await asyncio.to_thread(f.read, 8192)
                except (FileNotFoundError, OSError) as e:
                    # EIO from FUSE often means the in-flight read was
                    # canceled by the kernel (e.g. timeout). The fd is in
                    # a broken state — we must reopen, not just retry.
                    _say(f"[peribus.feed] feed read error: {e} (reopening)")
                    await asyncio.sleep(0.5)
                    if not await _reopen():
                        await asyncio.sleep(2.0)
                    continue
                if not chunk:
                    # Daemon closed the file (shut down). Back off and retry.
                    await asyncio.sleep(0.5)
                    continue

                _say(f"[peribus.feed] read {len(chunk)} bytes from feed")

                partial += chunk
                while b"\n" in partial:
                    line, partial = partial.split(b"\n", 1)
                    if not line.strip():
                        continue
                    card = Card.from_feed_line(
                        line.decode("utf-8", errors="replace")
                    )
                    if card is None:
                        _say(f"[peribus.feed] could not parse line: " f"{line[:80]!r}")
                        continue
                    try:
                        self.on_card(card)
                    except Exception as e:
                        import traceback
                        _say(f"[peribus.feed] on_card raised: {e}")
                        traceback.print_exc()
        finally:
            try:
                f.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Renderer — owns the card column on the scene
# ---------------------------------------------------------------------------

class FeedRenderer:
    """
    Maintains a column of cards on the rio scene.

    Layout: cards stack downward from the top. When a new card arrives,
    it goes at the top and existing cards slide down. The user pans the
    QGraphicsView to see older cards (rio's scene is huge — 3840×2160
    by default — so there's room).

    Threading: cards arrive from the FeedTailer, which lives on the
    asyncio loop. Qt widget construction must happen on Qt's main thread
    (the thread that owns QApplication.instance()). We use a small
    QObject helper to hop add_card() over to the Qt thread via a queued
    signal — which is exactly the pattern rio itself uses elsewhere
    (see filesystem.py: QMetaObject.invokeMethod with QueuedConnection).
    """

    CARD_WIDTH = 460
    CARD_HEIGHT = 150
    CARD_GAP = 12
    # Default column position. Rio's QGraphicsView usually starts looking
    # at the top-left of the scene — so we place the card column there.
    # Override via column_x/top_y if you have other widgets in the way.
    DEFAULT_COL_X = 64
    DEFAULT_TOP_Y = 64

    def __init__(
        self,
        graphics_scene: Any,
        scene_manager: Any,
        peribus_root: str = "/n/peribus",
        column_x: Optional[int] = None,
        top_y: Optional[int] = None,
    ):
        self.graphics_scene = graphics_scene
        self.scene_manager = scene_manager
        self.peribus_root = peribus_root
        self.column_x = column_x if column_x is not None else self.DEFAULT_COL_X
        self.top_y = top_y if top_y is not None else self.DEFAULT_TOP_Y
        self.cards: List[Card] = []
        self._seen_post_ids: set = set()  # dedup if the daemon re-emits

        # Build a tiny QObject that owns a queued slot. We move it to the
        # Qt main thread (the thread of QApplication.instance()) so that
        # any invokeMethod with QueuedConnection runs the slot there.
        # Without this, _build_and_place would touch QFrame / addWidget
        # from the asyncio thread — which is undefined behaviour on Qt
        # and the most common reason "the cards are constructed but
        # nothing shows up on the canvas".
        from PySide6.QtCore import QObject, Signal, Slot, QCoreApplication, Qt

        renderer = self  # closure for the slot

        class _Bridge(QObject):
            cardArrived = Signal(object)

            @Slot(object)
            def _on_card(self, card):
                # Always runs on the thread this QObject lives on (Qt main).
                try:
                    renderer._build_and_place(card)
                    _say(f"[peribus.feed] placed card on scene")
                except Exception as e:
                    import traceback
                    _say(f"[peribus.feed] _build_and_place FAILED: {e}")
                    traceback.print_exc()

        self._bridge = _Bridge()
        self._bridge.cardArrived.connect(
            self._bridge._on_card,
            type=Qt.QueuedConnection,
        )

        # Move the bridge to the Qt main thread. If we're already on it,
        # this is a no-op. If we're being constructed from the asyncio
        # thread (the usual case), this is what makes the queued slot
        # actually run on the right thread.
        app = QCoreApplication.instance()
        if app is not None:
            self._bridge.moveToThread(app.thread())

    def add_card(self, card: Card) -> None:
        # Dedup — feed/new might re-emit the same post on cursor reset.
        # The dedup is fine to do on the asyncio thread; it touches no Qt.
        if card.post_id in self._seen_post_ids:
            return
        self._seen_post_ids.add(card.post_id)

        # Print rather than logger.info — rio's stderr always shows prints
        # but may have logging configured at WARNING level.
        _say(f"[peribus.feed] add_card: {card.title!r} " f"resonance={card.resonance:.2f} at ({self.column_x}, {self.top_y})")

        # Hand off to the Qt main thread. emit() with a queued connection
        # is thread-safe and posts an event onto Qt's main loop; the
        # actual widget construction happens there.
        try:
            self._bridge.cardArrived.emit(card)
        except Exception as e:
            import traceback
            _say(f"[peribus.feed] add_card emit FAILED: {e}")
            traceback.print_exc()

    def _build_and_place(self, card: Card) -> None:
        from PySide6.QtWidgets import (
            QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
        )

        card_widget = QFrame()
        card_widget.setFrameShape(QFrame.StyledPanel)
        card_widget.setFixedSize(self.CARD_WIDTH, self.CARD_HEIGHT)
        card_widget.setStyleSheet(self._card_css(card.resonance))

        layout = QVBoxLayout(card_widget)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        # Header: title + resonance dot
        header = QHBoxLayout()
        title = QLabel(card.title or "(untitled)")
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        title.setWordWrap(True)
        header.addWidget(title, stretch=1)
        dot = QLabel(self._resonance_glyph(card.resonance))
        dot.setToolTip(f"resonance: {card.resonance:.2f}")
        dot.setStyleSheet("font-size: 18px;")
        header.addWidget(dot)
        layout.addLayout(header)

        # Meta line
        meta = QLabel(f"{self._short_nodeid(card.author)} · "
                      f"{self._humanize_age(card.ts)}")
        meta.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(meta)

        # Body preview
        body_text = card.body
        if len(body_text) > 180:
            body_text = body_text[:177] + "…"
        body = QLabel(body_text)
        body.setWordWrap(True)
        body.setStyleSheet("font-size: 12px;")
        layout.addWidget(body, stretch=1)

        # Actions
        actions = QHBoxLayout()
        actions.setSpacing(6)

        deepen = QPushButton("deepen")
        deepen.setStyleSheet("font-size: 11px; padding: 3px 9px;")
        deepen.clicked.connect(lambda: self._on_deepen(card))
        actions.addWidget(deepen)

        if card.attachments:
            render = QPushButton("render")
            render.setStyleSheet("font-size: 11px; padding: 3px 9px;")
            render.clicked.connect(lambda: self._on_render(card))
            actions.addWidget(render)

        reply = QPushButton("reply")
        reply.setStyleSheet("font-size: 11px; padding: 3px 9px;")
        reply.clicked.connect(lambda: self._on_reply(card))
        actions.addWidget(reply)

        actions.addStretch()
        layout.addLayout(actions)

        # Place on the scene at top, push others down.
        proxy = self.graphics_scene.addWidget(card_widget)
        proxy.setPos(self.column_x, self.top_y)
        for existing in self.cards:
            if existing.qt_item is not None:
                existing.qt_item.setY(
                    existing.qt_item.y() + self.CARD_HEIGHT + self.CARD_GAP
                )

        card.qt_item = proxy
        self.cards.insert(0, card)

        # Register with the scene manager so it shows up in the version
        # system and the user can include feed cards in their workflow.
        if self.scene_manager is not None:
            try:
                self.scene_manager.register_parsed_item(proxy)
            except Exception as e:
                logger.debug(f"register_parsed_item: {e}")

    # ----- visual helpers -----

    def _card_css(self, resonance: float) -> str:
        # Background tint scales with resonance: cool slate at 0,
        # warm amber at 1.
        r = int(45 + (215 - 45) * resonance)
        g = int(50 + (170 - 50) * resonance)
        b = int(60 + (95 - 60) * (1 - resonance))
        return (
            f"QFrame {{ background-color: rgb({r},{g},{b}); "
            f"border-radius: 8px; color: #f0f0f0; }} "
            f"QPushButton {{ background-color: rgba(255,255,255,0.14); "
            f"color: #f0f0f0; border: none; border-radius: 4px; }} "
            f"QPushButton:hover {{ background-color: rgba(255,255,255,0.26); }} "
            f"QLabel {{ color: #f0f0f0; }} "
        )

    def _resonance_glyph(self, r: float) -> str:
        if r > 0.7:  return "●"
        if r > 0.3:  return "◐"
        return "○"

    def _short_nodeid(self, nid: str) -> str:
        return nid[:10] + "…" if len(nid) > 10 else nid

    def _humanize_age(self, ts: float) -> str:
        age = time.time() - ts
        if age < 60:    return f"{int(age)}s ago"
        if age < 3600:  return f"{int(age/60)}m ago"
        if age < 86400: return f"{int(age/3600)}h ago"
        return f"{int(age/86400)}d ago"

    # ----- action handlers -----

    def _on_deepen(self, card: Card) -> None:
        """Push identity vector toward this card's topic."""
        try:
            with open(f"{self.peribus_root}/ctl", "wb") as f:
                payload = f"attract {card.title} {card.body[:200]}\n"
                f.write(payload.encode("utf-8"))
        except OSError as e:
            logger.warning(f"deepen failed: {e}")

    def _on_render(self, card: Card) -> None:
        """
        Fetch a widget attachment, validate, and execute in the runtime.

        The attachment is identified by content hash. We read it from the
        peer's social/ dir — the daemon does the MSG_FETCH round-trip
        if we don't have the content cached locally.

        File I/O happens on a worker thread so the Qt event loop stays
        responsive during the round-trip; the actual widget execution
        and scene updates run back on the main (Qt) thread.
        """
        if not card.attachments:
            return
        attachment = card.attachments[0]
        path = f"{self.peribus_root}/nodes/{card.author}/social/{attachment}"

        async def _do_render():
            try:
                source_bytes = await asyncio.to_thread(_read_file, path)
            except OSError as e:
                self._show_card_error(card, f"fetch failed: {e}")
                return

            try:
                source = source_bytes.decode("utf-8")
            except UnicodeDecodeError:
                self._show_card_error(card, "attachment is not text")
                return

            from peribus._content import execute_widget, PeribusAPI

            # Stub PeribusAPI — a real one would talk to the daemon over
            # a side channel for stream/post/signal. For v0.1 the widget
            # gets a no-op API; visual widgets work, networked widgets
            # silently no-op on those calls.
            async def _noop_post(_text): pass
            async def _noop_signal(_n, _d): pass
            api = PeribusAPI(
                me="(local)",
                peer=card.author,
                stream_factory=lambda name: None,
                post_callback=_noop_post,
                signal_callback=_noop_signal,
            )

            try:
                execute_widget(
                    source, api,
                    {"graphics_scene": self.graphics_scene},
                )
            except ValueError as e:
                # AST validator rejected it — show why.
                self._show_card_error(card, str(e))
            except Exception as e:
                self._show_card_error(card, f"runtime: {e}")

        asyncio.create_task(_do_render(), name=f"peribus.render.{card.post_id[:8]}")

    def _on_reply(self, card: Card) -> None:
        """Tiny inline reply box — write to <peer>/inbox on Enter."""
        from PySide6.QtWidgets import QLineEdit
        line = QLineEdit()
        line.setPlaceholderText(f"reply to {self._short_nodeid(card.author)}…")
        line.setFixedWidth(self.CARD_WIDTH)
        line.setStyleSheet(
            "QLineEdit { background: #1c1c1c; color: #f0f0f0; "
            "border: 1px solid #555; border-radius: 4px; padding: 6px; }"
        )
        proxy = self.graphics_scene.addWidget(line)
        if card.qt_item is not None:
            proxy.setPos(card.qt_item.x(),
                        card.qt_item.y() + self.CARD_HEIGHT + 4)

        peribus_root = self.peribus_root

        def submit():
            text = line.text().strip()
            if text:
                try:
                    with open(f"{peribus_root}/nodes/{card.author}/inbox",
                              "wb") as f:
                        f.write(text.encode("utf-8"))
                except OSError as e:
                    logger.warning(f"reply failed: {e}")
            self.graphics_scene.removeItem(proxy)

        line.returnPressed.connect(submit)
        line.setFocus()

    def _show_card_error(self, card: Card, msg: str) -> None:
        from PySide6.QtWidgets import QLabel
        lbl = QLabel(f"⚠ {msg[:240]}")
        lbl.setStyleSheet(
            "color: #f88; background: rgba(0,0,0,0.7); "
            "font-size: 11px; padding: 6px; border-radius: 4px;"
        )
        lbl.setWordWrap(True)
        lbl.setFixedWidth(self.CARD_WIDTH)
        proxy = self.graphics_scene.addWidget(lbl)
        if card.qt_item is not None:
            proxy.setPos(card.qt_item.x(),
                        card.qt_item.y() + self.CARD_HEIGHT + 4)


def _read_file(path: str) -> bytes:
    """Helper for asyncio.to_thread."""
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Public API — one function to bring up the whole feed in rio
# ---------------------------------------------------------------------------

class FeedBridge:
    """The handle to a running feed bridge. Held by whoever started it."""

    def __init__(self, renderer: FeedRenderer, tailer: FeedTailer):
        self.renderer = renderer
        self.tailer = tailer

    def stop(self):
        """Schedule a stop; safe to call from Qt slot."""
        # Drop our pin first, so the bridge becomes collectable once the
        # tailer task finishes. Then ask the tailer to stop.
        _LIVE_BRIDGES.discard(self)
        try:
            asyncio.create_task(self.tailer.stop())
        except RuntimeError:
            # No running loop (e.g. called during shutdown). Best-effort:
            # set the stop event so the tailer exits next time it ticks.
            self.tailer._stop.set()


def stop_feed_bridge(bridge: "FeedBridge | None" = None) -> None:
    """
    Stop a specific feed bridge, or all of them if `bridge` is None.

    Convenience wrapper for use from rio's parse file / scene code.
    """
    if bridge is not None:
        bridge.stop()
        return
    for b in list(_LIVE_BRIDGES):
        b.stop()


def start_feed_bridge(
    scene_manager: Any,
    graphics_scene: Any,
    peribus_root: str = "/n/peribus",
    column_x: Optional[int] = None,
    top_y: Optional[int] = None,
) -> FeedBridge:
    """
    Start the feed bridge in the current asyncio loop.

    Call this once from rio code (or from your scene/parse file). Returns
    a FeedBridge handle. You don't need to keep the handle around — the
    bridge pins itself in a module-level set so the parser executor's
    namespace teardown can't garbage-collect it.

    Example, from /n/<workspace>/scene/parse:

        from peribus._daemon import start_feed_bridge
        start_feed_bridge(scene_manager, graphics_scene)

    To stop everything later:

        from peribus._daemon import stop_feed_bridge
        stop_feed_bridge()
    """
    import os
    feed_path = f"{peribus_root}/feed/new"
    _say(f"[peribus.feed] start_feed_bridge called")
    _say(f"[peribus.feed]   peribus_root = {peribus_root}")
    _say(f"[peribus.feed]   feed_path    = {feed_path}")
    _say(f"[peribus.feed]   feed exists  = {os.path.exists(feed_path)}")
    _say(f"[peribus.feed]   column at    = ({column_x or FeedRenderer.DEFAULT_COL_X}, " f"{top_y or FeedRenderer.DEFAULT_TOP_Y})")

    # Sanity check: if there's no Qt application yet, we can't render
    # anything anyway. Warn loudly rather than fail silently downstream.
    try:
        from PySide6.QtCore import QCoreApplication
        if QCoreApplication.instance() is None:
            _say(f"[peribus.feed] WARNING: no QApplication — cards cannot render")
    except ImportError:
        _say(f"[peribus.feed] WARNING: PySide6 not importable")

    renderer = FeedRenderer(
        graphics_scene, scene_manager, peribus_root,
        column_x=column_x, top_y=top_y,
    )
    tailer = FeedTailer(
        feed_path=feed_path,
        on_card=renderer.add_card,
    )
    tailer.start()

    # Confirm the tailer task is actually scheduled.
    if tailer._task is None:
        _say(f"[peribus.feed] WARNING: tailer task not scheduled")
    else:
        _say(f"[peribus.feed] tailer task scheduled — waiting for posts")

    bridge = FeedBridge(renderer, tailer)
    # Pin so the parser executor's namespace teardown can't GC us.
    _LIVE_BRIDGES.add(bridge)
    return bridge