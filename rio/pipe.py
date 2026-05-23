"""
pipe.py — One unified abstraction for every filesystem interaction.
==================================================================

The operator's job is to move bytes between paths on the 9p filesystem.
Everything else — node types, port colors, scene layout — is decoration
on top of that. This file is the only place in the operator that knows
about `os.open`, `os.read`, `os.write`, or threading.

Three concepts, in order of importance:

  1. `FSWorker` — runs a callable on a daemon thread, fires its result
     back on the Qt main thread via signal. Used by everything that
     wants a non-blocking syscall.

  2. `Pipe` — a path on the filesystem. Exposes `read_async`,
     `write_async`, and `subscribe`. Knows nothing about node types or
     ports. Two `Pipe(path)` instances against the same path are
     equivalent and independent.

  3. `Subscription` — what `Pipe.subscribe` returns. Has `.stop()`. The
     subscription strategy (parked thread vs polling timer) is hidden
     from the caller; pick a `SubscribeMode` if you want to override
     the default chosen from the basename.

Design rules (these are what we're trading the old mess for):

  - One read path, one write path. Bytes only. Decode/encode is the
    caller's job. (TextNode does `data.decode(errors='replace')`.)

  - The "uppercase basename means blocking" convention is a *default*
    for `subscribe`, never a branch the caller writes. Caller-facing
    code says `pipe.subscribe(cb)` and doesn't care.

  - No syscall on the Qt thread. `read_async` and `write_async` always
    dispatch to `FSWorker`. The synchronous `read`/`write` exist for
    code that's already on a worker thread or for tests.

  - POLL dedupe is content-aware but caller-tunable. Default hashes the
    full bytes; large payloads (media) pass `dedupe_key=lambda d: (len(d),
    d[:64], d[-64:])` for effectively-free change detection. The hash
    runs on the Qt thread, so the default's O(n) cost matters at scale.

  - Subscription `stop()` is best-effort but always safe. A parked
    thread inside `os.read` can't be interrupted from outside without
    closing its fd from another thread (which on 9p risks the server's
    clunk semantics). We accept that: the thread is daemon, late emits
    are filtered out by the Subscription's `_stopped` flag, and the
    thread dies with the process or with its next read returning.

Things this file deliberately does NOT do:

  - inotify / fanotify. POLL with dedupe is enough for current
    workloads; WATCH is reserved in the enum for a future drop-in.

  - Multiplexing. Each Pipe owns its own fd lifecycle. If two nodes
    subscribe to the same path, that's two threads. The 9p server is
    the one place where multiplexing should happen (one Plan9Attachment
    per source), and it already does.

  - Route management. That lives in `graph.py` on top of a Pipe pointed
    at /n/<m>/routes. Routes are not special here.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional

from PySide6.QtCore import QObject, QProcess, QTimer, Signal, Slot


# ─── Tunables ───────────────────────────────────────────────────────────
# Kept module-level so they're discoverable; override per-Pipe via kwargs.

DEFAULT_MAX_BYTES = 4 * 1024 * 1024        # 4 MiB, defends against runaway reads
DEFAULT_POLL_MS = 1000                     # POLL subscription tick
STREAM_DOWNGRADE_GUARD_MS = 50             # if STREAM gets EOF this fast on
                                           # first read, the path isn't actually
                                           # blocking — downgrade to POLL
STREAM_BACKOFF_MS = 2000                   # error backoff inside STREAM loop


# Marker for "no previous value seen" in dedupe state. We can't use None
# because a caller's `dedupe_key` is allowed to return None as a legitimate
# key. `object()` gives us a unique identity for is-comparison.
_SENTINEL = object()


# ─── Public types ───────────────────────────────────────────────────────


class SubscribeMode(Enum):
    """How a `Pipe.subscribe` consumes its path.

    STREAM   parked thread, repeatedly `open → read-to-EOF → emit → close`.
             For llmfs streaming files (uppercase basenames: OUTPUT, PYTHON,
             STDERR) where EOF marks "one finished generation".
             Cost: one daemon thread per subscription, parked in os.read.

    POLL     QTimer + non-blocking read; emit only when content changed.
             For regular files and any non-blocking source.
             Cost: one Qt-thread tick per `poll_ms`. No threads of its own.

    WATCH    reserved. Future inotify/fanotify implementation; falls back to
             POLL for now. Caller can use it today and benefit automatically
             when WATCH lands.

    MANUAL   never starts a subscription. The Pipe is read on demand via
             `read_async`. Use when the node is button-driven.
    """
    STREAM = "stream"
    POLL = "poll"
    WATCH = "watch"
    MANUAL = "manual"


@dataclass
class ReadError(Exception):
    """A read failed. The path may be gone, permission-denied, or the 9p
    server may have hung up. Subscriptions emit this through their callback
    so the node can flash a status; one-shot reads return it via the
    `on_done` callback (so callers can `isinstance(result, ReadError)`)."""
    path: str
    cause: str

    def __str__(self) -> str:
        return f"read {self.path}: {self.cause}"


# ─── FSWorker ───────────────────────────────────────────────────────────


class FSWorker(QObject):
    """Run callables on daemon threads; deliver results on the Qt main thread.

    Stateless from the caller's perspective:

        worker.run_async(my_fn, arg1, arg2, on_done=lambda r: ...)

    `on_done` is invoked exactly once, on the Qt thread, with either the
    return value or the raised exception. Callers check `isinstance(r,
    Exception)` if they care.

    There is no timeout, by design: 9p reads can legitimately park for
    minutes (waiting for an agent generation to finish). The old operator
    had a 2-second timeout that silently dropped legitimate long reads.
    If a caller needs a deadline, it can implement one by holding the
    Subscription handle and calling `.stop()` after its own QTimer.
    """

    _result = Signal(int, object)   # (tag, result_or_exception)

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self._next_tag = 0
        self._pending: Dict[int, Callable] = {}
        self._result.connect(self._dispatch)

    def run_async(self, fn: Callable, *args,
                  on_done: Optional[Callable] = None) -> None:
        tag = self._next_tag
        self._next_tag += 1
        if on_done is not None:
            self._pending[tag] = on_done
        threading.Thread(
            target=self._run, args=(tag, fn, args),
            daemon=True,
        ).start()

    def _run(self, tag: int, fn: Callable, args: tuple) -> None:
        try:
            result = fn(*args)
        except Exception as exc:
            result = exc
        try:
            self._result.emit(tag, result)
        except RuntimeError:
            # QObject deleted while a thread was in flight; drop silently.
            pass

    @Slot(int, object)
    def _dispatch(self, tag: int, result: object) -> None:
        cb = self._pending.pop(tag, None)
        if cb is None:
            return
        try:
            cb(result)
        except Exception:
            traceback.print_exc()


# ─── Pipe ───────────────────────────────────────────────────────────────


def _basename_is_uppercase(path: str) -> bool:
    """llmfs's visual convention: uppercase basename = streaming/blocking
    read. We need at least one letter, and all letters must be uppercase.
    Digits and underscores are allowed in either case."""
    base = os.path.basename(path) if path else ""
    if not base or base.startswith("."):
        return False
    letters = [c for c in base if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _raw_read(path: str, max_bytes: int) -> bytes:
    """Blocking read-to-EOF (or to max_bytes). The right primitive for
    STREAM (caller wants the whole finished generation) AND for one-shot
    `read_async` against any path (a regular-file read returns immediately;
    a blocking-file read parks until finish() — both correct).

    Returns b'' on OSError. STREAM checks for empty returns to detect the
    blocking-tagged-but-actually-not-blocking case and downgrade."""
    try:
        with open(path, "rb") as f:
            return f.read(max_bytes)
    except OSError:
        return b""


def _raw_read_nonblocking(path: str, max_bytes: int) -> bytes:
    """Read up to `max_bytes` from `path`. Used by POLL ticks.

    Despite the name (kept for backcompat), this uses *blocking* I/O
    via Python's open() — read-to-EOF semantics. The old O_NONBLOCK
    variant was wrong for 9p multi-chunk replies: os.read after the
    first Tread reply returns BlockingIOError (EAGAIN) before the
    server's second reply arrives, and the read would break with a
    truncated buffer. Symptom: image files returning the first 39
    bytes and never the rest.

    For non-blocking-tagged paths (lowercase basenames) this read
    completes quickly because the server replies with all available
    bytes and EOFs. For blocking paths (uppercase basenames) Pipe
    would have routed through _StreamSubscription instead, not this
    function, so we don't have to worry about parking forever.

    Returns b'' on any error — callers treat that as "no data this
    tick" and try again next tick.
    """
    try:
        with open(path, "rb") as f:
            return f.read(max_bytes)
    except OSError:
        return b""


def _raw_write(path: str, data: bytes) -> int:
    """Write `data` to `path`, truncating any existing contents.

    9p endpoints interpret each Twrite as one command, so we use a single
    `os.write` call (not Python's text-mode buffering which can split).
    Parent directories are created if missing — that's a no-op on 9p
    files but lets the same call work for /tmp/* scratch paths.

    Returns the number of bytes written, or raises OSError."""
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except OSError:
        pass
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags)
    try:
        return os.write(fd, data)
    finally:
        os.close(fd)


class Pipe:
    """A path on the filesystem. Provides async one-shot I/O and a
    `subscribe` operation whose strategy is picked from the basename
    (or overridden by the caller).

    Two Pipe instances against the same path are independent and
    equivalent: there's no shared state. Reads don't interfere with
    writes; subscriptions don't interfere with one-shot reads. (The
    9p server may serialize accesses to a single backing file, but
    that's a server concern.)

    Typical use from a node:

        self.in_pipe = Pipe("/n/m/nodes/text_foo/in", worker)
        self.out_pipe = Pipe("/n/m/nodes/text_foo/OUT", worker)

        # one-shot
        self.in_pipe.read_async(self._on_read)
        self.out_pipe.write_async(b"hello\\n")

        # continuous
        self._sub = self.in_pipe.subscribe(self._on_chunk)
        # ... later
        self._sub.stop()
    """

    def __init__(self, path: str, worker: FSWorker,
                 max_bytes: int = DEFAULT_MAX_BYTES):
        self.path = path
        self._worker = worker
        self._max_bytes = max_bytes

    # ── introspection ────────────────────────────────────────────────

    @property
    def default_mode(self) -> SubscribeMode:
        """Best-guess subscription mode based on the basename. Callers
        can use this for UI hints ("this port streams") or just let
        `subscribe()` pick it implicitly."""
        return (SubscribeMode.STREAM if _basename_is_uppercase(self.path)
                else SubscribeMode.POLL)

    def __repr__(self) -> str:
        return f"Pipe({self.path!r}, default={self.default_mode.value})"

    # ── one-shot I/O ─────────────────────────────────────────────────

    def read_async(self, on_done: Callable[[bytes | ReadError], None]) -> None:
        """Read the path on a worker thread; deliver bytes (or ReadError)
        on the Qt main thread.

        Uses the blocking-read primitive — for non-blocking paths that's
        a fast EOF-to-EOF read; for blocking paths the worker thread
        parks until the producer calls finish(). Either is correct."""

        path = self.path
        max_bytes = self._max_bytes

        def _read() -> bytes | ReadError:
            try:
                with open(path, "rb") as f:
                    return f.read(max_bytes)
            except OSError as e:
                return ReadError(path=path, cause=str(e))

        self._worker.run_async(_read, on_done=on_done)

    def write_async(self, data: bytes,
                    on_done: Optional[Callable[[int | OSError], None]] = None
                    ) -> None:
        """Write `data` to the path on a worker thread.

        On 9p endpoints `data` is one Twrite — one command. Callers that
        want append semantics should compose the full payload themselves.

        `on_done(n_bytes)` fires on success; `on_done(exc)` fires on
        OSError. If `on_done` is None, errors are swallowed (acceptable
        for fire-and-forget writes like ctl commands)."""

        path = self.path

        def _write() -> int | OSError:
            try:
                return _raw_write(path, data)
            except OSError as e:
                return e

        self._worker.run_async(_write, on_done=on_done)

    # ── subscription ─────────────────────────────────────────────────

    def subscribe(self, on_chunk: Callable[[bytes], None], *,
                  mode: Optional[SubscribeMode] = None,
                  poll_ms: int = DEFAULT_POLL_MS,
                  dedupe: bool = True,
                  dedupe_key: Optional[Callable[[bytes], object]] = None,
                  on_error: Optional[Callable[[ReadError], None]] = None,
                  ) -> "Subscription":
        """Start a continuous read of the path. Returns a `Subscription`
        whose `.stop()` ends it.

        `mode`        overrides the basename-derived default. STREAM parks
                      a thread; POLL uses a QTimer. WATCH currently aliases
                      to POLL. MANUAL returns an inert subscription whose
                      `stop()` is a no-op (occasionally useful as a sentinel).

        `poll_ms`     POLL-only. Ignored by STREAM.

        `dedupe`      POLL-only. If True (default), emit only when the
                      content's dedupe key differs from the last emit. Set
                      False for "always re-emit on poll" semantics.

        `dedupe_key`  POLL-only. Function `bytes -> hashable` that picks
                      what to compare for dedupe. Default: `hash(bytes)`,
                      which costs O(n) per tick. For large payloads (media
                      bytes) pass a cheap key like
                      `lambda d: (len(d), d[:64], d[-64:])`. The function
                      runs on the Qt main thread, so keep it fast.

        `on_error`    called (on Qt thread) when a read fails. STREAM also
                      backs off briefly to avoid a hot error loop. If
                      omitted, errors are swallowed."""

        resolved = mode or self.default_mode

        if resolved is SubscribeMode.STREAM:
            return _StreamSubscription(self, on_chunk, on_error)
        if resolved is SubscribeMode.POLL or resolved is SubscribeMode.WATCH:
            # WATCH falls back to POLL until inotify lands. Code that
            # wants WATCH semantics today still gets correct behavior;
            # we just pay the polling cost.
            return _PollSubscription(self, on_chunk, on_error,
                                     poll_ms=poll_ms, dedupe=dedupe,
                                     dedupe_key=dedupe_key)
        if resolved is SubscribeMode.MANUAL:
            return _ManualSubscription()
        raise ValueError(f"unknown SubscribeMode: {resolved}")


# ─── Subscription implementations ───────────────────────────────────────


class Subscription:
    """A live or dormant subscription. Has `.stop()` (idempotent) and
    `.is_active`. Subclasses pick their own strategy."""

    def stop(self) -> None:
        raise NotImplementedError

    @property
    def is_active(self) -> bool:
        raise NotImplementedError


class _ManualSubscription(Subscription):
    """Inert sub returned by `subscribe(mode=MANUAL)`. `stop()` is a no-op
    so caller code that always calls `sub.stop()` on teardown is safe."""

    def stop(self) -> None:
        pass

    @property
    def is_active(self) -> bool:
        return False


class _PollSubscription(QObject, Subscription):
    """Push subscription via QProcess `cat` respawn — NOT actually
    polling, despite the legacy class name.

    Strategy: spawn `cat <path>` once. Server-side files like BufferFile
    and RoutesFile block-on-rearm: the first read returns the current
    bytes (or empty if nothing was written yet), then close. A second
    consecutive open blocks server-side until something writes the
    file. So we spawn cat, await its exit (which means data is ready),
    parse stdout, fire `_on_chunk`, and *immediately* respawn — the
    next cat will park on the server until the next write.

    No QTimer, no `poll_ms` honored. The constructor still accepts
    `poll_ms` to keep Pipe.subscribe()'s call signature stable; it's
    just ignored. Subprocess overhead per change is one fork+exec
    rather than one fork+exec per second, which is the win.

    For static files (no server-side push), the server will EOF
    immediately and cat returns immediately every time, degrading to a
    busy spin. We guard that with a backoff: if cat returns in under
    `_STATIC_GUARD_MS`, wait `_STATIC_GUARD_MS` before respawning. So
    a buggy static path costs at most ~10 ticks/s, not the kernel-
    saturating spin a naive loop would create.
    """

    _STATIC_GUARD_MS = 100   # min interval between cats if EOF is instant

    def __init__(self, pipe: Pipe,
                 on_chunk: Callable[[bytes], None],
                 on_error: Optional[Callable[[ReadError], None]],
                 *, poll_ms: int, dedupe: bool,
                 dedupe_key: Optional[Callable[[bytes], object]] = None):
        QObject.__init__(self)
        self._pipe = pipe
        self._on_chunk = on_chunk
        self._on_error = on_error
        self._dedupe = dedupe
        self._dedupe_key = dedupe_key or hash
        self._last_key: object = _SENTINEL
        self._stopped = False
        self._proc: Optional[QProcess] = None
        self._spawn_ts: float = 0.0
        # First cat on the next event-loop turn so construction returns.
        QTimer.singleShot(0, self._spawn_cat)

    def _spawn_cat(self) -> None:
        if self._stopped:
            return
        if self._proc is not None and \
                self._proc.state() != QProcess.NotRunning:
            return
        proc = QProcess(self)
        self._proc = proc
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.finished.connect(
            lambda _code, _status, p=proc: self._on_cat_done(p))
        proc.errorOccurred.connect(
            lambda _err, p=proc: self._on_cat_error(p))
        self._spawn_ts = time.monotonic()
        proc.start("cat", [self._pipe.path])

    def _on_cat_done(self, proc: "QProcess") -> None:
        if self._stopped or proc is not self._proc:
            return
        data = bytes(proc.readAllStandardOutput())
        self._proc = None
        elapsed_ms = (time.monotonic() - self._spawn_ts) * 1000
        # Respawn: immediate if the cat took meaningful time (real
        # blocking), throttled if it EOF'd instantly (static file or
        # no push wired on the server side).
        if elapsed_ms < self._STATIC_GUARD_MS:
            QTimer.singleShot(self._STATIC_GUARD_MS, self._spawn_cat)
        else:
            QTimer.singleShot(0, self._spawn_cat)
        # Empty read: nothing to deliver (file empty or doesn't exist).
        if not data:
            self._last_key = _SENTINEL
            return
        if self._dedupe:
            try:
                k = self._dedupe_key(data)
            except Exception:
                traceback.print_exc()
                k = _SENTINEL
            if k is not _SENTINEL and k == self._last_key:
                return
            self._last_key = k
        try:
            self._on_chunk(data)
        except Exception:
            traceback.print_exc()

    def _on_cat_error(self, proc: "QProcess") -> None:
        if self._stopped or proc is not self._proc:
            return
        err = proc.errorString()
        self._proc = None
        if self._on_error:
            self._on_error(ReadError(path=self._pipe.path,
                                     cause=err or "cat failed"))
        # Back off and retry, in case the file appears later.
        QTimer.singleShot(self._STATIC_GUARD_MS * 5, self._spawn_cat)

    def stop(self) -> None:
        self._stopped = True
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

    @property
    def is_active(self) -> bool:
        return not self._stopped


class _StreamSubscription(QObject, Subscription):
    """Parked-thread subscription for blocking llmfs streaming files.

    Runs `while not stop: open → read-to-EOF → emit → close → repeat` on
    its own daemon thread. Each completed read is one finished generation
    of the upstream producer; we emit the whole payload, never a partial.

    Includes a first-iteration downgrade check: if the very first read
    returns immediately with empty bytes (within STREAM_DOWNGRADE_GUARD_MS),
    the path isn't really blocking — we transparently swap ourselves out
    for a `_PollSubscription` against the same callback. The caller's
    `.stop()` still works because we hold a reference to the replacement.

    `stop()` flags the loop; the thread may still be parked in os.read
    until the producer EOFs (no way to interrupt that cleanly on 9p),
    but late emits are filtered by the `_stopped` flag and the thread is
    daemon, so worst case it dies with the process. This is the same
    trade the original _TextNodeTailer made; we accept it consciously."""

    _emit_chunk = Signal(bytes)
    _emit_error = Signal(object)   # ReadError
    _emit_downgrade = Signal()     # internal: worker thread → Qt thread swap

    def __init__(self, pipe: Pipe,
                 on_chunk: Callable[[bytes], None],
                 on_error: Optional[Callable[[ReadError], None]]):
        QObject.__init__(self)
        self._pipe = pipe
        self._on_chunk = on_chunk
        self._on_error = on_error
        self._stop = threading.Event()
        self._downgrade: Optional[_PollSubscription] = None

        self._emit_chunk.connect(self._dispatch_chunk)
        self._emit_error.connect(self._dispatch_error)
        self._emit_downgrade.connect(self._do_downgrade_swap)

        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"pipe-stream:{os.path.basename(pipe.path)}",
        )
        self._thread.start()

    def _loop(self) -> None:
        path = self._pipe.path
        max_bytes = self._pipe._max_bytes
        first_iteration = True

        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                with open(path, "rb") as f:
                    data = f.read(max_bytes)
            except OSError as e:
                if self._stop.is_set():
                    return
                try:
                    self._emit_error.emit(ReadError(path=path, cause=str(e)))
                except RuntimeError:
                    return
                # Back off so a missing path doesn't pin the CPU.
                if self._stop.wait(STREAM_BACKOFF_MS / 1000.0):
                    return
                continue

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            if first_iteration:
                first_iteration = False
                # If the very first read EOFed instantly with no data,
                # this path is not actually blocking. Hand off to POLL.
                if not data and elapsed_ms < STREAM_DOWNGRADE_GUARD_MS:
                    self._request_downgrade()
                    return

            if self._stop.is_set():
                return

            if not data:
                # Empty read on a path we believed was blocking, but not
                # instantly empty — could be a producer that finish()ed
                # without writing. Back off briefly and retry rather than
                # spinning.
                if self._stop.wait(STREAM_BACKOFF_MS / 1000.0):
                    return
                continue

            try:
                self._emit_chunk.emit(bytes(data))
            except RuntimeError:
                return

    def _request_downgrade(self) -> None:
        """Called from the worker thread when the path turns out to be
        non-blocking. We emit a signal — Qt marshals it to the main
        thread — and `_do_downgrade_swap` performs the swap there.

        Why not QTimer.singleShot from this thread: timers need a Qt
        event loop on the calling thread, which the worker thread
        doesn't have. Signals with a queued connection (the default for
        cross-thread signal/slot) are the canonical way to bounce work
        to the Qt main thread.

        Why we don't have the caller's `dedupe_key` here: subscribe()
        didn't pass it to the StreamSubscription because STREAM doesn't
        dedupe. The downgraded POLL uses default `hash`. Acceptable for
        the downgrade case — paths that auto-downgrade are by definition
        small (instantly-empty was the trigger) and full-hash is cheap.
        """
        try:
            self._emit_downgrade.emit()
        except RuntimeError:
            # QObject destroyed mid-flight; nothing to do.
            pass

    @Slot()
    def _do_downgrade_swap(self) -> None:
        """Qt-main-thread slot that performs the swap from STREAM to POLL."""
        if self._stop.is_set():
            return
        if self._downgrade is not None:
            return  # already swapped
        self._downgrade = _PollSubscription(
            self._pipe, self._on_chunk, self._on_error,
            poll_ms=DEFAULT_POLL_MS, dedupe=True,
        )

    @Slot(bytes)
    def _dispatch_chunk(self, data: bytes) -> None:
        if self._stop.is_set():
            return
        try:
            self._on_chunk(data)
        except Exception:
            traceback.print_exc()

    @Slot(object)
    def _dispatch_error(self, err: object) -> None:
        if self._stop.is_set() or self._on_error is None:
            return
        try:
            self._on_error(err)
        except Exception:
            traceback.print_exc()

    def stop(self) -> None:
        self._stop.set()
        if self._downgrade is not None:
            self._downgrade.stop()

    @property
    def is_active(self) -> bool:
        if self._stop.is_set():
            return False
        if self._downgrade is not None:
            return self._downgrade.is_active
        return self._thread.is_alive()