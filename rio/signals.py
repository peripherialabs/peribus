"""
rio.signals — Cross-machine PySide6 Signal pubsub over UDP.

Goal: make this work, verbatim, in any rio user namespace:

    subscribe("ekanza")
    text_spoken = Signal(str)
    text_spoken.connect(lambda msg: print("got:", msg))

When code on machine "ekanza" emits its own `text_spoken` signal, this
machine's `text_spoken` fires with the same args. The Qt signal works
locally too: `text_spoken.emit("hi")` runs all local slots AND broadcasts
to every machine subscribed to us.

Design
------
* Each rio binds one UDP socket. Packets are JSON (small, simple).
* Two registries:
    LOCAL  signal_name -> RemoteSignal      (objects in this namespace)
    PEERS  machine_name -> (host, signal_port)
* `subscribe(machine)` tells the bus "deliver emits from <machine> to me".
* `RemoteSignal.emit(*args)` does Qt emit + UDP broadcast to every peer
  that has subscribed to *us* (we track inbound subscribers).

Wire protocol (UDP, JSON, one packet per message)
-------------------------------------------------
    {"v":1,"k":"hello","from":"machine_a","sigs":{"text_spoken":["str"]}}
    {"v":1,"k":"sub",  "from":"machine_a"}                  # subscribe to peer
    {"v":1,"k":"unsub","from":"machine_a"}
    {"v":1,"k":"emit", "from":"machine_a","name":"text_spoken","args":["hi"]}
    {"v":1,"k":"bye",  "from":"machine_a"}

UDP fits because signals are tiny and idempotent slots are rare; if a
packet drops, that's the same observable behavior as a dropped click.
Anything bigger than ~60 KB you should not be putting on a Signal.

Threading
---------
The bus owns a small asyncio loop on a background thread. All inbound
packets land there, then dispatch hops to the Qt main thread via
`QMetaObject.invokeMethod` (queued) so user slots run on the GUI
thread — which is what Qt expects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import threading
import time
import weakref
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("rio.signals")

# ── Qt import (lazy/optional) ────────────────────────────────────────
# We tolerate headless rio: if PySide6 is unavailable, RemoteSignal
# degrades to a plain callback list. Real rio always has Qt.

try:
    from PySide6.QtCore import (
        QObject, Signal as _QSignal, QMetaObject, Qt, Q_ARG, Slot,
    )
    _HAS_QT = True
except ImportError:  # pragma: no cover
    _HAS_QT = False
    QObject = object  # type: ignore


# ── Wire constants ───────────────────────────────────────────────────

PROTO_VERSION = 1
DEFAULT_SIGNAL_PORT_OFFSET = 100   # signal_port = 9p_port + 100
MAX_PACKET = 65000                  # safe UDP payload ceiling
HEARTBEAT_INTERVAL = 20.0           # peers re-send `sub` so we re-arm subs

# Map JSON type tags <-> python types. RemoteSignal stores types as
# strings on the wire so peers don't have to agree on Python class
# identity; we keep things primitive on purpose.
_TYPE_TAGS = {
    str: "str", int: "int", float: "float", bool: "bool",
    bytes: "bytes", list: "list", dict: "dict", tuple: "tuple",
    type(None): "None",
}
_TAG_TO_TYPE = {v: k for k, v in _TYPE_TAGS.items()}


def _type_tag(t: type) -> str:
    return _TYPE_TAGS.get(t, getattr(t, "__name__", "object"))


# ────────────────────────────────────────────────────────────────────
# RemoteSignal — Qt Signal that also crosses the wire
# ────────────────────────────────────────────────────────────────────
#
# Why a factory class instead of a plain subclass?  PySide6 needs the
# Signal to be declared as a class attribute at class-definition time
# (the metaclass collects them). We can't add a Signal to an existing
# QObject subclass on the fly. So Signal(str, int) returns an instance
# of a freshly minted QObject subclass that has exactly one Signal
# attribute named `_qsig` with the right type signature.
#
# From the user's perspective it just looks like a Signal:
#
#     text_spoken = Signal(str)
#     text_spoken.connect(slot)
#     text_spoken.emit("hi")
#
# We keep an internal weak registry of all live RemoteSignal instances
# so the bus can deliver inbound emits to the matching named signal.

# Cache of dynamically created QObject *holder* classes keyed by type
# tuple. The holder is a minimal QObject subclass with one Signal
# attribute named `_qsig`. The RemoteSignal wrapper composes a holder
# rather than subclassing QObject itself — that way our `.connect` /
# `.disconnect` / `.emit` methods aren't subject to PySide6's C++
# method-resolution dispatch (which silently routes `connect` to
# `QObject.connect` when the subclass was built dynamically via
# `type()`, ignoring our Python override). Composition is also a hair
# faster because we skip QObject construction overhead per call site
# where there's no Qt available.

_holder_cache: Dict[tuple, type] = {}


def _holder_class(types: tuple) -> Optional[type]:
    if not _HAS_QT:
        return None
    cls = _holder_cache.get(types)
    if cls is None:
        # Class body must declare the Signal at class-definition time
        # so PySide6's metaclass picks it up. We accomplish that by
        # passing the attribute dict to `type()` — this works for
        # QObject subclasses as long as we keep the body minimal
        # (just the Signal). Adding Python methods here would land us
        # in the same dispatch trap that motivated composition.
        name_part = "_".join(_type_tag(t) for t in types) or "void"
        cls = type(
            f"_SignalHolder__{name_part}",
            (QObject,),
            {"_qsig": _QSignal(*types)},
        )
        _holder_cache[types] = cls
    return cls


# Define the QObject dispatcher class ONCE — it has a single Slot that
# accepts an `object` (a list of decoded args) and calls back into the
# RemoteSignal. We need a Slot so `QMetaObject.invokeMethod` can find
# it. Since the args are wrapped in a single Python list, the slot
# signature is fixed and doesn't depend on the user's type tuple.

if _HAS_QT:
    class _RemoteDispatcher(QObject):
        """
        Lives on the GUI thread, owns the Signal that mirrors a
        RemoteSignal. Has one Slot('QVariantList') that the bus invokes
        via QueuedConnection from the bus thread; the slot then emits
        the Qt Signal back on the GUI thread.

        Why `'QVariantList'` and not `list` or `object`?  PySide6's
        QMetaObject.invokeMethod needs a Qt-side type id. `list` and
        `object` aren't registered Qt meta-types in PySide6, so
        `Q_ARG(list, ...)` and `Q_ARG(object, ...)` both fail with
        `qArgDataFromPyType: Unable to find a QMetaType for ...`.
        `QVariantList` is the canonical Qt name for "list of QVariant"
        — it's the type Qt uses internally for JS-array-like data and
        round-trips arbitrary Python lists fine.

        Subclassed dynamically per type tuple so the Signal has the
        right signature.
        """
        # Subclasses inject `_qsig = Signal(*types)`.
        _qsig: Any = None

        def __init__(self, owner_ref):
            super().__init__()
            self._owner_ref = owner_ref  # weakref to RemoteSignal

        @Slot("QVariantList")
        def _deliver(self, args):
            try:
                self._qsig.emit(*args)
            except TypeError as e:
                owner = self._owner_ref() if self._owner_ref else None
                name = owner._name if owner else "?"
                logger.warning(
                    "type mismatch delivering %s: %s (args=%r)",
                    name, e, args,
                )

    _dispatcher_cache: Dict[tuple, type] = {}

    def _dispatcher_class(types: tuple) -> type:
        cls = _dispatcher_cache.get(types)
        if cls is None:
            name_part = "_".join(_type_tag(t) for t in types) or "void"
            cls = type(
                f"_RemoteDispatcher__{name_part}",
                (_RemoteDispatcher,),
                {"_qsig": _QSignal(*types)},
            )
            _dispatcher_cache[types] = cls
        return cls
else:
    def _dispatcher_class(types: tuple):
        return None


class RemoteSignal:
    """
    Plain Python wrapper that quacks like `PySide6.QtCore.Signal`.

    Internally holds a small QObject (`self._dispatcher`) whose Signal
    is the one user slots actually connect to. `emit(...)` runs both
    halves: local Qt emit + UDP broadcast to subscribed peers.

    Why a wrapper instead of subclassing QObject directly?
      * `type()`-created QObject subclasses don't honor Python-level
        `connect` / `disconnect` / `emit` overrides — PySide6's
        Shiboken metaclass routes those names straight to `QObject`
        slots, swallowing our logic.
      * Composition keeps our object trivially picklable / inspectable
        for the namespace adopt step (we just compare `id`s).

    Anatomy:
        self._dispatcher : QObject  (with `_qsig: Signal(*types)`)
        self._qsig       : SignalInstance — the bound signal you can
                           .connect / .disconnect; we expose it directly
                           in case the user does `s._qsig.connect(...)`,
                           but the public path is `s.connect(...)`.
        self._name       : str | None — the variable name; set later
                           by SignalBus.adopt_namespace.
        self._type_tags  : (str, ...) — for wire serialization.
    """

    __slots__ = (
        "_dispatcher", "_qsig", "_types", "_type_tags",
        "_name", "_bus", "_registered", "_fallback_slots",
        "__weakref__",
    )

    def __init__(self, types: tuple, *, _bus: Optional["SignalBus"] = None):
        self._types = types
        self._type_tags = tuple(_type_tag(t) for t in types)
        self._name: Optional[str] = None
        self._bus = _bus if _bus is not None else _global_bus()
        self._registered = False
        self._fallback_slots: List[Callable] = []

        if _HAS_QT:
            cls = _dispatcher_class(types)
            # weakref so the dispatcher doesn't keep us alive
            self._dispatcher = cls(weakref.ref(self))
            # SignalInstance — what user connects to.
            self._qsig = self._dispatcher._qsig
        else:
            self._dispatcher = None
            self._qsig = None

        # Hand ourselves to the bus so it can name us during the next
        # namespace scan. Bus may be None during tests / early import
        # — the local half (Qt emit + connect) still works; only
        # network delivery is dormant until a bus is initialized.
        if self._bus is not None:
            self._bus._track_anonymous(self)

    # ── Public API mirroring PySide6.Signal ─────────────────────────

    def connect(self, slot: Callable, *args, **kwargs):
        if _HAS_QT and self._qsig is not None:
            return self._qsig.connect(slot, *args, **kwargs)
        self._fallback_slots.append(slot)

    def disconnect(self, slot: Optional[Callable] = None):
        if _HAS_QT and self._qsig is not None:
            if slot is None:
                return self._qsig.disconnect()
            return self._qsig.disconnect(slot)
        if slot is None:
            self._fallback_slots.clear()
        else:
            try:
                self._fallback_slots.remove(slot)
            except ValueError:
                pass

    def emit(self, *args):
        """
        Local Qt emit + remote broadcast.

        Qt emit happens synchronously on the calling thread (this is
        how plain PySide6 Signals behave for direct connections —
        connected slots run inline). The remote half is fire-and-
        forget: we hand the args off to the bus, which serializes and
        sendto()s them on its own loop thread.
        """
        if _HAS_QT and self._qsig is not None:
            self._qsig.emit(*args)
        else:
            for slot in list(self._fallback_slots):
                try:
                    slot(*args)
                except Exception:
                    logger.exception("fallback slot raised")

        if self._registered and self._name and self._bus is not None:
            self._bus._broadcast_emit(self._name, args, self._type_tags)

    # ── Internal: called by the bus when a packet arrives ───────────

    def _deliver_remote(self, args: list):
        """
        Bus invokes this from the bus thread. Marshal onto the GUI
        thread via QueuedConnection so user slots run where Qt expects.
        """
        if _HAS_QT and self._dispatcher is not None:
            try:
                QMetaObject.invokeMethod(
                    self._dispatcher,
                    "_deliver",
                    Qt.QueuedConnection,
                    Q_ARG("QVariantList", list(args)),
                )
            except Exception:
                logger.exception("invokeMethod failed for %s", self._name)
        else:
            # Headless: run synchronously on the bus thread. Fine for
            # tests; production rio always has Qt.
            for slot in list(self._fallback_slots):
                try:
                    slot(*args)
                except Exception:
                    logger.exception("fallback slot raised (remote)")

    def __repr__(self):
        return f"<RemoteSignal {self._name or '?'}({','.join(self._type_tags)})>"


# Backward-compat alias used elsewhere in this module.
_SignalBase = RemoteSignal


def Signal(*types) -> RemoteSignal:  # noqa: N802 (intentional Signal-style API)
    """
    Create a new cross-machine RemoteSignal instance.

    This is the rio.signals module-level factory. In the rio user
    namespace (parsed code blocks) the same factory is exposed as
    `RemoteSignal` — that's the name to use in user code, to keep
    it visually distinct from PySide6's plain `Signal`:

        local  = Signal(str)          # PySide6, local only
        shared = RemoteSignal(str)    # crosses subscribed peers

    Type signature shape mirrors PySide6.Signal — no payload,
    one type, or many:

        empty    = RemoteSignal()
        text     = RemoteSignal(str)
        position = RemoteSignal(int, int)

    The variable name on the left side becomes the signal's network
    name — auto-detected by the bus after the parse cycle finishes
    (it scans the namespace and assigns names to anonymous
    RemoteSignals).

    For non-parse-namespace use (your own modules, internal helpers),
    use `make_signal(name, *types)` to set the name explicitly.
    """
    return RemoteSignal(types)


def make_signal(name: str, *types) -> RemoteSignal:
    """Explicit named-signal factory; bypasses namespace auto-detection."""
    sig = RemoteSignal(types)
    sig._name = name
    bus = _global_bus()
    if bus is not None:
        bus._register_named(sig)
    return sig


# ────────────────────────────────────────────────────────────────────
# Wire codec
# ────────────────────────────────────────────────────────────────────

def _encode_arg(arg: Any) -> Any:
    """Make `arg` JSON-safe. bytes -> hex; tuples -> lists; rest passthrough."""
    if isinstance(arg, bytes):
        return {"__b": arg.hex()}
    if isinstance(arg, tuple):
        return [_encode_arg(a) for a in arg]
    if isinstance(arg, list):
        return [_encode_arg(a) for a in arg]
    if isinstance(arg, dict):
        return {str(k): _encode_arg(v) for k, v in arg.items()}
    if isinstance(arg, (str, int, float, bool)) or arg is None:
        return arg
    # Last resort — repr it so the packet isn't lost. Real code should
    # only emit JSON-friendly primitives over the network.
    return repr(arg)


def _decode_arg(arg: Any) -> Any:
    if isinstance(arg, dict) and "__b" in arg and len(arg) == 1:
        return bytes.fromhex(arg["__b"])
    if isinstance(arg, list):
        return [_decode_arg(a) for a in arg]
    if isinstance(arg, dict):
        return {k: _decode_arg(v) for k, v in arg.items()}
    return arg


def _encode_packet(d: dict) -> bytes:
    return json.dumps(d, separators=(",", ":")).encode("utf-8")


def _decode_packet(data: bytes) -> Optional[dict]:
    try:
        d = json.loads(data.decode("utf-8"))
        if not isinstance(d, dict) or d.get("v") != PROTO_VERSION:
            return None
        return d
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


# ────────────────────────────────────────────────────────────────────
# SignalBus — the UDP transport + registry
# ────────────────────────────────────────────────────────────────────

class SignalBus:
    """
    One instance per rio. Owns the UDP socket, the local signal
    registry, and the inbound/outbound subscriber sets.

    `start()` must be called once Qt is up (so we can invoke onto
    the GUI thread). It spins a small asyncio loop on a daemon thread.

    `stop()` is best-effort: we send `bye` to known peers so they drop
    our subscription quickly instead of waiting for the heartbeat
    timeout.
    """

    def __init__(
        self,
        machine_name: str,
        bind_host: str = "0.0.0.0",
        bind_port: Optional[int] = None,
        mux_ctl_path: str = "/n/ctl",
    ):
        self.machine_name = machine_name
        self.bind_host = bind_host
        self.bind_port = bind_port  # filled by start() if None
        self.mux_ctl_path = mux_ctl_path

        # ── Local signal registry ─────────────────────────────────
        # signal_name -> set of weakrefs to RemoteSignal instances
        # Multiple distinct namespace objects can share a name (e.g.
        # two QObject classes both declaring `text_spoken`). We fire
        # all of them on delivery.
        self._signals: Dict[str, "weakref.WeakSet[_SignalBase]"] = {}
        # Anonymous (just-created, not-yet-named) signals — held
        # weakly until the parse-completion hook names them.
        self._anonymous: "weakref.WeakSet[_SignalBase]" = weakref.WeakSet()

        # ── Outbound: peers WE subscribed to ──────────────────────
        # machine_name -> (host, port). We send `sub` to them and
        # accept their `emit` packets.
        self._out_peers: Dict[str, Tuple[str, int]] = {}

        # ── Inbound: peers subscribed TO us ───────────────────────
        # addr (host, port) -> (machine_name, last_seen_ts).
        # Heartbeat refreshes last_seen; expired entries are pruned.
        self._in_subs: Dict[Tuple[str, int], Tuple[str, float]] = {}

        # ── Asyncio plumbing ──────────────────────────────────────
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[_DatagramProto] = None
        # Raw socket — cached during setup so `_broadcast_emit` can
        # call sendto() directly from the emitting thread instead of
        # bouncing through `call_soon_threadsafe`. See the stress-test
        # commentary in `_async_setup`.
        self._sock: Optional[socket.socket] = None
        self._stopped = threading.Event()
        self._started = threading.Event()

        # Pluggable resolver: machine_name -> (host, signal_port) or None.
        # Default reads /n/ctl. Tests / standalone callers can override.
        self._resolver: Callable[[str], Optional[Tuple[str, int]]] = self._default_resolve

    # ── Lifecycle ───────────────────────────────────────────────────

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._thread_main, name="rio-signalbus", daemon=True,
        )
        self._thread.start()
        # Block until the socket is bound; otherwise early subscribe()
        # calls would race the loop creation.
        self._started.wait(timeout=5.0)

    def _thread_main(self):
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_setup())
            self._started.set()
            loop.run_forever()
        finally:
            try:
                loop.run_until_complete(self._async_teardown())
            except Exception:
                logger.exception("teardown failed")
            loop.close()

    async def _async_setup(self):
        # Bind UDP. If bind_port is None, let the OS pick — but rio
        # passes the conventional `9p_port + 100` so peers can find us
        # deterministically.
        loop = asyncio.get_running_loop()
        port = self.bind_port if self.bind_port is not None else 0
        self._protocol = _DatagramProto(self)
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: self._protocol,
            local_addr=(self.bind_host, port),
            allow_broadcast=True,
        )
        recv_sock = self._transport.get_extra_info("socket")
        actual = recv_sock.getsockname()
        self.bind_port = actual[1]

        # Bump SO_RCVBUF on the inbound side so kernel UDP queues can
        # absorb bursts before any packets are dropped on receive.
        try:
            recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        except OSError as e:
            logger.debug("setsockopt SO_RCVBUF failed: %s", e)

        # ── Separate outbound socket ──────────────────────────────
        # We could not reuse the asyncio-managed inbound socket for
        # direct sends from the emit thread: asyncio wraps it in a
        # `TransportSocket` which deliberately omits sendto() so
        # callers don't bypass the transport's flow control. Wrapping
        # it back via socket.socket(fileno=...) duplicates the fd and
        # produces "Bad file descriptor" on close.
        #
        # Workaround: a second, send-only UDP socket. Peers don't need
        # to see our source port (they reply to our receive port,
        # which we tell them about explicitly in `sub` / `hello`).
        # SO_REUSEADDR is fine; we don't bind it.
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        except OSError as e:
            logger.debug("setsockopt SO_SNDBUF failed: %s", e)

        logger.info(
            "SignalBus '%s' listening on %s:%d",
            self.machine_name, actual[0], actual[1],
        )

        # Heartbeat / prune task. We hold a ref so teardown can
        # cancel it cleanly — otherwise loop.close() raises a noisy
        # "Task was destroyed but it is pending" warning at exit.
        self._housekeeping_task = loop.create_task(self._housekeeping())

    async def _async_teardown(self):
        # Stop housekeeping first so it doesn't try to send on a
        # closing transport.
        task = getattr(self, "_housekeeping_task", None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        # Tell peers we're going. They'll remove us from their inbound
        # subscriber set; otherwise they'd wait one heartbeat window.
        for name, (host, port) in list(self._out_peers.items()):
            try:
                self._send_to(
                    {"v": PROTO_VERSION, "k": "bye", "from": self.machine_name},
                    host, port,
                )
            except Exception:
                pass
        # Likewise tell our inbound subscribers — symmetric goodbye so
        # they free their outbound entry.
        for addr in list(self._in_subs.keys()):
            try:
                self._send_to(
                    {"v": PROTO_VERSION, "k": "bye", "from": self.machine_name},
                    addr[0], addr[1],
                )
            except Exception:
                pass
        if self._transport:
            self._transport.close()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def stop(self):
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)

    # ── Signal registry ─────────────────────────────────────────────

    def _track_anonymous(self, sig: _SignalBase):
        self._anonymous.add(sig)

    def _register_named(self, sig: _SignalBase):
        """Make `sig` deliverable as `sig._name`. Idempotent."""
        if not sig._name:
            return
        sigset = self._signals.setdefault(sig._name, weakref.WeakSet())
        sigset.add(sig)
        sig._registered = True
        # Push a hello so already-subscribed peers learn we now know
        # this signal (mostly cosmetic — emits work either way).
        self._broadcast_hello()

    def adopt_namespace(self, namespace: dict):
        """
        Look at a user namespace dict and bind names to any
        still-anonymous RemoteSignals it contains. Idempotent — once
        a signal has a name, this call ignores it.

        We scan three levels:

          1. Top-level names in the namespace.
                 text_spoken = Signal(str)
             → named "text_spoken".

          2. One step into instance `__dict__`s of objects, plus the
             instance's class `__dict__` (its `type(val).__dict__`).
             This catches both common patterns:

                 class W:
                     def __init__(self):
                         self.text_spoken = Signal(str)   # instance attr
                 w = W()

                 class Bridge(QObject):
                     calendar_color = Signal(str)          # CLASS attr
                 b = Bridge()

             The second pattern is the natural way to write Qt-style
             code — `Signal()` declared at class level — and our
             RemoteSignal survives it (it's a plain Python object,
             so PySide6's metaclass doesn't replace it). We just need
             to FIND it by also scanning the type.

          3. The `__dict__` of any class object that itself appears
             in the namespace. So even before the user instantiates
             their class, defining

                 class Bridge(QObject):
                     calendar_color = Signal(str)

             names the signal "calendar_color".

        We don't recurse further — `obj.foo.bar.signal` stays
        anonymous on purpose. If a user needs explicit naming inside
        nested structures, they can use `make_signal("name", *types)`.
        """
        if not self._anonymous:
            return
        anon_ids = {id(s): s for s in self._anonymous}
        if not anon_ids:
            return

        def _try_name(key: str, val) -> bool:
            sid = id(val)
            if sid not in anon_ids:
                return False
            if not isinstance(val, _SignalBase) or val._name:
                return False
            val._name = key
            self._register_named(val)
            anon_ids.pop(sid, None)
            logger.debug("adopted signal name %r", key)
            return True

        def _scan_dict(d):
            # Accept both real dicts (instance __dict__) and
            # mappingproxy (class __dict__). Anything else — None,
            # a slot list, an exotic descriptor — we just skip.
            if d is None:
                return
            try:
                items = list(d.items())
            except (AttributeError, TypeError):
                return
            for attr, sub in items:
                if attr.startswith("_"):
                    continue
                _try_name(attr, sub)

        # Level 1: top-level namespace.
        for key, val in list(namespace.items()):
            if key.startswith("_"):
                continue
            _try_name(key, val)

        if not anon_ids:
            return

        import types as _types

        # Level 2 + 3: walk one step deeper. For each value in the
        # namespace, scan both its own `__dict__` (instance attrs)
        # and — crucially — its class's `__dict__` (class attrs like
        # the PySide6-style `calendar_color = Signal(str)` shown
        # above). Classes that appear directly in the namespace are
        # scanned the same way.
        seen_types: set = set()
        for key, val in list(namespace.items()):
            if key.startswith("_"):
                continue
            # Skip primitives — they don't have user-meaningful
            # __dict__s, and isinstance() on them is cheap. Modules
            # also stay out (they contain other modules' signals at
            # most, which aren't ours to name).
            if isinstance(val, (_SignalBase, _types.ModuleType)):
                continue
            if isinstance(val, (int, float, str, bool, bytes,
                                list, dict, tuple, set, frozenset)):
                continue

            # Walk the value's own __dict__ (covers `self.x = Signal()`).
            _scan_dict(getattr(val, "__dict__", None))

            # Walk the value's class __dict__ — but only once per
            # class, since many instances may share it.
            cls = val if isinstance(val, type) else type(val)
            if cls in seen_types:
                continue
            seen_types.add(cls)
            # Skip builtin / Qt / PySide6 types so we don't scan, e.g.,
            # QPushButton's class dict for every button. Heuristic: if
            # the class's module is one of the stdlib / PySide6 trees,
            # the user didn't define it, and any RemoteSignal hiding
            # there is either ours-by-accident or unrelated.
            mod = getattr(cls, "__module__", "") or ""
            if mod.startswith(("PySide6", "PyQt", "shiboken", "builtins",
                               "typing", "abc", "collections")):
                continue
            _scan_dict(getattr(cls, "__dict__", None))

            # Also walk base classes, in case the user inherits from
            # another user-defined class with the Signal.
            for base in cls.__mro__[1:]:
                if base in seen_types:
                    continue
                seen_types.add(base)
                mod = getattr(base, "__module__", "") or ""
                if mod.startswith(("PySide6", "PyQt", "shiboken", "builtins",
                                   "typing", "abc", "collections")):
                    continue
                _scan_dict(getattr(base, "__dict__", None))

            if not anon_ids:
                return

    def registered_signals(self) -> Dict[str, List[str]]:
        """Snapshot: name -> [type_tag, ...] for /scene/signals/registered."""
        out: Dict[str, List[str]] = {}
        for name, sigset in self._signals.items():
            for sig in sigset:
                out[name] = list(sig._type_tags)
                break
        return out

    # ── Subscriptions ───────────────────────────────────────────────

    def set_resolver(self, fn: Callable[[str], Optional[Tuple[str, int]]]):
        """Override machine-name → (host, signal_port) resolution."""
        self._resolver = fn

    def _default_resolve(self, machine: str) -> Optional[Tuple[str, int]]:
        """
        Read /n/ctl produced by riomux and pick out `<machine>`.

        Format (per riomux/mux.py:_format_ctl_listing):
            <name> <host>:<9p_port>

        We assume signal_port = 9p_port + DEFAULT_SIGNAL_PORT_OFFSET.
        That convention is fixed by `start_rio()` setting the bind
        port the same way.
        """
        try:
            with open(self.mux_ctl_path, "r") as f:
                text = f.read()
        except OSError as e:
            logger.warning("can't read %s: %s", self.mux_ctl_path, e)
            return None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2 or parts[0] != machine:
                continue
            addr = parts[1]
            if ":" not in addr:
                continue
            host, port_s = addr.rsplit(":", 1)
            try:
                port = int(port_s) + DEFAULT_SIGNAL_PORT_OFFSET
            except ValueError:
                continue
            return host, port
        return None

    def subscribe(self, machine: str,
                  host: Optional[str] = None,
                  port: Optional[int] = None) -> bool:
        """
        Subscribe to remote `machine`. Returns True if we found an
        address for it (or one was provided) and queued the `sub`
        packet. False if resolution failed.

        Safe to call repeatedly — re-subscribes refresh the peer's
        last-seen timer.
        """
        if machine == self.machine_name:
            logger.debug("ignoring self-subscribe %r", machine)
            return True

        if host is None or port is None:
            resolved = self._resolver(machine)
            if resolved is None:
                logger.warning("subscribe: can't resolve %r", machine)
                return False
            host, port = resolved

        self._out_peers[machine] = (host, port)
        # Send sub + hello so the peer immediately knows about us.
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._send_sub, machine, host, port)
        return True

    def unsubscribe(self, machine: str) -> bool:
        peer = self._out_peers.pop(machine, None)
        if peer is None:
            return False
        host, port = peer
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                self._send_to,
                {"v": PROTO_VERSION, "k": "unsub", "from": self.machine_name},
                host, port,
            )
        return True

    def list_subscriptions(self) -> Dict[str, Tuple[str, int]]:
        return dict(self._out_peers)

    def list_subscribers(self) -> List[Tuple[str, str, int]]:
        """(machine, host, port) tuples for everyone listening to us."""
        return [(name, addr[0], addr[1]) for addr, (name, _) in self._in_subs.items()]

    # ── Outbound emit ───────────────────────────────────────────────

    def _broadcast_emit(self, name: str, args: tuple, type_tags: tuple):
        """
        Called by RemoteSignal.emit(). Fan out to inbound subscribers.

        Sends directly on the cached UDP socket from the *calling*
        thread (typically the Qt main thread). UDP sendto() is thread-
        safe on Linux/macOS/Windows so this avoids the round-trip
        through `loop.call_soon_threadsafe`, which dropped ~half of a
        600-emit burst in the stress test because callbacks queued
        faster than the bus loop could drain them.
        """
        if not self._in_subs or self._sock is None:
            return  # nobody's listening, or bus not yet started
        packet = {
            "v": PROTO_VERSION,
            "k": "emit",
            "from": self.machine_name,
            "name": name,
            "args": [_encode_arg(a) for a in args],
        }
        data = _encode_packet(packet)
        if len(data) > MAX_PACKET:
            logger.warning("dropping oversized emit packet %d bytes", len(data))
            return
        # Snapshot subscribers — the dict may mutate from the bus
        # loop (housekeeping / packet handler).
        targets = list(self._in_subs.keys())
        sock = self._sock
        for host, port in targets:
            try:
                sock.sendto(data, (host, port))
            except OSError:
                # Peer unreachable / buffer full. Drop and continue —
                # housekeeping will eventually time them out.
                pass

    def _broadcast_hello(self):
        if not self._in_subs or self._loop is None:
            return
        packet = self._hello_packet()
        targets = list(self._in_subs.keys())
        # Hello packets are rare (signal-registry changes only) — the
        # call_soon_threadsafe overhead is fine here.
        self._loop.call_soon_threadsafe(self._send_many, packet, targets)

    def _hello_packet(self) -> dict:
        return {
            "v": PROTO_VERSION,
            "k": "hello",
            "from": self.machine_name,
            "sigs": self.registered_signals(),
        }

    # ── Send helpers (run on the bus loop only) ─────────────────────

    def _send_to(self, packet: dict, host: str, port: int):
        if self._transport is None:
            return
        data = _encode_packet(packet)
        if len(data) > MAX_PACKET:
            logger.warning("dropping oversized packet %d bytes", len(data))
            return
        try:
            self._transport.sendto(data, (host, port))
        except OSError as e:
            logger.debug("sendto %s:%d failed: %s", host, port, e)

    def _send_many(self, packet: dict, targets: List[Tuple[str, int]]):
        if self._transport is None:
            return
        data = _encode_packet(packet)
        if len(data) > MAX_PACKET:
            logger.warning("dropping oversized packet %d bytes", len(data))
            return
        for host, port in targets:
            try:
                self._transport.sendto(data, (host, port))
            except OSError:
                pass  # peer dead; housekeeping will reap

    def _send_sub(self, machine: str, host: str, port: int):
        self._send_to(
            {"v": PROTO_VERSION, "k": "sub", "from": self.machine_name},
            host, port,
        )
        # Also send our hello so peer can verify our signal vocabulary.
        self._send_to(self._hello_packet(), host, port)

    # ── Inbound packet handling ─────────────────────────────────────

    def _on_packet(self, data: bytes, addr: Tuple[str, int]):
        d = _decode_packet(data)
        if d is None:
            return
        kind = d.get("k")
        sender = d.get("from")
        if not isinstance(sender, str):
            return

        if kind == "sub":
            self._in_subs[addr] = (sender, time.monotonic())
            # Send a hello back so they discover our signals.
            self._send_to(self._hello_packet(), addr[0], addr[1])
            logger.info("subscriber added: %s@%s:%d", sender, addr[0], addr[1])

        elif kind == "unsub" or kind == "bye":
            existed = self._in_subs.pop(addr, None)
            if existed:
                logger.info("subscriber removed: %s@%s:%d", sender, addr[0], addr[1])
            # If we had subscribed to them, drop that too on `bye`.
            if kind == "bye":
                self._out_peers.pop(sender, None)

        elif kind == "hello":
            # Informational. We could mirror their signal vocabulary
            # for introspection. For now we just log.
            sigs = d.get("sigs", {})
            logger.debug("hello from %s: %s", sender, list(sigs.keys()))

        elif kind == "emit":
            # Deliver only if we subscribed to this sender — prevents
            # arbitrary hosts from injecting signals.
            if sender not in self._out_peers:
                logger.debug("dropping unsolicited emit from %r", sender)
                return
            name = d.get("name")
            args = d.get("args", [])
            if not isinstance(name, str):
                return
            decoded = [_decode_arg(a) for a in args] if isinstance(args, list) else []
            self._deliver_local(name, decoded)

        else:
            logger.debug("unknown packet kind %r from %s", kind, sender)

    def _deliver_local(self, name: str, args: list):
        sigset = self._signals.get(name)
        if not sigset:
            return
        # Iterate over a snapshot — slots may add/remove signals.
        for sig in list(sigset):
            sig._deliver_remote(args)

    # ── Housekeeping ────────────────────────────────────────────────

    async def _housekeeping(self):
        """Resend `sub` to our peers and prune dead inbound subscribers."""
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:
                return

            # Re-arm outbound subs — peers might have restarted and
            # lost our entry. Cheap (one packet per peer per 20s).
            for machine, (host, port) in list(self._out_peers.items()):
                self._send_sub(machine, host, port)

            # Prune inbound subs that haven't refreshed in 3 windows.
            now = time.monotonic()
            stale_cutoff = now - (HEARTBEAT_INTERVAL * 3)
            for addr, (name, ts) in list(self._in_subs.items()):
                if ts < stale_cutoff:
                    self._in_subs.pop(addr, None)
                    logger.info("subscriber timed out: %s@%s:%d", name, *addr)


class _DatagramProto(asyncio.DatagramProtocol):
    def __init__(self, bus: SignalBus):
        self._bus = bus

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        try:
            self._bus._on_packet(data, addr)
        except Exception:
            logger.exception("packet handler raised")

    def error_received(self, exc):
        logger.debug("UDP error: %s", exc)


# ────────────────────────────────────────────────────────────────────
# Module-level singleton wiring
# ────────────────────────────────────────────────────────────────────
#
# rio creates exactly one SignalBus per process. The user's namespace
# `subscribe()` / `unsubscribe()` and the `Signal` factory all go
# through this singleton.
#
# We don't auto-start the bus on import — that would bind a UDP port
# in any process that imports `rio.signals` (e.g. tests). The rio
# server explicitly calls `init_global_bus(...)` once during startup.

_BUS: Optional[SignalBus] = None
_BUS_LOCK = threading.Lock()


def _global_bus() -> Optional[SignalBus]:
    return _BUS


def init_global_bus(machine_name: str,
                    bind_host: str = "0.0.0.0",
                    bind_port: Optional[int] = None,
                    mux_ctl_path: str = "/n/ctl") -> SignalBus:
    """
    Idempotent. Called from rio startup. Subsequent calls return the
    existing bus; if a different machine_name is requested we log a
    warning and keep the original (you'd have to restart the process
    to change identity, which is the saner semantics anyway).
    """
    global _BUS
    with _BUS_LOCK:
        if _BUS is not None:
            if _BUS.machine_name != machine_name:
                logger.warning(
                    "init_global_bus: keeping existing identity %r (requested %r)",
                    _BUS.machine_name, machine_name,
                )
            return _BUS
        _BUS = SignalBus(
            machine_name=machine_name,
            bind_host=bind_host,
            bind_port=bind_port,
            mux_ctl_path=mux_ctl_path,
        )
        _BUS.start()
        return _BUS


def shutdown_global_bus():
    global _BUS
    with _BUS_LOCK:
        if _BUS is not None:
            _BUS.stop()
            _BUS = None


# ────────────────────────────────────────────────────────────────────
# User-facing free functions (injected into the parse namespace)
# ────────────────────────────────────────────────────────────────────

def subscribe(machine: str,
              host: Optional[str] = None,
              port: Optional[int] = None) -> bool:
    """
    Listen for signal emits from `machine`. After this call, any local
    Signal with the same name as one of <machine>'s signals will fire
    when <machine> emits.

    Returns True on success (subscription queued), False if the machine
    can't be resolved AND no explicit host/port was given.
    """
    bus = _global_bus()
    if bus is None:
        raise RuntimeError(
            "rio.signals not initialized — start rio with signals enabled."
        )
    return bus.subscribe(machine, host=host, port=port)


def unsubscribe(machine: str) -> bool:
    bus = _global_bus()
    if bus is None:
        return False
    return bus.unsubscribe(machine)


def subscriptions() -> Dict[str, Tuple[str, int]]:
    """{ machine_name: (host, port), ... }"""
    bus = _global_bus()
    return bus.list_subscriptions() if bus else {}


def subscribers() -> List[Tuple[str, str, int]]:
    """Peers currently listening to us."""
    bus = _global_bus()
    return bus.list_subscribers() if bus else []


__all__ = [
    "Signal",
    "make_signal",
    "subscribe",
    "unsubscribe",
    "subscriptions",
    "subscribers",
    "SignalBus",
    "init_global_bus",
    "shutdown_global_bus",
    "DEFAULT_SIGNAL_PORT_OFFSET",
]