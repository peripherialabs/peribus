"""
rio.signals_fs — 9P filesystem exposure for the SignalBus.

Mounts under `/scene/signals/` on the rio root:

    signals/
    ├── ctl              # write commands; cat = list subscriptions
    ├── port             # UDP port the bus is bound on
    ├── machine          # our machine name
    ├── registered       # JSON: local signals with type signatures
    ├── subscriptions    # JSON: machines we listen to
    └── subscribers      # JSON: machines listening to us

ctl commands:
    subscribe <machine> [<host>:<port>]
    unsubscribe <machine>
    list

Why expose this at all when we already have a Python API? Three reasons:
  1. Debuggability — `cat /n/<me>/scene/signals/subscriptions` from any
     shell, on any peer, just works.
  2. Cross-language clients — anything that speaks 9P can drive the bus
     without importing Python.
  3. Symmetric with the rest of rio — every other knob is a file.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from core.files import SyntheticDir, SyntheticFile, CtlFile, CtlHandler
from core.types import FidState

from . import signals as _signals

logger = logging.getLogger("rio.signals_fs")


class SignalsCtlHandler(CtlHandler):
    """Handles writes to /scene/signals/ctl."""

    def __init__(self, bus_getter):
        # Stored as a callable so the dir can be constructed before
        # the bus is up. bus_getter() -> SignalBus | None.
        self._bus_getter = bus_getter

    async def execute(self, command: str) -> Optional[str]:
        bus = self._bus_getter()
        if bus is None:
            return "error: signal bus not running\n"

        parts = command.strip().split()
        if not parts:
            return ""
        verb = parts[0].lower()

        if verb in ("subscribe", "sub"):
            if len(parts) < 2:
                return "usage: subscribe <machine> [<host>:<port>]\n"
            machine = parts[1]
            host, port = None, None
            if len(parts) >= 3 and ":" in parts[2]:
                h, p = parts[2].rsplit(":", 1)
                try:
                    host, port = h, int(p)
                except ValueError:
                    return f"error: bad addr {parts[2]!r}\n"
            ok = bus.subscribe(machine, host=host, port=port)
            return f"subscribed {machine}\n" if ok else f"error: could not resolve {machine!r}\n"

        elif verb in ("unsubscribe", "unsub"):
            if len(parts) < 2:
                return "usage: unsubscribe <machine>\n"
            ok = bus.unsubscribe(parts[1])
            return f"unsubscribed {parts[1]}\n" if ok else f"error: not subscribed to {parts[1]!r}\n"

        elif verb == "list":
            lines = []
            for m, (h, p) in bus.list_subscriptions().items():
                lines.append(f"out {m} {h}:{p}")
            for m, h, p in bus.list_subscribers():
                lines.append(f"in  {m} {h}:{p}")
            return ("\n".join(lines) + "\n") if lines else ""

        else:
            return f"error: unknown command {verb!r}\n"

    async def get_status(self) -> bytes:
        """cat /signals/ctl — show current state."""
        bus = self._bus_getter()
        if bus is None:
            return b"bus not running\n"
        lines = []
        lines.append(f"machine {bus.machine_name}")
        lines.append(f"port {bus.bind_port}")
        for m, (h, p) in bus.list_subscriptions().items():
            lines.append(f"out {m} {h}:{p}")
        for m, h, p in bus.list_subscribers():
            lines.append(f"in  {m} {h}:{p}")
        return ("\n".join(lines) + "\n").encode("utf-8")


class _ReadOnlyTextFile(SyntheticFile):
    """Tiny helper: read-only file whose content is recomputed on read."""

    def __init__(self, name: str, producer):
        super().__init__(name)
        self._producer = producer

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        try:
            data = self._producer()
            if isinstance(data, str):
                data = data.encode("utf-8")
        except Exception as e:
            data = f"error: {e}\n".encode("utf-8")
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError(f"{self.name} is read-only")


class SignalsDir(SyntheticDir):
    """
    /scene/signals/ — see module docstring for layout.

    Constructed lazily (no SignalBus required at construction time);
    files resolve the bus through `bus_getter` at each read.
    """

    def __init__(self, bus_getter):
        super().__init__("signals")
        self._bus_getter = bus_getter

        self.add(CtlFile("ctl", SignalsCtlHandler(bus_getter)))
        self.add(_ReadOnlyTextFile("port", self._read_port))
        self.add(_ReadOnlyTextFile("machine", self._read_machine))
        self.add(_ReadOnlyTextFile("registered", self._read_registered))
        self.add(_ReadOnlyTextFile("subscriptions", self._read_subscriptions))
        self.add(_ReadOnlyTextFile("subscribers", self._read_subscribers))

    # ── producers ───────────────────────────────────────────────────

    def _read_port(self) -> str:
        bus = self._bus_getter()
        return f"{bus.bind_port}\n" if bus else "0\n"

    def _read_machine(self) -> str:
        bus = self._bus_getter()
        return f"{bus.machine_name}\n" if bus else "\n"

    def _read_registered(self) -> str:
        bus = self._bus_getter()
        if bus is None:
            return "{}\n"
        return json.dumps(bus.registered_signals(), indent=2) + "\n"

    def _read_subscriptions(self) -> str:
        bus = self._bus_getter()
        if bus is None:
            return "{}\n"
        # Tuples aren't JSON; flatten to {machine: "host:port"}.
        d = {m: f"{h}:{p}" for m, (h, p) in bus.list_subscriptions().items()}
        return json.dumps(d, indent=2) + "\n"

    def _read_subscribers(self) -> str:
        bus = self._bus_getter()
        if bus is None:
            return "[]\n"
        out = [{"machine": m, "host": h, "port": p}
               for m, h, p in bus.list_subscribers()]
        return json.dumps(out, indent=2) + "\n"