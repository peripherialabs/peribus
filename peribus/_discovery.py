"""
peribus._discovery — concatenation of: discovery.py, rendezvous.py, kademlia.py, overlay.py, dht_discovery.py

This is a build artefact. The original module names live as section
banners below so `grep "^# ===="` jumps to each one.
"""

from __future__ import annotations


# ============================================================================
# discovery.py
# ----------------------------------------------------------------------------
"""
peribus.discovery — finding other peers

This module abstracts "how do peers find each other" behind a tiny
interface, so we can ship local-network mDNS today and swap in libp2p
DHT later without touching the rest of the daemon.

The interface:

    class Discovery:
        async def start(self) -> None
        async def stop(self) -> None
        # Called by the daemon when a new peer appears.
        on_peer_appeared: Callable[[PeerInfo], Awaitable[None]]
        # Called when a peer goes silent / leaves.
        on_peer_disappeared: Callable[[str], Awaitable[None]]
        # Periodically called to refresh the local advertisement.
        async def announce(self, info: LocalAnnouncement) -> None

PeerInfo carries everything the daemon needs to start talking to a peer:
nodeid, host, port, advertised vector sketch, signed timestamp.
"""


import asyncio
import json
import socket
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional


PERIBUS_SERVICE_TYPE = "_peribus._tcp.local."


@dataclass
class PeerInfo:
    """What we know about a discovered peer before we've talked to them."""
    nodeid: str
    host: str
    port: int
    pubkey: bytes = b""           # raw 32-byte ed25519 pubkey (verified later)
    sketch: List[float] = field(default_factory=list)  # vector sketch for fast filter
    last_seen: float = 0.0


@dataclass
class LocalAnnouncement:
    """What we advertise to other peers."""
    nodeid: str
    port: int
    pubkey: bytes
    sketch: List[float]


class Discovery:
    """Base class — concrete backends subclass this."""

    on_peer_appeared: Optional[Callable[[PeerInfo], Awaitable[None]]] = None
    on_peer_disappeared: Optional[Callable[[str], Awaitable[None]]] = None

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError

    async def announce(self, info: LocalAnnouncement) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# mDNS backend — works on the local network with zero infra.
#
# We use python-zeroconf (`pip install zeroconf`). The advertised TXT
# record carries the NodeID, a base64-encoded pubkey, and a base64-encoded
# packed sketch vector. mDNS limits TXT records to 255 bytes per entry, so
# we keep the sketch small (32 dims × 4 bytes = 128 bytes raw, ~172 base64).
# ---------------------------------------------------------------------------

class MdnsDiscovery(Discovery):
    """
    Local-network discovery via mDNS / DNS-SD.

    Uses python-zeroconf's AsyncZeroconf API so registration runs entirely on
    the asyncio loop (no thread bridging, no EventLoopBlocked errors when
    other things on the loop are busy).

    A lock around announce() serializes concurrent refresh calls — if two
    things try to update the advertised sketch at the same time (e.g. two
    quick share writes), the second one waits for the first instead of
    racing into NonUniqueNameException.

    update_service is preferred over unregister+register: it's atomic and
    doesn't briefly remove our entry from the network, which would cause
    every peer to flap us as "appeared/disappeared".
    """

    def __init__(self):
        self._azc = None              # AsyncZeroconf
        self._service_info = None     # currently-advertised ServiceInfo
        self._registered = False      # has register_service ever succeeded?
        self._browser = None          # AsyncServiceBrowser
        self._listener = None
        self._known: Dict[str, PeerInfo] = {}
        self._announce_lock: Optional[asyncio.Lock] = None
        # Listener callbacks fire on AsyncZeroconf's internal task loop (which
        # is the same loop we run on, but the API still says "schedule via
        # this method" — we cache the loop ref for run_coroutine_threadsafe in
        # case zeroconf ever changes that internally).
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        try:
            from zeroconf.asyncio import AsyncZeroconf, AsyncServiceBrowser
            from zeroconf import ServiceListener
        except ImportError:
            raise RuntimeError(
                "mDNS discovery needs `zeroconf` — install: pip install zeroconf"
            )

        self._loop = asyncio.get_running_loop()
        self._announce_lock = asyncio.Lock()
        self._azc = AsyncZeroconf()

        backend = self  # closure capture for the listener

        # Helper: do the async info lookup + dispatch on the loop.
        async def _lookup_and_dispatch(type_: str, name: str) -> None:
            from zeroconf.asyncio import AsyncServiceInfo
            info = AsyncServiceInfo(type_, name)
            try:
                if await info.async_request(backend._azc.zeroconf, 3000):
                    backend._handle_service(info, present=True)
            except Exception as e:
                print(f"[peribus.discovery] lookup {name} failed: {e}")

        class _Listener(ServiceListener):
            # These callbacks fire on zeroconf's own task — we just schedule
            # an async lookup to run on our loop and return immediately.
            def add_service(self, zc, type_, name):
                asyncio.run_coroutine_threadsafe(
                    _lookup_and_dispatch(type_, name), backend._loop,
                )

            def update_service(self, zc, type_, name):
                asyncio.run_coroutine_threadsafe(
                    _lookup_and_dispatch(type_, name), backend._loop,
                )

            def remove_service(self, zc, type_, name):
                # Sync — no info request needed. We can dispatch directly.
                backend._handle_remove(name)

        self._listener = _Listener()
        # AsyncServiceBrowser takes the sync Zeroconf instance under .zeroconf.
        self._browser = AsyncServiceBrowser(
            self._azc.zeroconf, PERIBUS_SERVICE_TYPE, listener=self._listener,
        )

    async def stop(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.async_cancel()
            except Exception:
                pass
            self._browser = None

        if self._azc is not None:
            if self._registered and self._service_info is not None:
                try:
                    await self._azc.async_unregister_service(self._service_info)
                except Exception:
                    pass
            try:
                await self._azc.async_close()
            except Exception:
                pass
            self._azc = None

        self._service_info = None
        self._registered = False
        self._listener = None

    async def announce(self, info: LocalAnnouncement) -> None:
        from zeroconf import ServiceInfo
        import base64

        if self._azc is None:
            return

        # Build the ServiceInfo. Re-built every call so the TXT record
        # reflects the current sketch.
        from peribus._foundation import pack_vector
        sketch_b64 = base64.b64encode(pack_vector(info.sketch))
        pub_b64 = base64.b64encode(info.pubkey)
        local_ip = _get_local_ip_bytes()
        service_name = f"{info.nodeid}.{PERIBUS_SERVICE_TYPE}"

        new_info = ServiceInfo(
            PERIBUS_SERVICE_TYPE,
            service_name,
            addresses=[local_ip],
            port=info.port,
            properties={
                b"nodeid": info.nodeid.encode("ascii"),
                b"pubkey": pub_b64,
                b"sketch": sketch_b64,
                b"v": b"peribus/0.1",
            },
            server=f"{info.nodeid}.local.",
        )

        # Serialize concurrent announces. Without this, a second call can
        # land mid-registration and trigger NonUniqueNameException.
        async with self._announce_lock:
            try:
                if not self._registered:
                    # First time: full registration.
                    await self._azc.async_register_service(new_info)
                    self._registered = True
                else:
                    # Subsequent: atomic update of the existing record.
                    # No unregister gap, so peers don't see us flap.
                    await self._azc.async_update_service(new_info)
                self._service_info = new_info
            except Exception as e:
                # Don't crash the daemon over an mDNS hiccup. Log via print
                # since we don't import logging at module load.
                print(f"[peribus.discovery] announce failed: {e}")

    # --- internal callbacks (called from zeroconf threads) ---

    def _handle_service(self, info, present: bool) -> None:
        import base64
        try:
            props = info.properties or {}
            nodeid_b = props.get(b"nodeid") or props.get("nodeid")
            if not nodeid_b:
                return
            nodeid = nodeid_b.decode("ascii") if isinstance(nodeid_b, bytes) else nodeid_b

            pub_b64 = props.get(b"pubkey") or props.get("pubkey") or b""
            sketch_b64 = props.get(b"sketch") or props.get("sketch") or b""

            pubkey = base64.b64decode(pub_b64) if pub_b64 else b""
            sketch_bytes = base64.b64decode(sketch_b64) if sketch_b64 else b""

            from peribus._foundation import unpack_vector
            sketch = unpack_vector(sketch_bytes) if sketch_bytes else []

            # Take the first IPv4 address.
            host = None
            for addr in info.addresses or []:
                if len(addr) == 4:
                    host = socket.inet_ntoa(addr)
                    break
            if host is None:
                return

            peer = PeerInfo(
                nodeid=nodeid,
                host=host,
                port=info.port,
                pubkey=pubkey,
                sketch=sketch,
                last_seen=time.time(),
            )

            # Skip ourselves.
            if self._our_nodeid and nodeid == self._our_nodeid:
                return

            self._known[nodeid] = peer

            if self._loop and self.on_peer_appeared:
                asyncio.run_coroutine_threadsafe(
                    self.on_peer_appeared(peer), self._loop
                )
        except Exception as e:
            # Don't kill zeroconf threads on bad data.
            print(f"[peribus.discovery] mdns handle error: {e}")

    def _handle_remove(self, service_name: str) -> None:
        # Service name is "<nodeid>._peribus._tcp.local." — strip the suffix.
        if not service_name.endswith("." + PERIBUS_SERVICE_TYPE):
            return
        nodeid = service_name[: -(len(PERIBUS_SERVICE_TYPE) + 1)]
        # Skip self — re-announcements briefly remove + re-add our own
        # service, and we don't want the daemon to think we've left.
        if self._our_nodeid and nodeid == self._our_nodeid:
            return
        self._known.pop(nodeid, None)
        if self._loop and self.on_peer_disappeared:
            asyncio.run_coroutine_threadsafe(
                self.on_peer_disappeared(nodeid), self._loop
            )

    # The daemon sets this so we don't echo our own announcements back.
    _our_nodeid: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_local_ip_bytes() -> bytes:
    """Best-effort outbound IP, packed as 4 bytes for ServiceInfo."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return socket.inet_aton(ip)
    except Exception:
        return socket.inet_aton("127.0.0.1")

# ============================================================================
# rendezvous.py
# ----------------------------------------------------------------------------
"""
peribus.rendezvous — global peer discovery via public bootstrap nodes

mDNS (discovery.py) finds peers on your local network. RendezvousDiscovery
finds peers across the internet by talking to one or more public
"rendezvous" nodes — small servers (see rendezvous_server.py) whose only
job is to make peers visible to each other.

The contract matches discovery.Discovery exactly, so the daemon can run
both side-by-side: mDNS for the LAN, rendezvous for the world. Peers
discovered through either path go through the same _on_peer_appeared
codepath in the daemon.

How it works:

  1. On start, we open a long-lived TCP connection to each configured
     rendezvous server. We send a `register` with our nodeid, pubkey,
     listening port, and current vector sketch.

  2. The server keeps that registration alive as long as the connection
     is. If we drop, the server forgets us (other peers will see us
     "disappear" — same UX as mDNS).

  3. We periodically send `query` to ask for peers similar to us. The
     server replies with up to N (nodeid, public_ip, port, sketch,
     pubkey) tuples, ranked by sketch cosine to ours. Each new one
     fires on_peer_appeared; ones we stop seeing fire on_peer_disappeared.

  4. When the daemon dials a discovered peer and the direct connection
     fails (likely a NAT in the way), it can ask us to coordinate a
     hole punch. We forward the request to the target peer via the
     server (which has open connections to both sides).

What we DON'T do here:

  * Relay actual peer-to-peer traffic. Posts/messages/streams go direct
    once peers know each other's address. The rendezvous server is for
    discovery only — keeping it small and stateless is the whole point.

  * Trust the server with content. Pubkeys are advertised through it
    but every post is signed; the server can't forge a peer.

  * Accept anonymous registrations. Every `register` is signed by the
    nodeid's private key. A misbehaving server can drop you, but
    can't impersonate you.
"""


import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from peribus._discovery import Discovery, LocalAnnouncement, PeerInfo
from peribus._foundation import pack_vector, unpack_vector

logger = logging.getLogger(__name__)


# Default public rendezvous nodes. Empty by default — operators of
# peribus deployments fill this in via --bootstrap or by editing it
# here for their fork. The list is intentionally small: one or two
# is plenty, and giving every node the same well-known list defeats
# the purpose of having alternatives.
DEFAULT_BOOTSTRAP: List[str] = []


# How often we re-query the server for peers. Short enough to feel
# alive, long enough that a single rendezvous handles many clients.
QUERY_INTERVAL_S = 30.0

# How long before we consider a discovered peer "gone" if we stop seeing
# it in query results. mDNS handles its own lifecycle; for rendezvous we
# do it client-side based on absence from successive query responses.
PEER_TTL_S = 180.0

# How many peers to ask for per query. The server may return fewer.
QUERY_LIMIT = 32

# Reconnect backoff bounds.
RECONNECT_MIN_S = 1.0
RECONNECT_MAX_S = 60.0


# ---------------------------------------------------------------------------
# Bootstrap address parsing
# ---------------------------------------------------------------------------

def parse_bootstrap(spec: str) -> Tuple[str, int]:
    """
    Parse a bootstrap address string.

    Accepts:
        host:port           e.g. "rdv.example.org:5670"
        tcp!host!port       9P-style, for consistency with the rest of rio
        host                with default port 5670

    Returns (host, port).
    """
    spec = spec.strip()
    if spec.startswith("tcp!"):
        parts = spec.split("!")
        if len(parts) == 3:
            return parts[1], int(parts[2])
        raise ValueError(f"bad tcp! bootstrap spec: {spec}")
    if ":" in spec:
        host, _, port = spec.rpartition(":")
        return host, int(port)
    return spec, 5670


# ---------------------------------------------------------------------------
# Connection to one rendezvous server
# ---------------------------------------------------------------------------

@dataclass
class _ServerConn:
    """One open connection to a rendezvous server, plus its read state."""
    host: str
    port: int
    reader: Optional[asyncio.StreamReader] = None
    writer: Optional[asyncio.StreamWriter] = None
    last_query_at: float = 0.0
    seen_peers: Dict[str, float] = field(default_factory=dict)  # nodeid -> last_seen
    backoff: float = RECONNECT_MIN_S


# ---------------------------------------------------------------------------
# RendezvousDiscovery
# ---------------------------------------------------------------------------

class RendezvousDiscovery(Discovery):
    """
    Internet-wide discovery via public rendezvous servers.

    Construct with a list of bootstrap addresses; we'll stay connected to
    each, register ourselves, query periodically, and dispatch
    on_peer_appeared / on_peer_disappeared like the mDNS backend does.

    Multiple bootstrap servers are queried in parallel. A peer seen on
    any server fires on_peer_appeared once; the same nodeid on a second
    server is deduped.
    """

    def __init__(self, bootstrap: List[str], identity_signer=None):
        """
        bootstrap: list of "host:port" or "tcp!host!port" strings.
        identity_signer: optional callable bytes -> bytes (Identity.sign).
            If provided, registers are signed; servers can choose to require this.
        """
        self._bootstrap = [parse_bootstrap(s) for s in bootstrap]
        self._signer = identity_signer
        self._conns: List[_ServerConn] = [
            _ServerConn(host=h, port=p) for h, p in self._bootstrap
        ]
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self._our_announcement: Optional[LocalAnnouncement] = None
        # Aggregate view across all servers: nodeid -> last time any server saw it.
        self._global_seen: Dict[str, float] = {}
        # NodeIDs we've already announced via on_peer_appeared.
        self._announced: Dict[str, PeerInfo] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Set by the daemon, used to skip ourselves.
        self._our_nodeid: Optional[str] = None

    async def start(self) -> None:
        if self._running:
            return
        if not self._conns:
            logger.info("rendezvous: no bootstrap nodes configured; nothing to do")
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        # One task per server: keep connection alive, query loop runs inside.
        for conn in self._conns:
            self._tasks.append(asyncio.create_task(self._server_loop(conn)))
        # Reaper: periodically expire peers we haven't seen recently on any server.
        self._tasks.append(asyncio.create_task(self._reaper_loop()))
        logger.info(
            f"rendezvous: started, bootstrap nodes: "
            f"{', '.join(f'{c.host}:{c.port}' for c in self._conns)}"
        )

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
        for conn in self._conns:
            await self._close_conn(conn)
        # Fire disappear for everyone we'd announced.
        for nodeid in list(self._announced.keys()):
            if self.on_peer_disappeared:
                try:
                    await self.on_peer_disappeared(nodeid)
                except Exception as e:
                    logger.debug(f"on_peer_disappeared raised: {e}")
        self._announced.clear()
        self._global_seen.clear()

    async def announce(self, info: LocalAnnouncement) -> None:
        """
        Update what we advertise. Called by the daemon when our sketch
        drifts. We re-register on every connected server, and also fire
        a fresh query so we pick up peers that registered between our
        scheduled query intervals (otherwise the first peer to connect
        sees nobody until the next 30s tick).
        """
        self._our_announcement = info
        self._our_nodeid = info.nodeid
        # Push to every live server. Failures are silent — the server_loop
        # will retry on next cycle.
        for conn in self._conns:
            if conn.writer is not None:
                try:
                    await self._send_register(conn)
                    await self._send_query(conn)
                except Exception as e:
                    logger.debug(f"rendezvous re-register {conn.host}: {e}")

    # ------------------------------------------------------------------
    # Server connection lifecycle
    # ------------------------------------------------------------------

    async def _server_loop(self, conn: _ServerConn) -> None:
        """Keep one rendezvous server connection alive; query peers periodically."""
        while self._running:
            try:
                await self._connect(conn)
                # Send initial register if we have an announcement.
                if self._our_announcement is not None:
                    await self._send_register(conn)
                conn.backoff = RECONNECT_MIN_S
                # Read loop: parse incoming messages until close.
                await self._read_loop(conn)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(f"rendezvous {conn.host}:{conn.port}: {e}")
            # Disconnected. Clean up and back off.
            await self._close_conn(conn)
            if not self._running:
                return
            await asyncio.sleep(conn.backoff)
            conn.backoff = min(conn.backoff * 2, RECONNECT_MAX_S)

    async def _connect(self, conn: _ServerConn) -> None:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(conn.host, conn.port), timeout=10.0,
        )
        conn.reader = reader
        conn.writer = writer
        logger.debug(f"rendezvous: connected to {conn.host}:{conn.port}")

    async def _close_conn(self, conn: _ServerConn) -> None:
        if conn.writer is not None:
            try:
                conn.writer.close()
                await conn.writer.wait_closed()
            except Exception:
                pass
        conn.reader = None
        conn.writer = None
        # When a server connection drops, peers we ONLY saw via that server
        # should expire. The reaper handles this via timestamp; for promptness
        # we also reset the per-conn seen list.
        conn.seen_peers.clear()

    async def _send_line(self, conn: _ServerConn, msg: dict) -> None:
        if conn.writer is None:
            return
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        conn.writer.write(line)
        await conn.writer.drain()

    async def _send_register(self, conn: _ServerConn) -> None:
        ann = self._our_announcement
        if ann is None:
            return
        msg = {
            "type": "register",
            "nodeid": ann.nodeid,
            "port": ann.port,
            "pubkey": base64.b64encode(ann.pubkey).decode("ascii"),
            "sketch": base64.b64encode(pack_vector(ann.sketch)).decode("ascii"),
            "ts": int(time.time() * 1000),
            "v": "peribus-rdv/0.1",
        }
        # Sign the canonical body so the server (and other peers, if it
        # forwards them) can verify we control the claimed nodeid.
        if self._signer is not None:
            canon = f"{ann.nodeid}|{ann.port}|{msg['ts']}".encode("utf-8")
            msg["sig"] = base64.b64encode(self._signer(canon)).decode("ascii")
        await self._send_line(conn, msg)

    async def _send_query(self, conn: _ServerConn) -> None:
        ann = self._our_announcement
        sketch_b64 = ""
        if ann is not None:
            sketch_b64 = base64.b64encode(pack_vector(ann.sketch)).decode("ascii")
        await self._send_line(conn, {
            "type": "query",
            "limit": QUERY_LIMIT,
            "sketch": sketch_b64,
        })
        conn.last_query_at = time.time()

    async def _read_loop(self, conn: _ServerConn) -> None:
        """Read messages from this server; also drives the periodic query."""
        # Kick off the first query — but only if we've sent a register.
        # Otherwise the server (correctly) replies with "must register
        # before query" and we get spurious error noise. announce() will
        # trigger a query as soon as it arrives.
        if self._our_announcement is not None:
            await self._send_query(conn)
        while self._running and conn.reader is not None:
            # If we still have no announcement, wait indefinitely for messages
            # without a periodic query timeout. Once announce() arrives it
            # will send register+query and the periodic loop kicks in.
            if self._our_announcement is None:
                timeout = None
            else:
                timeout = max(0.5, QUERY_INTERVAL_S - (time.time() - conn.last_query_at))
            try:
                if timeout is None:
                    line = await conn.reader.readline()
                else:
                    line = await asyncio.wait_for(conn.reader.readline(), timeout=timeout)
            except asyncio.TimeoutError:
                # Time for another query.
                await self._send_query(conn)
                continue
            if not line:
                # Server closed.
                return
            try:
                msg = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            await self._handle_server_msg(conn, msg)

    # ------------------------------------------------------------------
    # Server message handling
    # ------------------------------------------------------------------

    async def _handle_server_msg(self, conn: _ServerConn, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "peers":
            await self._handle_peers(conn, msg)
        elif mtype == "punch_request":
            # Server forwarded a hole-punch coordination request from another peer.
            # The daemon hooks into this via the on_punch_request callback, set
            # by the daemon at construction. If unset, we ignore.
            await self._handle_punch_request(msg)
        elif mtype == "error":
            logger.warning(
                f"rendezvous {conn.host}: error from server: {msg.get('reason')}"
            )
        elif mtype == "ack":
            pass  # register/punch acknowledgment, fine to ignore
        else:
            logger.debug(f"rendezvous: unknown message type {mtype!r}")

    async def _handle_peers(self, conn: _ServerConn, msg: dict) -> None:
        """Server's response to a query. List of peer dicts."""
        now = time.time()
        peers_seen_this_response: List[str] = []
        for p in msg.get("peers", []):
            try:
                nodeid = p["nodeid"]
                if self._our_nodeid and nodeid == self._our_nodeid:
                    continue  # never announce ourselves
                host = p["host"]
                port = int(p["port"])
                pubkey = base64.b64decode(p.get("pubkey", "") or "")
                sketch_bytes = base64.b64decode(p.get("sketch", "") or "")
                sketch = unpack_vector(sketch_bytes) if sketch_bytes else []
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"rendezvous: bad peer entry: {e}")
                continue

            conn.seen_peers[nodeid] = now
            self._global_seen[nodeid] = now
            peers_seen_this_response.append(nodeid)

            # Only fire on_peer_appeared the first time we hear about this peer
            # (across all servers). Subsequent sightings just refresh _global_seen.
            if nodeid not in self._announced:
                info = PeerInfo(
                    nodeid=nodeid,
                    host=host,
                    port=port,
                    pubkey=pubkey,
                    sketch=sketch,
                    last_seen=now,
                )
                self._announced[nodeid] = info
                if self.on_peer_appeared:
                    try:
                        await self.on_peer_appeared(info)
                    except Exception as e:
                        logger.debug(f"on_peer_appeared raised: {e}")
            else:
                # Already announced: just bump our cached last_seen and update
                # sketch in case it drifted (the daemon recomputes resonance
                # via its own announce-message handling, so we don't re-fire).
                self._announced[nodeid].last_seen = now
                if sketch:
                    self._announced[nodeid].sketch = sketch

    async def _handle_punch_request(self, msg: dict) -> None:
        """
        A peer asked the server to coordinate a hole punch with us. The server
        forwarded their public (host, port). We pass to the daemon via the
        on_punch_request callback if set; the daemon dials that address.

        Both sides do this at roughly the same moment, which is what makes
        UDP-style hole punching work. Over TCP it's flakier but often works
        when the NATs aren't symmetric.
        """
        if self.on_punch_request is None:
            return
        try:
            nodeid = msg["nodeid"]
            host = msg["host"]
            port = int(msg["port"])
        except (KeyError, ValueError) as e:
            logger.debug(f"rendezvous: bad punch_request: {e}")
            return
        try:
            await self.on_punch_request(nodeid, host, port)
        except Exception as e:
            logger.debug(f"on_punch_request raised: {e}")

    # ------------------------------------------------------------------
    # Outbound: ask the server to coordinate a hole punch with target
    # ------------------------------------------------------------------

    async def request_punch(self, target_nodeid: str) -> bool:
        """
        Ask any connected rendezvous server to coordinate a hole-punch
        with `target_nodeid`. Returns True if at least one server got the
        request; False if we have no live server connection.
        """
        sent = False
        for conn in self._conns:
            if conn.writer is None:
                continue
            try:
                await self._send_line(conn, {
                    "type": "punch",
                    "target": target_nodeid,
                })
                sent = True
            except Exception as e:
                logger.debug(f"request_punch send failed: {e}")
        return sent

    # ------------------------------------------------------------------
    # Reaper: expire peers we haven't seen in a while
    # ------------------------------------------------------------------

    async def _reaper_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(30.0)
                now = time.time()
                for nodeid in list(self._announced.keys()):
                    last = self._global_seen.get(nodeid, 0.0)
                    if now - last > PEER_TTL_S:
                        self._announced.pop(nodeid, None)
                        self._global_seen.pop(nodeid, None)
                        if self.on_peer_disappeared:
                            try:
                                await self.on_peer_disappeared(nodeid)
                            except Exception as e:
                                logger.debug(f"on_peer_disappeared raised: {e}")
        except asyncio.CancelledError:
            pass

    # Daemon sets this to receive punch coordination requests.
    on_punch_request: Optional[Callable[[str, str, int], Awaitable[None]]] = None

# ============================================================================
# kademlia.py
# ----------------------------------------------------------------------------
"""
peribus.kademlia — a pure-Python Kademlia DHT

This is the spine of server-free discovery. Once a node knows even a
single other node's address, it can iteratively find every peer in the
network, store key-value pairs distributed across the closest nodes,
and route messages without anyone in the middle.

The protocol follows the original Kademlia paper (Maymounkov & Mazières,
2002) closely. Four RPCs:

    PING(target)         — is anyone home?
    STORE(key, value)    — please remember this k:v
    FIND_NODE(target)    — give me the k closest nodes to <target> you know
    FIND_VALUE(key)      — same, but if you have <key> stored, return it instead

Distance is XOR on the 256-bit raw hash bytes. The routing table is a
bunch of "k-buckets" — one per bit of the ID space — each holding up to
k=20 nodes whose IDs share that prefix length with us. Buckets fill from
the bottom up; the more network you've seen, the deeper your buckets go.

Iterative lookup is the magic. To find target T:
  1. Pick the α=3 closest nodes to T from your routing table.
  2. Send FIND_NODE(T) to all three in parallel.
  3. Each replies with its own k-closest-to-T list.
  4. Merge those into your candidate set, dedupe, sort by distance to T.
  5. Pick the next α closest you haven't queried yet, repeat.
  6. Stop when a round produces no closer nodes than you've already seen.

After log₂(N) rounds for a network of size N, you've found the actual
k-closest. For a million nodes that's 20 round-trips. The DHT scales.

Wire format: JSON-line over UDP. Each datagram is one RPC. Transaction
IDs (txid) match responses to requests; we keep a dict of pending
futures keyed by txid. Senders sign every RPC with their Ed25519 key
so receivers can verify authenticity (no anonymous spoofers polluting
routing tables). Signatures are optional in v0.1 to keep the dep on
`cryptography` soft, but enabling them is recommended.

Storage: in-memory dict with TTL. Stored values are republished every
hour by their original publisher. Replicas expire after 24h if not
refreshed. This means stale data falls out of the network without an
explicit DELETE, which we don't have.

Bootstrap: you arrive in the network knowing at least one other peer's
(NodeID, host, port). You PING them; if they answer, you do a
FIND_NODE for your own NodeID. Their reply gives you peers near you in
the keyspace. You query each of those, fanning out, until the iterative
lookup converges. Now your routing table is seeded.

This module only handles the DHT itself. The vector-resonance overlay
(overlay.py) and the discovery wrapper (dht_discovery.py) build on top.
"""


import asyncio
import base64
import hashlib
import json
import logging
import secrets
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kademlia parameters
# ---------------------------------------------------------------------------

K = 20                       # max nodes per k-bucket
ALPHA = 3                    # concurrency for iterative lookups
ID_BITS = 256                # SHA-256 raw hash size
RPC_TIMEOUT_S = 3.0          # how long to wait for a reply
REFRESH_INTERVAL_S = 3600    # how often to republish stored values + refresh stale buckets
EXPIRE_TTL_S = 86400         # storage entry expiry
MAX_VALUE_BYTES = 4096       # bound on stored values to prevent abuse
MAX_PACKET_BYTES = 8192      # UDP datagram cap; any RPC bigger gets dropped


# ---------------------------------------------------------------------------
# NodeID handling
#
# Externally we use the base32 form (matches the rest of peribus). Internally
# we work with the 32-byte raw SHA-256 hash, because XOR distance is computed
# bitwise.
# ---------------------------------------------------------------------------

def _raw_from_nodeid(nodeid: str) -> bytes:
    """
    Convert a peribus NodeID (26-char base32 of SHA-256(pubkey), no padding)
    back to the raw 32-byte hash. Since peribus truncates to 26 chars
    (≈130 bits), we pad with zeros to recover a 32-byte ID for distance math.

    This means two nodes whose NodeIDs differ only in the truncated tail
    will look like distance-zero to the DHT. The probability of collision
    in a 130-bit space is negligible (~10^-20 for a billion nodes), so
    this is acceptable in practice.
    """
    # Restore base32 padding ('=' to multiple of 8) and decode.
    s = nodeid.upper()
    pad = (8 - len(s) % 8) % 8
    raw = base64.b32decode(s + "=" * pad)
    # Pad with zeros to 32 bytes. Truncate if longer (shouldn't happen).
    if len(raw) >= 32:
        return raw[:32]
    return raw + b"\x00" * (32 - len(raw))


def _xor_distance(a: bytes, b: bytes) -> int:
    """XOR distance on raw 32-byte IDs, returned as an integer."""
    return int.from_bytes(bytes(x ^ y for x, y in zip(a, b)), "big")


def _bucket_index(self_raw: bytes, other_raw: bytes) -> int:
    """
    Which k-bucket does `other_raw` go into for a node whose ID is `self_raw`?

    The bucket index is the position of the highest differing bit. So nodes
    very close to us (only the last bit differs) go into bucket 0; nodes
    halfway across the keyspace go into bucket 255. There are ID_BITS
    buckets total (0..255).
    """
    distance = _xor_distance(self_raw, other_raw)
    if distance == 0:
        return 0
    return distance.bit_length() - 1


# ---------------------------------------------------------------------------
# Contact: a known peer, identified by NodeID and reachable at host:port
# ---------------------------------------------------------------------------

@dataclass
class Contact:
    """A peer in the routing table or a query result."""
    nodeid: str          # base32 NodeID (peribus form)
    host: str
    port: int
    last_seen: float = 0.0

    def __post_init__(self):
        # Cache the raw bytes for distance math; computed once per contact.
        self._raw = _raw_from_nodeid(self.nodeid)

    @property
    def raw(self) -> bytes:
        return self._raw

    def to_json(self) -> dict:
        return {"id": self.nodeid, "host": self.host, "port": self.port}

    @classmethod
    def from_json(cls, d: dict) -> "Contact":
        return cls(nodeid=d["id"], host=d["host"], port=int(d["port"]))


# ---------------------------------------------------------------------------
# Routing table — array of k-buckets
# ---------------------------------------------------------------------------

class KBucket:
    """
    One bucket: up to K contacts, ordered by recency (most-recently-seen last).

    On insert:
      - If the contact is already known, move it to the tail (recency bump).
      - If the bucket has space, append.
      - If full, the rule is: ping the head; if it answers, drop the new
        contact (incumbent wins); if it doesn't, evict the head and append
        the new one. Implemented async — the bucket itself doesn't ping;
        it returns the head as a "needs_check" hint and the caller decides.
    """

    __slots__ = ("contacts", "last_updated")

    def __init__(self):
        self.contacts: List[Contact] = []
        self.last_updated: float = time.time()

    def has(self, nodeid: str) -> bool:
        return any(c.nodeid == nodeid for c in self.contacts)

    def get(self, nodeid: str) -> Optional[Contact]:
        for c in self.contacts:
            if c.nodeid == nodeid:
                return c
        return None

    def touch(self, contact: Contact) -> Tuple[bool, Optional[Contact]]:
        """
        Insert or refresh a contact. Returns (inserted, eviction_candidate).
        - inserted=True, candidate=None: contact was added (or refreshed).
        - inserted=False, candidate=<head>: bucket is full; caller must
          ping the head and decide whether to evict.
        """
        self.last_updated = time.time()
        # Already known: move to tail (most-recently-seen).
        for i, c in enumerate(self.contacts):
            if c.nodeid == contact.nodeid:
                c.last_seen = time.time()
                # Move to tail.
                self.contacts.pop(i)
                self.contacts.append(c)
                return True, None
        # Has space.
        if len(self.contacts) < K:
            contact.last_seen = time.time()
            self.contacts.append(contact)
            return True, None
        # Full — caller must check the head.
        return False, self.contacts[0]

    def replace_head(self, new_contact: Contact) -> None:
        """Called by the caller when the head failed to respond to a ping."""
        if self.contacts:
            self.contacts.pop(0)
        new_contact.last_seen = time.time()
        self.contacts.append(new_contact)

    def remove(self, nodeid: str) -> bool:
        for i, c in enumerate(self.contacts):
            if c.nodeid == nodeid:
                self.contacts.pop(i)
                return True
        return False


class RoutingTable:
    """ID_BITS buckets of K contacts each. Owns the routing state for one node."""

    def __init__(self, self_nodeid: str):
        self.self_nodeid = self_nodeid
        self.self_raw = _raw_from_nodeid(self_nodeid)
        self.buckets: List[KBucket] = [KBucket() for _ in range(ID_BITS)]

    def touch(self, contact: Contact) -> Tuple[bool, Optional[Contact]]:
        """Add or refresh a contact. Caller handles the eviction-candidate case."""
        if contact.nodeid == self.self_nodeid:
            return True, None  # never store ourselves
        idx = _bucket_index(self.self_raw, contact.raw)
        return self.buckets[idx].touch(contact)

    def remove(self, nodeid: str) -> None:
        if nodeid == self.self_nodeid:
            return
        idx = _bucket_index(self.self_raw, _raw_from_nodeid(nodeid))
        self.buckets[idx].remove(nodeid)

    def closest(self, target_raw: bytes, count: int = K) -> List[Contact]:
        """
        Return up to `count` contacts closest to `target_raw`, sorted nearest
        first. Used to seed iterative lookups and to answer FIND_NODE.
        """
        all_contacts: List[Tuple[int, Contact]] = []
        for bucket in self.buckets:
            for c in bucket.contacts:
                d = _xor_distance(target_raw, c.raw)
                all_contacts.append((d, c))
        all_contacts.sort(key=lambda t: t[0])
        return [c for _, c in all_contacts[:count]]

    def stale_buckets(self, threshold_s: float = REFRESH_INTERVAL_S) -> List[int]:
        """Bucket indices that haven't been updated in a while — needs refresh."""
        now = time.time()
        return [
            i for i, b in enumerate(self.buckets)
            if b.contacts and now - b.last_updated > threshold_s
        ]

    def all_contacts(self) -> List[Contact]:
        """Snapshot of every known peer. Used for debugging and for the overlay."""
        out = []
        for b in self.buckets:
            out.extend(b.contacts)
        return out

    def size(self) -> int:
        return sum(len(b.contacts) for b in self.buckets)


# ---------------------------------------------------------------------------
# Wire protocol — UDP datagrams of JSON
# ---------------------------------------------------------------------------

# RPC types
RPC_PING        = "ping"
RPC_PONG        = "pong"
RPC_STORE       = "store"
RPC_STORE_OK    = "store_ok"
RPC_FIND_NODE   = "find_node"
RPC_FOUND_NODES = "found_nodes"
RPC_FIND_VALUE  = "find_value"
RPC_FOUND_VALUE = "found_value"
RPC_ERROR       = "error"


def _encode(msg: dict) -> bytes:
    """JSON-encode an RPC. We use compact separators and append a newline so
    that anyone packet-sniffing can read the wire by eye."""
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")


def _decode(data: bytes) -> Optional[dict]:
    try:
        return json.loads(data.decode("utf-8").rstrip("\n"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DHT Node — owns the socket, routing table, storage, and pending RPCs
# ---------------------------------------------------------------------------

class _Storage:
    """Tiny in-memory key-value store with TTL.

    Keys are arbitrary strings (typically hex of a hash). Values are bytes
    capped at MAX_VALUE_BYTES. Each entry remembers when it was last
    refreshed; the periodic janitor task drops anything past EXPIRE_TTL_S."""

    def __init__(self):
        self._data: Dict[str, Tuple[bytes, float]] = {}  # key -> (value, expires_at)

    def put(self, key: str, value: bytes, ttl: float = EXPIRE_TTL_S) -> None:
        if len(value) > MAX_VALUE_BYTES:
            raise ValueError(f"value too large ({len(value)} > {MAX_VALUE_BYTES})")
        self._data[key] = (value, time.time() + ttl)

    def get(self, key: str) -> Optional[bytes]:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._data[key]
            return None
        return value

    def janitor(self) -> int:
        """Drop expired entries. Returns count dropped."""
        now = time.time()
        dead = [k for k, (_, exp) in self._data.items() if now > exp]
        for k in dead:
            del self._data[k]
        return len(dead)

    def size(self) -> int:
        return len(self._data)


class KademliaNode:
    """
    One DHT node. Owns a UDP socket, a routing table, a key-value store,
    and the pending-RPC table.

    Lifecycle:
        node = KademliaNode(nodeid, sign=identity.sign, verify=verify_signature)
        await node.start(host="0.0.0.0", port=5680)
        await node.bootstrap([Contact(...)])  # optional, for joining a network
        # ... use node.find_value(), node.store(), etc ...
        await node.stop()

    The constructor takes optional sign/verify callbacks. If both are
    provided, every outgoing RPC is signed and every incoming one is
    checked. We pass these in rather than depend on peribus.identity
    so this module is reusable.
    """

    def __init__(
        self,
        nodeid: str,
        sign: Optional[Callable[[bytes], bytes]] = None,
        verify: Optional[Callable[[bytes, bytes, bytes], bool]] = None,
        pubkey_provider: Optional[Callable[[], bytes]] = None,
        peer_pubkey_lookup: Optional[Callable[[str], Optional[bytes]]] = None,
    ):
        self.nodeid = nodeid
        self.routing = RoutingTable(nodeid)
        self.storage = _Storage()
        self._sign = sign
        self._verify = verify
        self._pubkey_provider = pubkey_provider
        # Optional: lookup function for verifying incoming signatures by
        # peer NodeID. If absent we skip signature verification (since we
        # can't recover the pubkey from the NodeID alone).
        self._peer_pubkey_lookup = peer_pubkey_lookup

        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[_DhtProtocol] = None
        self._pending: Dict[str, asyncio.Future] = {}  # txid -> future
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self._host = "0.0.0.0"
        self._port = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, host: str = "0.0.0.0", port: int = 0) -> int:
        """Start the UDP listener. Returns the port we ended up on."""
        if self._running:
            return self._port
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _DhtProtocol(self), local_addr=(host, port),
        )
        self._transport = transport
        self._protocol = protocol
        sock = transport.get_extra_info("socket")
        self._host, self._port = sock.getsockname()[:2]
        self._running = True
        # Background tasks: republish + bucket refresh + storage janitor.
        self._tasks.append(asyncio.create_task(self._maintenance_loop()))
        logger.info(f"kademlia: listening on {self._host}:{self._port} (id={self.nodeid})")
        return self._port

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
        # Cancel any pending RPCs.
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    @property
    def port(self) -> int:
        return self._port

    # ------------------------------------------------------------------
    # Public API: bootstrap, store, find
    # ------------------------------------------------------------------

    async def bootstrap(self, seeds: List[Contact]) -> int:
        """
        Join an existing network using one or more seed contacts.
        Returns the number of seeds that responded.

        Steps:
          1. PING each seed; live ones go into the routing table.
          2. Iterative FIND_NODE for our own ID — this fills our neighborhood.
          3. Refresh every bucket at least once, so we know nodes far away too.
        """
        if not seeds:
            return 0

        live: List[Contact] = []
        for seed in seeds:
            if seed.nodeid == self.nodeid:
                continue
            if await self._rpc_ping(seed):
                self.routing.touch(seed)
                live.append(seed)

        if not live:
            logger.warning("kademlia: bootstrap failed — no seeds responded")
            return 0

        # Discover neighborhood by looking up our own ID.
        await self.iterative_find_node(_raw_from_nodeid(self.nodeid))

        # Refresh buckets that didn't get populated by the self-lookup.
        for bucket_idx in range(ID_BITS):
            if not self.routing.buckets[bucket_idx].contacts:
                # Pick a random ID in this bucket's range and look it up.
                random_id = self._random_id_in_bucket(bucket_idx)
                # Skip empty regions of the keyspace (most of them, for small networks).
                # We only refresh buckets that already have something OR that are near us.
                if bucket_idx < 8:
                    await self.iterative_find_node(random_id)

        logger.info(
            f"kademlia: bootstrap complete — {len(live)}/{len(seeds)} seeds answered, "
            f"routing table has {self.routing.size()} contacts"
        )
        return len(live)

    def _random_id_in_bucket(self, bucket_idx: int) -> bytes:
        """
        Generate a random 32-byte ID that would fall into bucket `bucket_idx`
        from our perspective. Useful for refreshing stale buckets.
        """
        # Bucket idx i covers IDs whose XOR distance from us has its highest
        # bit at position i. So we flip bit (255-i) of our own ID and randomize
        # the bits below it.
        target = bytearray(self.routing.self_raw)
        bit_pos = bucket_idx  # 0 = least significant
        # Flip the bit at bit_pos.
        byte_pos = 31 - bit_pos // 8
        bit_in_byte = bit_pos % 8
        target[byte_pos] ^= 1 << bit_in_byte
        # Randomize bits below.
        rand = secrets.token_bytes(32)
        # Lower bits of `target` (below bit_pos) get replaced with random.
        for b in range(bit_pos):
            bp = 31 - b // 8
            bib = b % 8
            target[bp] = (target[bp] & ~(1 << bib)) | (rand[bp] & (1 << bib))
        return bytes(target)

    async def iterative_find_node(
        self, target_raw: bytes,
    ) -> List[Contact]:
        """
        The core Kademlia algorithm. Find the K nodes closest to `target_raw`
        in the network.

        Maintains a "shortlist" of candidates. At each step, query the α
        closest-not-yet-queried. Any closer candidates returned go into the
        shortlist. Stop when no new closer candidates appear in a round.
        """
        return await self._iterative_lookup(target_raw, find_value=False)

    async def iterative_find_value(
        self, key: str,
    ) -> Optional[bytes]:
        """
        Like iterative_find_node, but if any peer has the value stored,
        we get it back (and we cache-store it on the node closest to the
        key that didn't have it, per the Kademlia paper).
        """
        target_raw = self._key_to_raw(key)
        result = await self._iterative_lookup(target_raw, find_value=True, value_key=key)
        if isinstance(result, bytes):
            return result
        return None  # only contacts came back

    async def store(self, key: str, value: bytes) -> int:
        """
        Find the K nodes closest to `key` and ask them to STORE.
        Returns count that ack'd.

        We also store locally — convenient for the publisher to be able
        to look up its own keys without round-tripping.
        """
        target_raw = self._key_to_raw(key)
        try:
            self.storage.put(key, value)
        except ValueError as e:
            logger.warning(f"kademlia: local store rejected: {e}")
            return 0

        closest = await self.iterative_find_node(target_raw)
        # Send STORE to each in parallel.
        async def _one(c: Contact) -> bool:
            return await self._rpc_store(c, key, value)
        results = await asyncio.gather(*(_one(c) for c in closest), return_exceptions=True)
        return sum(1 for r in results if r is True)

    @staticmethod
    def _key_to_raw(key: str) -> bytes:
        """
        Map an arbitrary key string to a 32-byte point in ID-space.
        SHA-256(key). Same hash family as NodeIDs so distances are comparable.
        """
        return hashlib.sha256(key.encode("utf-8")).digest()

    # ------------------------------------------------------------------
    # Iterative lookup (find_node and find_value share most of this)
    # ------------------------------------------------------------------

    async def _iterative_lookup(
        self,
        target_raw: bytes,
        *,
        find_value: bool,
        value_key: Optional[str] = None,
    ):
        # Shortlist: dict nodeid -> (Contact, distance, queried).
        seed = self.routing.closest(target_raw, K)
        if not seed:
            return None if find_value else []

        # Local lookup short-circuit for find_value.
        if find_value and value_key is not None:
            local = self.storage.get(value_key)
            if local is not None:
                return local

        # State for the lookup.
        shortlist: Dict[str, Contact] = {c.nodeid: c for c in seed}
        queried: Set[str] = set()
        # The "closest seen so far" — when a round produces no closer, we stop.
        best_dist = min(_xor_distance(target_raw, c.raw) for c in seed)

        while True:
            # Pick α not-yet-queried, sorted by distance.
            candidates = [
                c for c in shortlist.values() if c.nodeid not in queried
            ]
            candidates.sort(key=lambda c: _xor_distance(target_raw, c.raw))
            batch = candidates[:ALPHA]
            if not batch:
                break

            for c in batch:
                queried.add(c.nodeid)

            # Query in parallel.
            async def _query(c: Contact):
                if find_value:
                    return c, await self._rpc_find_value(c, value_key)  # type: ignore[arg-type]
                return c, await self._rpc_find_node(c, target_raw)

            results = await asyncio.gather(*(_query(c) for c in batch), return_exceptions=True)

            improved = False
            for r in results:
                if isinstance(r, BaseException):
                    continue
                contact, response = r
                if response is None:
                    # Timed out / no response — drop from routing table.
                    self.routing.remove(contact.nodeid)
                    continue

                # If find_value and the peer had it, we're done.
                if find_value and isinstance(response, bytes):
                    return response

                # Otherwise it's a list of contacts.
                for c in response:
                    if c.nodeid == self.nodeid:
                        continue
                    # Add to shortlist if new.
                    if c.nodeid not in shortlist:
                        shortlist[c.nodeid] = c
                    # Tickle our routing table — these are useful peers we now know.
                    self.routing.touch(c)
                    d = _xor_distance(target_raw, c.raw)
                    if d < best_dist:
                        best_dist = d
                        improved = True

            if not improved:
                # Termination: this round produced nothing closer.
                # Per the paper, we still query the K closest from shortlist
                # we haven't queried yet, to make sure we have THE closest K.
                final_batch = [
                    c for c in sorted(
                        shortlist.values(),
                        key=lambda c: _xor_distance(target_raw, c.raw),
                    )[:K]
                    if c.nodeid not in queried
                ]
                if not final_batch:
                    break
                for c in final_batch:
                    queried.add(c.nodeid)
                final_results = await asyncio.gather(
                    *(_query(c) for c in final_batch), return_exceptions=True,
                )
                for r in final_results:
                    if isinstance(r, BaseException):
                        continue
                    contact, response = r
                    if response is None:
                        self.routing.remove(contact.nodeid)
                        continue
                    if find_value and isinstance(response, bytes):
                        return response
                    for c in response:
                        if c.nodeid == self.nodeid:
                            continue
                        if c.nodeid not in shortlist:
                            shortlist[c.nodeid] = c
                        self.routing.touch(c)
                break

        if find_value:
            return None  # nobody had it
        return sorted(
            shortlist.values(),
            key=lambda c: _xor_distance(target_raw, c.raw),
        )[:K]

    # ------------------------------------------------------------------
    # RPC senders
    # ------------------------------------------------------------------

    async def _rpc_ping(self, contact: Contact) -> bool:
        resp = await self._send_and_wait(contact, {"type": RPC_PING})
        return resp is not None and resp.get("type") == RPC_PONG

    async def _rpc_store(self, contact: Contact, key: str, value: bytes) -> bool:
        resp = await self._send_and_wait(contact, {
            "type": RPC_STORE,
            "key": key,
            "value": base64.b64encode(value).decode("ascii"),
        })
        return resp is not None and resp.get("type") == RPC_STORE_OK

    async def _rpc_find_node(self, contact: Contact, target_raw: bytes) -> Optional[List[Contact]]:
        resp = await self._send_and_wait(contact, {
            "type": RPC_FIND_NODE,
            "target": base64.b64encode(target_raw).decode("ascii"),
        })
        if resp is None or resp.get("type") != RPC_FOUND_NODES:
            return None
        return [Contact.from_json(c) for c in resp.get("nodes", [])]

    async def _rpc_find_value(
        self, contact: Contact, key: str,
    ):
        """Returns either bytes (value found) or List[Contact] (closer nodes) or None."""
        resp = await self._send_and_wait(contact, {
            "type": RPC_FIND_VALUE,
            "key": key,
        })
        if resp is None:
            return None
        if resp.get("type") == RPC_FOUND_VALUE:
            try:
                return base64.b64decode(resp["value"])
            except Exception:
                return None
        if resp.get("type") == RPC_FOUND_NODES:
            return [Contact.from_json(c) for c in resp.get("nodes", [])]
        return None

    async def _send_and_wait(
        self, contact: Contact, payload: dict, timeout: float = RPC_TIMEOUT_S,
    ) -> Optional[dict]:
        """Send an RPC and wait for the reply. Returns None on timeout."""
        if self._transport is None:
            return None
        txid = secrets.token_hex(8)
        payload["txid"] = txid
        payload["from"] = self.nodeid
        # Sign the canonical body. Order matters: add `pk` first so it's part
        # of the signed bytes, then compute canonical, then add `sig`. The
        # verifier strips `sig` and recomputes; this way both sides hash the
        # exact same bytes.
        if self._sign is not None and self._pubkey_provider is not None:
            payload["pk"] = base64.b64encode(self._pubkey_provider()).decode("ascii")
            canon = _canonical(payload)
            payload["sig"] = base64.b64encode(self._sign(canon)).decode("ascii")

        data = _encode(payload)
        if len(data) > MAX_PACKET_BYTES:
            logger.warning(f"kademlia: outbound RPC too large ({len(data)} bytes), dropping")
            return None

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending[txid] = fut
        try:
            self._transport.sendto(data, (contact.host, contact.port))
            try:
                return await asyncio.wait_for(fut, timeout=timeout)
            except asyncio.TimeoutError:
                return None
        finally:
            self._pending.pop(txid, None)
    # ------------------------------------------------------------------
    # Inbound RPC handling — called by _DhtProtocol on each datagram
    # ------------------------------------------------------------------

    def _on_datagram(self, data: bytes, addr: Tuple[str, int]) -> None:
        if len(data) > MAX_PACKET_BYTES:
            return  # ignore oversized
        msg = _decode(data)
        if not isinstance(msg, dict):
            return
        mtype = msg.get("type")
        if not isinstance(mtype, str):
            return
        # Verify signature if we can.
        if self._verify is not None and "sig" in msg and "pk" in msg:
            try:
                pk = base64.b64decode(msg["pk"])
                sig = base64.b64decode(msg["sig"])
                # Build the body without sig for verification.
                without_sig = {k: v for k, v in msg.items() if k != "sig"}
                canon = _canonical(without_sig)
                if not self._verify(pk, canon, sig):
                    return  # silently drop forged messages
                # Bind: pubkey must match the claimed sender NodeID.
                from_id = msg.get("from")
                if isinstance(from_id, str):
                    # Use the same hash function as identity.py
                    expected = base64.b32encode(
                        hashlib.sha256(pk).digest()
                    ).decode("ascii").lower().rstrip("=")[:26]
                    if expected != from_id:
                        return  # spoofed sender
            except Exception:
                return

        # Is this a response to one of our outgoing RPCs?
        txid = msg.get("txid")
        if isinstance(txid, str) and mtype in (
            RPC_PONG, RPC_STORE_OK, RPC_FOUND_NODES, RPC_FOUND_VALUE, RPC_ERROR,
        ):
            fut = self._pending.get(txid)
            if fut is not None and not fut.done():
                fut.set_result(msg)
            return

        # Otherwise it's a request — handle and respond.
        sender_id = msg.get("from")
        if isinstance(sender_id, str) and sender_id != self.nodeid:
            sender_contact = Contact(nodeid=sender_id, host=addr[0], port=addr[1])
            inserted, evict_candidate = self.routing.touch(sender_contact)
            if not inserted and evict_candidate is not None:
                # Bucket full; ping the head async to decide.
                asyncio.create_task(self._challenge_head(evict_candidate, sender_contact))

        # Dispatch.
        if mtype == RPC_PING:
            self._reply(addr, {"type": RPC_PONG, "txid": txid})
        elif mtype == RPC_STORE:
            self._handle_store(msg, addr, txid)
        elif mtype == RPC_FIND_NODE:
            self._handle_find_node(msg, addr, txid)
        elif mtype == RPC_FIND_VALUE:
            self._handle_find_value(msg, addr, txid)
        # Unknown message types are ignored.

    async def _challenge_head(self, head: Contact, candidate: Contact) -> None:
        """When a bucket is full and a new candidate arrives, ping the head.
        If it answers, the incumbent stays. If not, evict and admit the new one."""
        if await self._rpc_ping(head):
            head.last_seen = time.time()  # incumbent wins
        else:
            idx = _bucket_index(self.routing.self_raw, candidate.raw)
            self.routing.buckets[idx].replace_head(candidate)

    def _reply(self, addr: Tuple[str, int], payload: dict) -> None:
        if self._transport is None:
            return
        payload["from"] = self.nodeid
        if self._sign is not None and self._pubkey_provider is not None:
            payload["pk"] = base64.b64encode(self._pubkey_provider()).decode("ascii")
            canon = _canonical(payload)
            payload["sig"] = base64.b64encode(self._sign(canon)).decode("ascii")
        data = _encode(payload)
        if len(data) > MAX_PACKET_BYTES:
            return
        try:
            self._transport.sendto(data, addr)
        except Exception:
            pass

    def _handle_store(self, msg: dict, addr: Tuple[str, int], txid: Optional[str]) -> None:
        try:
            key = msg["key"]
            value = base64.b64decode(msg["value"])
        except (KeyError, ValueError, TypeError):
            return
        try:
            self.storage.put(key, value)
            self._reply(addr, {"type": RPC_STORE_OK, "txid": txid, "key": key})
        except ValueError as e:
            self._reply(addr, {"type": RPC_ERROR, "txid": txid, "reason": str(e)})

    def _handle_find_node(self, msg: dict, addr: Tuple[str, int], txid: Optional[str]) -> None:
        try:
            target_raw = base64.b64decode(msg["target"])
        except (KeyError, ValueError, TypeError):
            return
        nodes = self.routing.closest(target_raw, K)
        self._reply(addr, {
            "type": RPC_FOUND_NODES,
            "txid": txid,
            "nodes": [c.to_json() for c in nodes],
        })

    def _handle_find_value(self, msg: dict, addr: Tuple[str, int], txid: Optional[str]) -> None:
        key = msg.get("key")
        if not isinstance(key, str):
            return
        value = self.storage.get(key)
        if value is not None:
            self._reply(addr, {
                "type": RPC_FOUND_VALUE,
                "txid": txid,
                "key": key,
                "value": base64.b64encode(value).decode("ascii"),
            })
        else:
            target_raw = self._key_to_raw(key)
            nodes = self.routing.closest(target_raw, K)
            self._reply(addr, {
                "type": RPC_FOUND_NODES,
                "txid": txid,
                "nodes": [c.to_json() for c in nodes],
            })

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    async def _maintenance_loop(self) -> None:
        try:
            while self._running:
                # Run every 5 minutes — frequent enough that bucket refresh is
                # responsive, infrequent enough not to waste bandwidth.
                await asyncio.sleep(300.0)
                if not self._running:
                    break
                # Drop expired storage.
                dropped = self.storage.janitor()
                if dropped:
                    logger.debug(f"kademlia: storage janitor dropped {dropped} expired keys")
                # Refresh stale buckets.
                stale = self.routing.stale_buckets()
                for idx in stale[:5]:  # cap per cycle
                    rid = self._random_id_in_bucket(idx)
                    try:
                        await self.iterative_find_node(rid)
                    except Exception as e:
                        logger.debug(f"kademlia: bucket {idx} refresh failed: {e}")
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "nodeid": self.nodeid,
            "host": self._host,
            "port": self._port,
            "routing_size": self.routing.size(),
            "storage_size": self.storage.size(),
            "pending_rpcs": len(self._pending),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical(msg: dict) -> bytes:
    """
    Stable, sorted JSON encoding for signing. Key order is sorted so two
    encoders produce the same bytes for the same dict. The signature
    field itself MUST be excluded by the caller before passing in.
    """
    return json.dumps(msg, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# asyncio DatagramProtocol — bridges the loop and the KademliaNode
# ---------------------------------------------------------------------------

class _DhtProtocol(asyncio.DatagramProtocol):
    def __init__(self, node: KademliaNode):
        self._node = node

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        try:
            self._node._on_datagram(data, addr)
        except Exception as e:
            logger.warning(f"kademlia: error handling datagram from {addr}: {e}")

    def error_received(self, exc: Exception) -> None:
        # ICMP unreachable etc. We don't care — failed peers expire from
        # the routing table on the next RPC timeout.
        logger.debug(f"kademlia: udp error: {exc}")

# ============================================================================
# overlay.py
# ----------------------------------------------------------------------------
"""
peribus.overlay — vector-resonance overlay on top of Kademlia

Kademlia gives us "find a peer by NodeID" and a distributed key-value
store. What it doesn't give us is what peribus actually needs: "find
peers whose vectors are close to mine."

That problem — semantic similarity routing in a P2P network — has a
classical solution called Vicinity (or T-Man, Cyclon — there's a small
family of these). The idea is simple:

  1. Each node maintains a small, ranked list of "best neighbors" —
     peers it knows about whose sketches are most similar to its own.
  2. Periodically, the node picks a random known peer and asks for
     *their* best neighbors. Merges the response into its candidate
     pool, re-ranks, keeps the top N.
  3. Over O(log N) gossip rounds, each node converges to its true
     nearest neighbors in vector space, even if it started with random
     contacts.

The "random known peer" comes from the Kademlia routing table — that's
our random sample of the network, courtesy of the DHT. The "best
neighbors" view is what gets surfaced as "interesting peers" to the
peribus daemon, replacing what the rendezvous server used to do
centrally.

We also publish our current sketch into the DHT under a stable key
(our NodeID) so any peer doing a vicinity merge can fetch our vector
without already having heard it from us.

Why not just do approximate-nearest-neighbor over DHT keys? Because
ANN-on-DHT is a research problem and Vicinity is two pages of code
that has been known to work since 2007. Pragmatism wins.

Outputs surfaced through this module:
  * `top_resonant(n)` — current best-N peers by vector similarity to us
  * `on_peer_added` callback — fires when a new peer enters our top set
  * `on_peer_removed` callback — fires when a peer falls out

These are the moral equivalent of "found a peer on the rendezvous
server" — the daemon wires its `_on_peer_appeared` to `on_peer_added`
and gets the same UX without anyone in the middle.
"""


import asyncio
import base64
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from peribus._discovery import KademliaNode, Contact

from peribus._foundation import cosine, pack_vector, unpack_vector

logger = logging.getLogger(__name__)


# How many peers we keep in our resonance view. Larger = more accurate
# but more bandwidth on each gossip exchange. 20 is a sweet spot.
VIEW_SIZE = 20

# Gossip period. Every cycle: pick a random peer, swap views.
GOSSIP_INTERVAL_S = 15.0

# Sketch publish period. Refresh our sketch in the DHT this often so
# our entry doesn't expire and so peers see drift.
PUBLISH_INTERVAL_S = 600.0

# DHT key prefix for sketch records.
SKETCH_KEY_PREFIX = "peribus/sketch/v1/"


# ---------------------------------------------------------------------------
# A single entry in the resonance view
# ---------------------------------------------------------------------------

@dataclass
class ViewEntry:
    """One peer in our top-N resonance view."""
    nodeid: str
    host: str
    port: int           # peribus wire port (NOT the kademlia UDP port)
    sketch: List[float]
    pubkey: bytes        # raw 32-byte pubkey, for the daemon's verify path
    resonance: float     # cosine to our current sketch at last update
    last_updated: float  # unix seconds

    def to_json(self) -> dict:
        return {
            "id": self.nodeid,
            "host": self.host,
            "port": self.port,
            "sketch": base64.b64encode(pack_vector(self.sketch)).decode("ascii"),
            "pubkey": base64.b64encode(self.pubkey).decode("ascii") if self.pubkey else "",
            "ts": int(self.last_updated),
        }

    @classmethod
    def from_json(cls, d: dict, our_sketch: List[float]) -> Optional["ViewEntry"]:
        try:
            sketch_b64 = d.get("sketch", "") or ""
            sketch = unpack_vector(base64.b64decode(sketch_b64)) if sketch_b64 else []
            pubkey_b64 = d.get("pubkey", "") or ""
            pubkey = base64.b64decode(pubkey_b64) if pubkey_b64 else b""
            return cls(
                nodeid=d["id"],
                host=d["host"],
                port=int(d["port"]),
                sketch=sketch,
                pubkey=pubkey,
                resonance=cosine(our_sketch, sketch) if sketch else 0.0,
                last_updated=float(d.get("ts", time.time())),
            )
        except (KeyError, ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# What we publish about ourselves
# ---------------------------------------------------------------------------

@dataclass
class SelfRecord:
    """The thing we publish into the DHT under our NodeID-derived key."""
    nodeid: str
    host: str           # public IP we want peers to dial us at
    wire_port: int      # peribus tcp wire port (not kademlia udp)
    pubkey: bytes
    sketch: List[float]
    ts: int             # unix seconds

    def to_bytes(self) -> bytes:
        return json.dumps({
            "id": self.nodeid,
            "host": self.host,
            "port": self.wire_port,
            "pubkey": base64.b64encode(self.pubkey).decode("ascii"),
            "sketch": base64.b64encode(pack_vector(self.sketch)).decode("ascii"),
            "ts": self.ts,
        }, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> Optional["SelfRecord"]:
        try:
            d = json.loads(data.decode("utf-8"))
            return cls(
                nodeid=d["id"],
                host=d["host"],
                wire_port=int(d["port"]),
                pubkey=base64.b64decode(d.get("pubkey", "") or ""),
                sketch=unpack_vector(base64.b64decode(d.get("sketch", "") or "")),
                ts=int(d.get("ts", 0)),
            )
        except Exception:
            return None


def sketch_key_for(nodeid: str) -> str:
    """The stable DHT key under which a node publishes its sketch."""
    return SKETCH_KEY_PREFIX + nodeid


# ---------------------------------------------------------------------------
# The overlay
# ---------------------------------------------------------------------------

class ResonanceOverlay:
    """
    Vicinity-style overlay on top of Kademlia.

    Owns:
      * Our current sketch (provided by the daemon via update_sketch())
      * The top-N peers by resonance
      * Gossip + publish loops
    """

    # Daemon hooks — set after construction.
    on_peer_added: Optional[Callable[[ViewEntry], Awaitable[None]]] = None
    on_peer_removed: Optional[Callable[[str], Awaitable[None]]] = None

    def __init__(
        self,
        nodeid: str,
        wire_port: int,
        pubkey: bytes,
        dht: "KademliaNode",
        sketch_provider: Callable[[], List[float]],
        host_provider: Optional[Callable[[], Optional[str]]] = None,
        view_size: int = VIEW_SIZE,
    ):
        """
        host_provider: returns our public IP (e.g. from STUN). If None or
        if it returns None, we fall back to the local IP — fine on a LAN
        but useless across the internet.
        """
        self.nodeid = nodeid
        self.wire_port = wire_port
        self.pubkey = pubkey
        self.dht = dht
        self._sketch_provider = sketch_provider
        self._host_provider = host_provider
        self._view_size = view_size

        # The view: top-N peers by resonance.
        self._view: Dict[str, ViewEntry] = {}
        # NodeIDs we've seen and rejected (e.g. zero-resonance) — small TTL cache
        # to avoid re-checking the same peer every cycle.
        self._cooldown: Dict[str, float] = {}
        self._tasks: List[asyncio.Task] = []
        self._running = False
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks.append(asyncio.create_task(self._publish_loop()))
        self._tasks.append(asyncio.create_task(self._gossip_loop()))
        logger.info(f"overlay: started for {self.nodeid}")

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def top_resonant(self, n: Optional[int] = None) -> List[ViewEntry]:
        """Current top-N peers by resonance, highest first."""
        sorted_view = sorted(self._view.values(), key=lambda v: -v.resonance)
        if n is None:
            return sorted_view
        return sorted_view[:n]

    def all_peers(self) -> List[ViewEntry]:
        return list(self._view.values())

    async def publish_now(self) -> None:
        """Force a fresh publish of our sketch into the DHT."""
        await self._publish()

    async def lookup(self, nodeid: str) -> Optional[ViewEntry]:
        """
        Fetch a specific peer's record from the DHT (used by the daemon to
        resolve a contact's address from their NodeID alone). Bypasses our
        local view — goes straight to find_value.
        """
        data = await self.dht.iterative_find_value(sketch_key_for(nodeid))
        if data is None:
            return None
        rec = SelfRecord.from_bytes(data)
        if rec is None or rec.nodeid != nodeid:
            return None
        our_sketch = self._sketch_provider()
        return ViewEntry(
            nodeid=rec.nodeid,
            host=rec.host,
            port=rec.wire_port,
            sketch=rec.sketch,
            pubkey=rec.pubkey,
            resonance=cosine(our_sketch, rec.sketch) if rec.sketch else 0.0,
            last_updated=float(rec.ts),
        )

    # ------------------------------------------------------------------
    # Publishing our own record
    # ------------------------------------------------------------------

    async def _publish_loop(self) -> None:
        try:
            # First publish ASAP; then every PUBLISH_INTERVAL_S.
            await asyncio.sleep(2.0)  # let bootstrap settle
            while self._running:
                try:
                    await self._publish()
                except Exception as e:
                    logger.warning(f"overlay: publish failed: {e}")
                await asyncio.sleep(PUBLISH_INTERVAL_S)
        except asyncio.CancelledError:
            pass

    async def _publish(self) -> None:
        host = None
        if self._host_provider is not None:
            try:
                host = self._host_provider()
            except Exception:
                host = None
        if not host:
            host = self.dht._host  # fallback; works on LAN at least
        rec = SelfRecord(
            nodeid=self.nodeid,
            host=host,
            wire_port=self.wire_port,
            pubkey=self.pubkey,
            sketch=self._sketch_provider(),
            ts=int(time.time()),
        )
        n = await self.dht.store(sketch_key_for(self.nodeid), rec.to_bytes())
        logger.debug(f"overlay: published self-record to {n} replicas")

    # ------------------------------------------------------------------
    # Gossip — pick a random known peer, swap views, merge
    # ------------------------------------------------------------------

    async def _gossip_loop(self) -> None:
        try:
            await asyncio.sleep(3.0)  # post-bootstrap settle
            while self._running:
                try:
                    await self._gossip_round()
                except Exception as e:
                    logger.debug(f"overlay: gossip round error: {e}")
                # Jittered sleep so different nodes don't sync up.
                await asyncio.sleep(GOSSIP_INTERVAL_S * (0.7 + random.random() * 0.6))
        except asyncio.CancelledError:
            pass

    async def _gossip_round(self) -> None:
        """
        One round: pick a partner, fetch their record (which contains their
        sketch — useful in itself), then fetch some of their resonance peers
        and merge. We bootstrap our view by also incorporating peers from
        the Kademlia routing table.
        """
        # Source of partners: prefer our current view, but mix in a fresh
        # random peer from the routing table so we explore.
        candidates: List[Tuple[str, str, int]] = []

        # From routing table.
        for c in self.dht.routing.all_contacts():
            candidates.append((c.nodeid, c.host, c.port))

        # From current view (their kademlia port may differ; we don't have
        # it here, so we'll have to rely on the routing table to know how to
        # reach them. View-only entries are not directly queryable by us
        # for gossip — but their existence still informs candidate picks.)

        if not candidates:
            return

        # Pick a partner — random, with a slight bias toward our routing
        # table because that's where we actually have a UDP port to talk to.
        partner_id, _, _ = random.choice(candidates)
        if partner_id == self.nodeid:
            return

        # Step 1: fetch their published record. This also adds them to our
        # view if their resonance is good.
        partner_entry = await self.lookup(partner_id)
        if partner_entry is None:
            return  # haven't published, or we're partitioned from them
        await self._consider(partner_entry)

        # Step 2: fetch a handful of THEIR top-N from a separate DHT key.
        # The convention: each node publishes its top-K view alongside its
        # own record — we use a sibling key. This is cheap; the view is
        # at most VIEW_SIZE entries.
        view_data = await self.dht.iterative_find_value(
            sketch_key_for(partner_id) + "/view",
        )
        if view_data is None:
            return
        try:
            view_blob = json.loads(view_data.decode("utf-8"))
            entries_json = view_blob.get("peers", [])
        except Exception:
            return

        our_sketch = self._sketch_provider()
        for entry_d in entries_json[:VIEW_SIZE]:
            entry = ViewEntry.from_json(entry_d, our_sketch)
            if entry is not None and entry.nodeid != self.nodeid:
                await self._consider(entry)

    async def _consider(self, entry: ViewEntry) -> None:
        """Decide whether to admit `entry` into our view."""
        async with self._lock:
            # Cooldown: drop peers we recently rejected.
            now = time.time()
            cool = self._cooldown.get(entry.nodeid)
            if cool is not None and now < cool:
                return
            # Refresh existing entries.
            if entry.nodeid in self._view:
                old = self._view[entry.nodeid]
                old.host = entry.host
                old.port = entry.port
                old.sketch = entry.sketch
                old.pubkey = entry.pubkey or old.pubkey
                old.resonance = entry.resonance
                old.last_updated = entry.last_updated
                return
            # If our view has space, just add.
            if len(self._view) < self._view_size:
                self._view[entry.nodeid] = entry
                if self.on_peer_added is not None:
                    try:
                        await self.on_peer_added(entry)
                    except Exception as e:
                        logger.debug(f"on_peer_added raised: {e}")
                return
            # Otherwise compare to weakest entry.
            weakest = min(self._view.values(), key=lambda v: v.resonance)
            if entry.resonance > weakest.resonance:
                # Evict weakest, admit new.
                evicted_id = weakest.nodeid
                del self._view[evicted_id]
                self._view[entry.nodeid] = entry
                if self.on_peer_removed is not None:
                    try:
                        await self.on_peer_removed(evicted_id)
                    except Exception as e:
                        logger.debug(f"on_peer_removed raised: {e}")
                if self.on_peer_added is not None:
                    try:
                        await self.on_peer_added(entry)
                    except Exception as e:
                        logger.debug(f"on_peer_added raised: {e}")
            else:
                # Cooldown for an hour.
                self._cooldown[entry.nodeid] = now + 3600.0

    # ------------------------------------------------------------------
    # Publishing our top-N view alongside our self-record
    # ------------------------------------------------------------------

    async def publish_view(self) -> None:
        """Publish our current view to a sibling DHT key. Called periodically."""
        view = self.top_resonant()
        blob = {
            "ts": int(time.time()),
            "peers": [v.to_json() for v in view],
        }
        data = json.dumps(blob, separators=(",", ":")).encode("utf-8")
        # The view can grow with VIEW_SIZE; check size.
        if len(data) > 4096:
            # Truncate from the bottom (least-resonant) until we fit.
            while len(view) > 1:
                view = view[:-1]
                blob["peers"] = [v.to_json() for v in view]
                data = json.dumps(blob, separators=(",", ":")).encode("utf-8")
                if len(data) <= 4096:
                    break
        try:
            await self.dht.store(sketch_key_for(self.nodeid) + "/view", data)
        except Exception as e:
            logger.debug(f"overlay: publish_view failed: {e}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        view = self.top_resonant()
        return {
            "view_size": len(self._view),
            "best_resonance": view[0].resonance if view else 0.0,
            "worst_resonance": view[-1].resonance if view else 0.0,
            "avg_resonance": (sum(v.resonance for v in view) / len(view)) if view else 0.0,
        }

# ============================================================================
# dht_discovery.py
# ----------------------------------------------------------------------------
"""
peribus.dht_discovery — Discovery backend backed by Kademlia + the overlay

This is the seam between the DHT machinery and the rest of peribus. It
exposes the same `Discovery` interface as MdnsDiscovery and
RendezvousDiscovery — `start`, `stop`, `announce`, plus the
`on_peer_appeared` / `on_peer_disappeared` callbacks — so the daemon
just adds it to its discovery list and gets server-free global
discovery for free.

What happens at start:
  1. We start a Kademlia node on a UDP port (default kademlia_port,
     typically wire_port + 10).
  2. We bootstrap from any provided `bootstrap_peers` list — a few
     known (NodeID, host, kademlia_port) tuples that the operator
     trusts. After this, the routing table is seeded.
  3. We start the resonance overlay, which begins gossiping.
  4. We hand the daemon two streams of "peers appeared":
     - From the routing table: any node whose pubkey/resonance is
       relevant. The daemon's existing dial logic decides.
     - From the overlay's top-N: peers that are vector-close to us.

What happens at announce:
  Daemon calls announce() with our current sketch. We update the
  sketch_provider so the overlay's next publish carries it.

The bootstrap-peer list is the only soft centralization left. But:
  * Any peribus daemon with its kademlia port reachable IS a bootstrap
    peer. There's nothing special about the role — no signing key, no
    server software. It's just a regular node that happens to be
    reachable from your network.
  * Once the routing table has more than zero entries, you don't need
    the bootstrap any more. You can drop it from your config.
  * Friends can act as each other's bootstrap. Run a node on a VPS,
    hand the (id, host, port) tuple to your relatives — that's it.
"""


import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from peribus._discovery import Discovery, LocalAnnouncement, PeerInfo
from peribus._discovery import KademliaNode, Contact
from peribus._discovery import ResonanceOverlay, ViewEntry, sketch_key_for

logger = logging.getLogger(__name__)


# Default UDP port for kademlia. Convention: wire_port + 10.
DEFAULT_KAD_PORT_OFFSET = 10


def parse_bootstrap_peer(spec: str) -> Tuple[str, str, int]:
    """
    Parse a bootstrap-peer spec.

    Format: "NODEID@host:port" — the kademlia (UDP) host:port, not the
    wire (TCP) port. Example:
        z7ykyfj2lxp5rc4snxbqbciujm@bootstrap.example.org:5670

    Returns (nodeid, host, port).
    """
    if "@" not in spec:
        raise ValueError(f"bootstrap peer spec must be NODEID@host:port — got {spec!r}")
    nodeid, _, addr = spec.partition("@")
    if ":" not in addr:
        raise ValueError(f"bootstrap address must be host:port — got {addr!r}")
    host, _, port = addr.rpartition(":")
    return nodeid.strip(), host.strip(), int(port)


@dataclass
class _Tracked:
    """One peer we've surfaced through this discovery, with its current info."""
    info: PeerInfo
    last_refreshed: float


class DhtDiscovery(Discovery):
    """
    Server-free discovery: Kademlia DHT + vector-resonance overlay.

    Construction:
        disc = DhtDiscovery(
            nodeid=...,
            wire_port=...,                # peribus TCP wire port (announced to peers)
            pubkey=...,                   # raw 32-byte ed25519 pubkey
            sign=identity.sign,           # for signing DHT RPCs
            verify=verify_signature,
            pubkey_provider=identity.public_key_bytes,
            sketch_provider=lambda: ...,  # current vector sketch
            bootstrap_peers=[...],        # list of "NODEID@host:port" strings
            kad_host="0.0.0.0",
            kad_port=5670,                # UDP port for kademlia traffic
            host_provider=lambda: ...,    # optional: our public IP from STUN
        )
        disc.on_peer_appeared = ...
        disc.on_peer_disappeared = ...
        await disc.start()
    """

    def __init__(
        self,
        *,
        nodeid: str,
        wire_port: int,
        pubkey: bytes,
        sign: Optional[Callable[[bytes], bytes]] = None,
        verify: Optional[Callable[[bytes, bytes, bytes], bool]] = None,
        pubkey_provider: Optional[Callable[[], bytes]] = None,
        sketch_provider: Callable[[], list] = lambda: [],
        bootstrap_peers: Optional[List[str]] = None,
        kad_host: str = "0.0.0.0",
        kad_port: int = 0,
        host_provider: Optional[Callable[[], Optional[str]]] = None,
    ):
        self.nodeid = nodeid
        self.wire_port = wire_port
        self.pubkey = pubkey
        self._kad_host = kad_host
        self._kad_port = kad_port
        self._sketch_provider = sketch_provider
        self._host_provider = host_provider
        self._bootstrap_peers = bootstrap_peers or []

        # Build the DHT node.
        self.dht = KademliaNode(
            nodeid=nodeid,
            sign=sign,
            verify=verify,
            pubkey_provider=pubkey_provider,
        )

        # Overlay built later in start() so we can pass the DHT.
        self.overlay: Optional[ResonanceOverlay] = None

        # Tracking peers we've surfaced.
        self._tracked: Dict[str, _Tracked] = {}
        self._tasks: List[asyncio.Task] = []
        self._running = False

        # Daemon sets these before start().
        self._our_nodeid = nodeid

    # ------------------------------------------------------------------
    # Discovery interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        # Bring up the DHT.
        actual_port = await self.dht.start(host=self._kad_host, port=self._kad_port)
        self._kad_port = actual_port

        # Seed from bootstrap peers, if any.
        seeds: List[Contact] = []
        for spec in self._bootstrap_peers:
            try:
                nid, host, port = parse_bootstrap_peer(spec)
                if nid == self.nodeid:
                    continue
                seeds.append(Contact(nodeid=nid, host=host, port=port))
            except ValueError as e:
                logger.warning(f"dht_discovery: bad bootstrap spec {spec!r}: {e}")
        if seeds:
            n_alive = await self.dht.bootstrap(seeds)
            logger.info(
                f"dht_discovery: bootstrapped from {n_alive}/{len(seeds)} peers, "
                f"routing table has {self.dht.routing.size()} contacts"
            )
        else:
            logger.info("dht_discovery: no bootstrap peers; waiting for inbound contact")

        # Start overlay.
        self.overlay = ResonanceOverlay(
            nodeid=self.nodeid,
            wire_port=self.wire_port,
            pubkey=self.pubkey,
            dht=self.dht,
            sketch_provider=self._sketch_provider,
            host_provider=self._host_provider,
        )
        self.overlay.on_peer_added = self._on_overlay_added
        self.overlay.on_peer_removed = self._on_overlay_removed
        await self.overlay.start()

        # Periodic view-publish — sibling key alongside the self-record.
        self._tasks.append(asyncio.create_task(self._view_publish_loop()))

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
        if self.overlay is not None:
            await self.overlay.stop()
            self.overlay = None
        await self.dht.stop()
        # Fire disappear for everything we surfaced.
        for nodeid in list(self._tracked.keys()):
            if self.on_peer_disappeared:
                try:
                    await self.on_peer_disappeared(nodeid)
                except Exception:
                    pass
        self._tracked.clear()

    async def announce(self, info: LocalAnnouncement) -> None:
        """
        Daemon's sketch drifted; re-publish into the DHT so peers see the new
        sketch on their next overlay round. The overlay auto-publishes every
        PUBLISH_INTERVAL_S, but a fresh announce kicks an immediate publish.
        """
        if self.overlay is None:
            return
        # The sketch_provider closure picks up new values from the daemon
        # automatically; we just push out a publish.
        try:
            await self.overlay.publish_now()
        except Exception as e:
            logger.debug(f"dht_discovery: announce publish failed: {e}")

    # ------------------------------------------------------------------
    # Overlay -> Discovery bridging
    # ------------------------------------------------------------------

    async def _on_overlay_added(self, entry: ViewEntry) -> None:
        """
        Overlay surfaced a new resonant peer. Translate to PeerInfo and
        fire on_peer_appeared.
        """
        info = PeerInfo(
            nodeid=entry.nodeid,
            host=entry.host,
            port=entry.port,
            pubkey=entry.pubkey,
            sketch=entry.sketch,
            last_seen=time.time(),
        )
        self._tracked[entry.nodeid] = _Tracked(info=info, last_refreshed=time.time())
        if self.on_peer_appeared is not None:
            try:
                await self.on_peer_appeared(info)
            except Exception as e:
                logger.debug(f"on_peer_appeared raised: {e}")

    async def _on_overlay_removed(self, nodeid: str) -> None:
        if nodeid in self._tracked:
            del self._tracked[nodeid]
            if self.on_peer_disappeared is not None:
                try:
                    await self.on_peer_disappeared(nodeid)
                except Exception as e:
                    logger.debug(f"on_peer_disappeared raised: {e}")

    async def _view_publish_loop(self) -> None:
        """Publish our resonance view periodically so peers can gossip from us."""
        try:
            await asyncio.sleep(5.0)  # let bootstrap settle
            while self._running and self.overlay is not None:
                try:
                    await self.overlay.publish_view()
                except Exception as e:
                    logger.debug(f"dht_discovery: publish_view failed: {e}")
                await asyncio.sleep(60.0)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Public helpers used by the daemon
    # ------------------------------------------------------------------

    async def lookup_peer(self, nodeid: str) -> Optional[PeerInfo]:
        """
        Resolve a NodeID to (host, port, pubkey, sketch) via the DHT.
        Used by the daemon for invitation-based connections — you got
        a NodeID from an invite URL but no address.
        """
        if self.overlay is None:
            return None
        entry = await self.overlay.lookup(nodeid)
        if entry is None:
            return None
        return PeerInfo(
            nodeid=entry.nodeid,
            host=entry.host,
            port=entry.port,
            pubkey=entry.pubkey,
            sketch=entry.sketch,
            last_seen=time.time(),
        )

    def stats(self) -> dict:
        out = self.dht.stats()
        if self.overlay is not None:
            out.update({"overlay": self.overlay.stats()})
        out["tracked"] = len(self._tracked)
        return out

    @property
    def kad_port(self) -> int:
        return self._kad_port

    def bootstrap_self_url(self) -> str:
        """Return a 'NODEID@host:port' string others can use to bootstrap from us.

        Uses the host_provider if available (i.e. our public address from
        STUN); otherwise the local address the DHT bound to. Useful for
        printing on startup so users can paste it to their relatives.
        """
        host = None
        if self._host_provider is not None:
            try:
                host = self._host_provider()
            except Exception:
                host = None
        if not host:
            host = self.dht._host
        return f"{self.nodeid}@{host}:{self.kad_port}"