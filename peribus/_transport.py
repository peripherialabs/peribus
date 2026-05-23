"""
peribus._transport — concatenation of: wire.py, nat.py, upnp.py

This is a build artefact. The original module names live as section
banners below so `grep "^# ===="` jumps to each one.
"""

from __future__ import annotations


# ============================================================================
# wire.py
# ----------------------------------------------------------------------------
"""
peribus.wire — the on-the-wire protocol between two peer daemons

For peribus/0.1 we use line-delimited JSON over TCP. It's not the most
efficient encoding, but:

  * It's easy to debug (you can `nc` a peer and see what they say)
  * It composes naturally with the file abstractions (each post = one line)
  * Switching to QUIC + protobuf later is a transport change, not a
    protocol redesign

Each connection is bidirectional: once a peer dials us, both sides can
send announce/post/fetch/data/msg messages on the same connection.
"""


import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# Message types
MSG_HELLO    = "hello"     # initial handshake — exchange nodeid + pubkey
MSG_ANNOUNCE = "announce"  # current sketch + summary
MSG_POST     = "post"      # broadcast a post
MSG_FETCH    = "fetch"     # request content by hash
MSG_DATA     = "data"      # response to fetch
MSG_MSG      = "msg"       # direct message
MSG_PING     = "ping"
MSG_PONG     = "pong"
# Swarm semantic search — see peribus.app_swarm. The handlers live in
# AppSwarm.handle_message; the daemon's _on_wire_message dispatches there
# before checking the rest of its table.
MSG_APP_SEARCH  = "app_search"
MSG_APP_RESULTS = "app_results"

# Per-line read buffer for the JSON wire protocol. asyncio's default is
# 64 KiB, which one large MSG_APP_RESULTS payload can blow through —
# triggering a LimitOverrunError that drops the conn mid-protocol. Set
# this generously: 4 MiB covers a handful of inlined app sources plus
# room for previews and metadata, while still being a hard upper bound
# against pathological messages.
WIRE_READLINE_LIMIT = 4 * 1024 * 1024

# Application-level keepalive cadence.
#
# Without these, idle TCP conns get reaped by middleboxes (home routers
# typically drop NAT mappings after 30–120s for UDP, and aggressive
# enterprise gear does the same to idle TCP). The symptom is "peer keeps
# disconnecting and reconnecting" — exactly what we hit. SO_KEEPALIVE
# on the socket would help, but it's not portable across all the
# platforms peribus targets, and the default kernel cadence (~2 hours)
# is far longer than the typical NAT timeout anyway.
#
# Cadence rationale:
#   PING_INTERVAL_S — send a ping if we haven't sent or received
#     anything for this long. 25s is below the most aggressive
#     consumer NAT timeouts (30s for UDP on some carriers; TCP is
#     more forgiving but we want margin).
#   IDLE_TIMEOUT_S — if we've heard nothing back for this long
#     (including no pongs), the conn is considered dead and the
#     read loop drops it. The daemon's dial/discovery logic will
#     re-establish if the peer is still reachable.
PING_INTERVAL_S = 25.0
IDLE_TIMEOUT_S  = 90.0


@dataclass
class WireConn:
    """One open TCP connection to a peer."""
    nodeid: str                    # remote NodeID (set after hello)
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    last_recv: float = 0.0
    # When we last sent anything on this conn. Used by the keepalive
    # task to decide whether a ping is needed — any traffic resets
    # the timer, so chatty conns never bother with explicit pings.
    last_send: float = 0.0

    async def send(self, msg: dict) -> None:
        """Send one JSON-line message."""
        try:
            line = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
            self.writer.write(line)
            await self.writer.drain()
            self.last_send = time.time()
        except (ConnectionError, OSError) as e:
            logger.debug(f"wire send to {self.nodeid}: {e}")
            raise

    async def close(self) -> None:
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass


class WireServer:
    """
    Accepts inbound peer connections and dials outbound ones.
    Owned by the daemon; not safe to use from multiple loops.
    """

    def __init__(
        self,
        listen_port: int,
        on_message: Callable[[WireConn, dict], Awaitable[None]],
        on_disconnect: Callable[[str], Awaitable[None]],
    ):
        self.listen_port = listen_port
        self._on_message = on_message
        self._on_disconnect = on_disconnect
        self._conns: Dict[str, WireConn] = {}     # nodeid -> conn
        self._server: Optional[asyncio.AbstractServer] = None
        self._our_hello: Optional[dict] = None    # set by daemon at start

    def set_hello(self, hello: dict) -> None:
        """Daemon sets the hello payload before starting."""
        self._our_hello = hello

    async def start(self, host: str = "0.0.0.0") -> None:
        # limit=WIRE_READLINE_LIMIT: asyncio's default StreamReader
        # buffer is 64 KiB, which is hit by any single wire message
        # over that size — notably MSG_APP_RESULTS when responders
        # ship multiple hits or large previews. Raising the ceiling
        # to 4 MiB gives plenty of headroom without leaving the door
        # wide open for resource-exhaustion attacks.
        self._server = await asyncio.start_server(
            self._handle_inbound,
            host,
            self.listen_port,
            limit=WIRE_READLINE_LIMIT,
        )
        sockets = self._server.sockets
        if sockets:
            logger.info(f"peribus wire listening on {sockets[0].getsockname()}")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        # Close all peer connections.
        for conn in list(self._conns.values()):
            await conn.close()
        self._conns.clear()

    async def dial(self, nodeid: str, host: str, port: int) -> Optional[WireConn]:
        """
        Open an outbound connection to a peer. Returns the WireConn or
        None on failure. If we already have a conn to this nodeid, returns
        the existing one.
        """
        if nodeid in self._conns:
            return self._conns[nodeid]

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    host, port, limit=WIRE_READLINE_LIMIT,
                ),
                timeout=5.0,
            )
        except (asyncio.TimeoutError, ConnectionError, OSError) as e:
            logger.debug(f"dial {nodeid} @ {host}:{port}: {e}")
            return None

        conn = WireConn(nodeid=nodeid, reader=reader, writer=writer,
                        last_recv=time.time())

        # Send hello first.
        if self._our_hello:
            try:
                await conn.send(self._our_hello)
            except Exception:
                await conn.close()
                return None

        self._conns[nodeid] = conn
        # Spawn the read loop and the keepalive loop in parallel.
        # _read_loop returns when the conn drops (calls _drop in its
        # finally). _keepalive_loop notices the conn is gone and exits.
        asyncio.create_task(self._read_loop(conn))
        asyncio.create_task(self._keepalive_loop(conn))
        return conn

    def get_conn(self, nodeid: str) -> Optional[WireConn]:
        return self._conns.get(nodeid)

    def re_key(self, old_key: str, new_key: str) -> bool:
        """
        Move a conn from old_key to new_key in the conn table.

        Used after a hello arrives on a connection that was opened with
        a placeholder key (e.g. manual `dial 192.168.1.42` puts the conn
        under "pending:192.168.1.42:5660" until we learn the real NodeID).

        Returns True if the rekey happened, False if old_key wasn't a
        conn we knew about, or if new_key already had a different conn
        (collision: prefer the existing one and drop the new).
        """
        conn = self._conns.get(old_key)
        if conn is None:
            return False
        if old_key == new_key:
            return True
        existing = self._conns.get(new_key)
        if existing is not None and existing is not conn:
            # We already have a conn under the real NodeID (probably
            # because the peer dialed us back in parallel). Keep the
            # established one, drop the placeholder.
            asyncio.create_task(conn.close())
            self._conns.pop(old_key, None)
            return False
        del self._conns[old_key]
        conn.nodeid = new_key
        self._conns[new_key] = conn
        return True

    async def broadcast(self, msg: dict) -> None:
        """Send a message to every connected peer. Failures drop that peer silently."""
        # Snapshot to allow concurrent disconnects.
        for conn in list(self._conns.values()):
            try:
                await conn.send(msg)
            except Exception:
                await self._drop(conn)

    async def _handle_inbound(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        """Inbound peer: read their hello, then drop into the read loop."""
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
        except (asyncio.TimeoutError, ConnectionError):
            writer.close()
            return

        if not line:
            writer.close()
            return

        try:
            hello = json.loads(line.decode("utf-8"))
        except Exception:
            writer.close()
            return

        if hello.get("type") != MSG_HELLO or "from" not in hello:
            writer.close()
            return

        nodeid = hello["from"]

        # If we already know this peer, prefer the existing conn (avoid
        # duplicate connections from racing dials).
        if nodeid in self._conns:
            writer.close()
            return

        conn = WireConn(nodeid=nodeid, reader=reader, writer=writer,
                        last_recv=time.time())
        self._conns[nodeid] = conn

        # Reply with our hello.
        if self._our_hello:
            try:
                await conn.send(self._our_hello)
            except Exception:
                await self._drop(conn)
                return

        # Hand the hello to the daemon (so it can verify pubkey, register peer).
        try:
            await self._on_message(conn, hello)
        except Exception as e:
            logger.warning(f"on_message(hello) raised: {e}")

        # Keepalive runs in parallel with the read loop. We await the
        # read loop here (it owns the lifetime of the inbound handler);
        # the keepalive task exits on its own once the conn is dropped.
        asyncio.create_task(self._keepalive_loop(conn))
        await self._read_loop(conn)

    async def _read_loop(self, conn: WireConn) -> None:
        """
        Drain JSON lines from a conn until it closes.

        Tolerates:
          - LimitOverrunError: a single line exceeded readline's buffer.
            We drain the offending data and continue with the next
            line instead of dropping the whole conn. Without this, a
            single oversized message (e.g. an MSG_APP_RESULTS with too
            much inline content) would kill peer communication entirely.
          - JSONDecodeError / UnicodeDecodeError: garbled line, skip.
          - on_message exceptions: log and move on; never let one bad
            dispatch take down the read loop.

        Hard errors (ConnectionError, OSError) still drop the conn —
        those mean the socket itself is broken, not just one message.
        """
        try:
            while True:
                try:
                    line = await conn.reader.readline()
                except asyncio.LimitOverrunError as e:
                    # Drain past the oversized line: read and discard
                    # `consumed` bytes (which is the size of the buffer
                    # that contained no separator). The conn stays open.
                    logger.warning(
                        f"wire: oversized message from {conn.nodeid} "
                        f"({e.consumed} bytes); dropping line, keeping conn"
                    )
                    try:
                        await conn.reader.readexactly(e.consumed)
                    except (asyncio.IncompleteReadError, ConnectionError, OSError):
                        break
                    # Now eat through to the next newline so we
                    # resynchronize on a line boundary.
                    try:
                        await conn.reader.readuntil(b"\n")
                    except (asyncio.LimitOverrunError, ValueError):
                        # The next "line" is also oversized; give up
                        # on resync and drop. This shouldn't happen in
                        # practice unless the peer is malformed.
                        logger.warning(
                            f"wire: cannot resync stream from {conn.nodeid}; "
                            f"dropping conn"
                        )
                        break
                    except (asyncio.IncompleteReadError, ConnectionError, OSError):
                        break
                    continue
                except ValueError as e:
                    # readline can also raise plain ValueError in some
                    # asyncio versions when the limit is hit.
                    logger.warning(
                        f"wire: readline error from {conn.nodeid}: {e}; "
                        f"dropping conn"
                    )
                    break

                if not line:
                    break
                conn.last_recv = time.time()
                try:
                    msg = json.loads(line.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                try:
                    await self._on_message(conn, msg)
                except Exception as e:
                    logger.warning(f"on_message raised: {e}")
        except (ConnectionError, OSError):
            pass
        finally:
            await self._drop(conn)

    async def _keepalive_loop(self, conn: WireConn) -> None:
        """
        Send a MSG_PING when the conn has been idle longer than
        PING_INTERVAL_S, and drop the conn entirely if we haven't
        heard back from the peer for IDLE_TIMEOUT_S.

        Runs in parallel with _read_loop for the lifetime of one
        WireConn. Exits when:
          * The conn is no longer in the conn table (_drop ran).
          * IDLE_TIMEOUT_S elapses with no recv — we initiate the
            drop ourselves so a half-open TCP conn (the typical
            "NAT silently expired the mapping" case) doesn't sit
            around forever pretending to be alive.

        The peer side responds to MSG_PING with MSG_PONG (see the
        daemon's _on_wire_message dispatch table). The pong updates
        conn.last_recv via the normal _read_loop path, so a working
        peer never trips the idle timeout.

        Why explicit pings rather than SO_KEEPALIVE on the socket:
        the kernel default is ~2 hours, which is far longer than
        the timeouts we actually care about (consumer-grade NAT,
        carrier-grade NAT, corporate middleboxes). Tuning the
        per-socket TCP_KEEPIDLE/INTVL/CNT is possible but not
        portable across all platforms peribus targets; doing it
        in userspace works everywhere and is cheap.
        """
        # How often we wake up to check the idle threshold. Smaller
        # than PING_INTERVAL_S so jitter from sleep granularity
        # doesn't push the actual ping cadence well past the limit.
        check_interval = max(1.0, PING_INTERVAL_S / 5.0)
        try:
            while self._conns.get(conn.nodeid) is conn:
                await asyncio.sleep(check_interval)
                # _drop may have replaced us in the table while we
                # were sleeping; bail before doing any I/O.
                if self._conns.get(conn.nodeid) is not conn:
                    return
                now = time.time()
                # Idle timeout: we haven't heard from the peer in
                # too long. Force the conn down so the upper layers
                # (daemon dial / discovery) can redial cleanly.
                # last_recv is set on every successful readline in
                # _read_loop; if it's 0 we haven't received anything
                # yet, which is fine — fall back to the conn-creation
                # time captured by the initial last_recv.
                if conn.last_recv and now - conn.last_recv > IDLE_TIMEOUT_S:
                    logger.info(
                        f"wire: {conn.nodeid} idle for "
                        f"{now - conn.last_recv:.0f}s; dropping"
                    )
                    # Closing the writer poisons the reader on the
                    # peer side and, more importantly, makes our
                    # own readline return b"" so _read_loop falls
                    # into its finally and calls _drop. We don't
                    # call _drop directly here because the read
                    # loop is the canonical owner of conn lifetime.
                    try:
                        await conn.close()
                    except Exception:
                        pass
                    return
                # Ping if we haven't sent anything recently. Any
                # outbound traffic (announces, posts, fetches,
                # pongs) counts, so chatty conns send zero pings.
                if now - conn.last_send >= PING_INTERVAL_S:
                    try:
                        await conn.send({"type": MSG_PING})
                    except Exception:
                        # send() already logged; the read loop will
                        # surface the same socket error and drop.
                        return
        except asyncio.CancelledError:
            # Task was cancelled (shutdown). Don't try to do anything
            # clever; let the conn drop through the normal path.
            raise

    async def _drop(self, conn: WireConn) -> None:
        await conn.close()
        if self._conns.get(conn.nodeid) is conn:
            del self._conns[conn.nodeid]
            try:
                await self._on_disconnect(conn.nodeid)
            except Exception:
                pass

# ============================================================================
# nat.py
# ----------------------------------------------------------------------------
"""
peribus.nat — NAT traversal helpers

Two small things:

  1. STUN client. Asks a public STUN server "what does my outbound
     traffic look like from your end?" and gets back our public IP
     and port. We don't strictly need this — the rendezvous server
     observes our public IP from the connection itself — but knowing
     our own public address lets us advertise it accurately and
     detect symmetric NATs (where the public port we get to one
     destination is different from another, defeating hole punching).

  2. Hole-punch dialing. When the rendezvous server forwards a
     `punch_request` from a peer, both daemons should dial each other
     simultaneously. The first SYN from each side carves a NAT mapping;
     the second arrives at a now-open port. This is best-effort —
     symmetric-NAT-to-symmetric-NAT cannot be punched without a relay,
     and we don't ship a relay in v0.1.

We use STUN over UDP (RFC 5389) because that's what every public STUN
server speaks. The peribus wire is TCP, but a UDP STUN check is
sufficient for "is my NAT cone-ish or symmetric" — it's an approximation
either way.

Public STUN servers come and go. The defaults below are widely used and
operated by major providers; users can override via --stun.
"""


import asyncio
import logging
import os
import secrets
import socket
import struct
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


DEFAULT_STUN_SERVERS: List[str] = [
    "stun.l.google.com:19302",
    "stun1.l.google.com:19302",
    "stun.cloudflare.com:3478",
]


# RFC 5389 message types and attributes we care about.
_BINDING_REQUEST = 0x0001
_BINDING_RESPONSE = 0x0101
_ATTR_MAPPED_ADDRESS = 0x0001
_ATTR_XOR_MAPPED_ADDRESS = 0x0020
_MAGIC_COOKIE = 0x2112A442


@dataclass
class NatMapping:
    """Result of one STUN check."""
    public_ip: str
    public_port: int
    local_ip: str
    local_port: int

    @property
    def is_natted(self) -> bool:
        return self.public_ip != self.local_ip or self.public_port != self.local_port


def _parse_stun_addr(spec: str) -> Tuple[str, int]:
    if ":" in spec:
        host, _, port = spec.rpartition(":")
        return host, int(port)
    return spec, 3478


# ---------------------------------------------------------------------------
# STUN client
# ---------------------------------------------------------------------------

async def stun_lookup(
    servers: Optional[List[str]] = None, timeout: float = 3.0,
) -> Optional[NatMapping]:
    """
    Ask a STUN server for our public address. Tries each server in turn,
    returns the first successful response, or None if all fail.
    """
    servers = servers or DEFAULT_STUN_SERVERS
    for spec in servers:
        try:
            host, port = _parse_stun_addr(spec)
            mapping = await _stun_one(host, port, timeout)
            if mapping is not None:
                return mapping
        except Exception as e:
            logger.debug(f"stun {spec}: {e}")
            continue
    return None


async def _stun_one(host: str, port: int, timeout: float) -> Optional[NatMapping]:
    """Single STUN binding request/response over UDP."""
    loop = asyncio.get_running_loop()

    # Build a STUN binding request: header (20 bytes), no attributes.
    txid = secrets.token_bytes(12)
    request = struct.pack(
        "!HHI12s",
        _BINDING_REQUEST,    # message type
        0,                   # message length (no attrs)
        _MAGIC_COOKIE,       # magic cookie
        txid,                # transaction id
    )

    # We need a UDP socket and the local address it picks. We send to the
    # STUN server's resolved address.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    try:
        # Resolve in a thread so we don't block the loop.
        addrinfo = await loop.run_in_executor(
            None, socket.getaddrinfo, host, port, socket.AF_INET, socket.SOCK_DGRAM,
        )
        if not addrinfo:
            return None
        server_addr = addrinfo[0][4]

        # Bind so we know the local port.
        sock.bind(("0.0.0.0", 0))
        local_ip, local_port = sock.getsockname()

        await loop.sock_sendto(sock, request, server_addr) if hasattr(loop, "sock_sendto") else _sendto_compat(loop, sock, request, server_addr)

        # Wait for a response with timeout.
        try:
            data, _ = await asyncio.wait_for(_recvfrom(loop, sock), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        return _parse_stun_response(data, txid, local_ip, local_port)
    finally:
        sock.close()


async def _recvfrom(loop: asyncio.AbstractEventLoop, sock: socket.socket) -> Tuple[bytes, tuple]:
    """sock_recv but for UDP recvfrom. asyncio doesn't expose this directly."""
    fut = loop.create_future()

    def _ready():
        try:
            data, addr = sock.recvfrom(2048)
        except BlockingIOError:
            return
        except Exception as e:
            loop.remove_reader(sock.fileno())
            if not fut.done():
                fut.set_exception(e)
            return
        loop.remove_reader(sock.fileno())
        if not fut.done():
            fut.set_result((data, addr))

    loop.add_reader(sock.fileno(), _ready)
    try:
        return await fut
    finally:
        try:
            loop.remove_reader(sock.fileno())
        except Exception:
            pass


def _sendto_compat(
    loop: asyncio.AbstractEventLoop, sock: socket.socket, data: bytes, addr: tuple,
) -> None:
    """Fallback sendto for older asyncio without sock_sendto."""
    sock.sendto(data, addr)


def _parse_stun_response(
    data: bytes, expected_txid: bytes, local_ip: str, local_port: int,
) -> Optional[NatMapping]:
    """Walk the STUN response and pull out the mapped address."""
    if len(data) < 20:
        return None
    msg_type, msg_len, cookie, txid = struct.unpack("!HHI12s", data[:20])
    if msg_type != _BINDING_RESPONSE or txid != expected_txid:
        return None

    pos = 20
    end = 20 + msg_len
    if end > len(data):
        return None

    # Walk attributes. XOR-MAPPED-ADDRESS is preferred (RFC 5389) but old
    # servers only emit MAPPED-ADDRESS; accept either.
    mapped: Optional[Tuple[str, int]] = None
    while pos + 4 <= end:
        attr_type, attr_len = struct.unpack("!HH", data[pos:pos + 4])
        pos += 4
        attr_data = data[pos:pos + attr_len]
        pos += attr_len
        # Pad to 4-byte boundary.
        pad = (4 - (attr_len % 4)) % 4
        pos += pad

        if attr_type == _ATTR_XOR_MAPPED_ADDRESS and len(attr_data) >= 8:
            family = attr_data[1]
            xport = struct.unpack("!H", attr_data[2:4])[0] ^ (_MAGIC_COOKIE >> 16)
            if family == 0x01:  # IPv4
                xaddr = struct.unpack("!I", attr_data[4:8])[0] ^ _MAGIC_COOKIE
                ip = socket.inet_ntoa(struct.pack("!I", xaddr))
                mapped = (ip, xport)
                break  # XOR-MAPPED is the canonical answer; stop here.
        elif attr_type == _ATTR_MAPPED_ADDRESS and len(attr_data) >= 8:
            family = attr_data[1]
            port_v = struct.unpack("!H", attr_data[2:4])[0]
            if family == 0x01:
                ip = socket.inet_ntoa(attr_data[4:8])
                mapped = (ip, port_v)
                # don't break — prefer XOR-MAPPED if it shows up later

    if mapped is None:
        return None
    return NatMapping(
        public_ip=mapped[0],
        public_port=mapped[1],
        local_ip=local_ip,
        local_port=local_port,
    )


# ---------------------------------------------------------------------------
# Hole-punch dialing
# ---------------------------------------------------------------------------

async def punch_dial(
    host: str,
    port: int,
    *,
    attempts: int = 5,
    interval: float = 0.4,
    timeout_per_attempt: float = 2.0,
) -> Optional[Tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
    """
    Try to TCP-connect to (host, port) several times in quick succession.

    The point: when both peers receive a punch_request and start dialing
    each other simultaneously, the first SYN from each side opens a NAT
    pinhole. Subsequent SYNs from the other side find a now-open port
    and the connection completes. Without simultaneity, both sides get
    rejected by their own NAT.

    Returns the (reader, writer) of a successful connection, or None
    after `attempts` failures.

    This is a TCP version. UDP hole-punching is more reliable but
    peribus speaks TCP. In practice: works when at least one side has a
    cone NAT, fails on symmetric-to-symmetric. For the latter we'd need
    a TURN-style relay, which is future work.
    """
    for i in range(attempts):
        try:
            return await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout_per_attempt,
            )
        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            logger.debug(f"punch_dial attempt {i + 1}/{attempts} {host}:{port}: {e}")
            if i + 1 < attempts:
                await asyncio.sleep(interval)
    return None

# ============================================================================
# upnp.py
# ----------------------------------------------------------------------------
"""
peribus.upnp — automatic NAT traversal via UPnP IGD

The Internet Gateway Device profile of UPnP is what 99% of consumer
routers implement. The dance:

  1. SSDP discovery: send a multicast M-SEARCH to 239.255.255.250:1900
     asking for "WANIPConnection". The router answers with a URL pointing
     at its device-description XML.
  2. Fetch that XML, find the controlURL for WANIPConnection (or
     WANPPPConnection — older routers).
  3. Send a SOAP `AddPortMapping` to that controlURL, asking the router
     to forward "external port -> our LAN IP:internal port".
  4. On daemon shutdown, send `DeletePortMapping` to undo the forward —
     because leaving a permanent forward in someone's router is rude.

This module also returns the public IP the gateway sees, as a side
benefit. Saves us a STUN round-trip when UPnP succeeds.

We implement everything in pure Python (no `miniupnpc` C dep) because
peribus aims to install with `pip install` only, and adding a C
extension breaks that on systems without build tools. The implementation
is ~250 lines but well within asyncio's comfort zone — SSDP is one UDP
broadcast, SOAP is one HTTP POST.

Failure mode: every step degrades gracefully. If UPnP discovery fails
(common — some networks block SSDP, some routers have UPnP disabled),
we log a warning and return None. The caller treats UPnP as "tried,
didn't work" and the daemon falls back to the existing path (be a leech,
or rely on the user having configured port forwarding manually).

Security note: we deliberately do NOT use the SUBSCRIBE eventing or
GENA. The attack surface is large there and we don't need it. We only
make the two SOAP calls.
"""


import asyncio
import ipaddress
import logging
import re
import socket
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# Standards-defined SSDP multicast address + port. Same on all networks.
SSDP_ADDR = "239.255.255.250"
SSDP_PORT = 1900
SSDP_TIMEOUT_S = 3.0
HTTP_TIMEOUT_S = 5.0

# Service types we care about. WANIPConnection is the modern one (most
# routers since ~2008); WANPPPConnection is older but still seen on
# DSL gear that hasn't been updated in a decade.
SERVICE_TYPES = (
    "urn:schemas-upnp-org:service:WANIPConnection:1",
    "urn:schemas-upnp-org:service:WANIPConnection:2",
    "urn:schemas-upnp-org:service:WANPPPConnection:1",
)


@dataclass
class UpnpMapping:
    """A live port forward, with enough state to remove it on shutdown."""
    gateway_url: str
    service_type: str
    control_url: str
    external_port: int
    internal_port: int
    internal_ip: str
    protocol: str         # "UDP" or "TCP"
    public_ip: Optional[str] = None
    description: str = "peribus"


# ---------------------------------------------------------------------------
# Step 1: SSDP discovery
# ---------------------------------------------------------------------------

async def _discover_gateway(timeout: float = SSDP_TIMEOUT_S) -> Optional[Tuple[str, str]]:
    """
    Send a multicast M-SEARCH and listen for a router to respond with its
    device-description URL. Returns (location_url, service_type) of the
    first router that answers, or None.

    We send one M-SEARCH per service type and pick whichever responds first.
    """
    loop = asyncio.get_running_loop()

    # We need a socket with multicast enabled. asyncio's create_datagram_endpoint
    # doesn't expose all the multicast options we need cleanly, so we build the
    # socket by hand and wrap it.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.setblocking(False)
    try:
        # Bind to ephemeral so the kernel picks a port and so multiple
        # peribuses on the same host don't collide.
        sock.bind(("", 0))
    except OSError as e:
        logger.debug(f"upnp: couldn't bind ssdp socket: {e}")
        sock.close()
        return None

    # Build M-SEARCH messages. One per service type.
    requests = []
    for svc in SERVICE_TYPES:
        msg = (
            "M-SEARCH * HTTP/1.1\r\n"
            f"HOST: {SSDP_ADDR}:{SSDP_PORT}\r\n"
            'MAN: "ssdp:discover"\r\n'
            "MX: 2\r\n"
            f"ST: {svc}\r\n"
            "\r\n"
        )
        requests.append((msg.encode("ascii"), svc))

    # Send all of them. Routers reply by unicast back to our socket.
    for msg, _svc in requests:
        try:
            sock.sendto(msg, (SSDP_ADDR, SSDP_PORT))
        except OSError as e:
            logger.debug(f"upnp: ssdp send failed: {e}")
            sock.close()
            return None

    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            remaining = deadline - time.time()
            try:
                data = await asyncio.wait_for(
                    loop.sock_recv(sock, 4096), timeout=remaining,
                )
            except asyncio.TimeoutError:
                break
            text = data.decode("utf-8", errors="replace")
            # Parse the headers we care about: LOCATION and ST.
            location = _http_header(text, "LOCATION")
            st = _http_header(text, "ST")
            if location and st:
                logger.debug(f"upnp: discovered gateway {location} ({st})")
                return location, st
    finally:
        sock.close()

    logger.debug("upnp: no gateway responded to SSDP discovery")
    return None


def _http_header(text: str, name: str) -> Optional[str]:
    """Case-insensitive HTTP header lookup. SSDP/HTTP responses are small,
    so naive line splitting is fine."""
    name_lower = name.lower()
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        if k.strip().lower() == name_lower:
            return v.strip()
    return None


# ---------------------------------------------------------------------------
# Step 2: fetch device-description XML, find the WAN service controlURL
# ---------------------------------------------------------------------------

async def _fetch_device_description(
    location_url: str, want_service: str,
) -> Optional[Tuple[str, str]]:
    """
    GET the device description, parse it, return (service_type, control_url)
    for the first WAN service we find. We accept any of SERVICE_TYPES so a
    router that advertised WANIPConnection:2 in its SSDP response but only
    has WANIPConnection:1 in its description still works.
    """
    body = await _http_get(location_url)
    if body is None:
        return None
    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        logger.debug(f"upnp: device description xml parse error: {e}")
        return None

    # The XML namespace shifts between vendors. Strip it and walk the tree
    # by local-name only — uglier but vendor-portable.
    ns = "{urn:schemas-upnp-org:device-1-0}"
    for service in root.iter(ns + "service"):
        st_el = service.find(ns + "serviceType")
        cu_el = service.find(ns + "controlURL")
        if st_el is None or cu_el is None:
            continue
        st_text = (st_el.text or "").strip()
        cu_text = (cu_el.text or "").strip()
        if st_text in SERVICE_TYPES:
            # controlURL is often relative; resolve against location.
            absolute = urllib.parse.urljoin(location_url, cu_text)
            return st_text, absolute

    logger.debug("upnp: no WAN service found in device description")
    return None


async def _http_get(url: str) -> Optional[bytes]:
    """Tiny HTTP GET. We avoid bringing in a dep for one fetch."""
    return await _http_request("GET", url, headers={}, body=b"")


async def _http_request(
    method: str, url: str, *, headers: dict, body: bytes,
) -> Optional[bytes]:
    """Crude async HTTP/1.1 client — enough for two endpoints."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http":
        # UPnP IGD is always plain HTTP. HTTPS would be exotic and we'd skip.
        return None
    port = parsed.port or 80
    host = parsed.hostname
    if not host:
        return None
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=HTTP_TIMEOUT_S,
        )
    except (OSError, asyncio.TimeoutError) as e:
        logger.debug(f"upnp: http connect to {host}:{port} failed: {e}")
        return None

    try:
        # Build request.
        req_lines = [f"{method} {path} HTTP/1.1", f"Host: {host}:{port}",
                     "Connection: close", "Accept: */*"]
        for k, v in headers.items():
            req_lines.append(f"{k}: {v}")
        if body:
            req_lines.append(f"Content-Length: {len(body)}")
        req = ("\r\n".join(req_lines) + "\r\n\r\n").encode("ascii") + body
        writer.write(req)
        await writer.drain()

        # Read response.
        try:
            raw = await asyncio.wait_for(reader.read(65536), timeout=HTTP_TIMEOUT_S)
        except asyncio.TimeoutError:
            return None
        # Split headers from body. Routers do real HTTP/1.1, no chunked transfer
        # at typical sizes here.
        sep = raw.find(b"\r\n\r\n")
        if sep < 0:
            return None
        head = raw[:sep].decode("ascii", errors="replace")
        body_bytes = raw[sep + 4:]
        # Parse status.
        status_line = head.splitlines()[0] if head else ""
        if not status_line.startswith("HTTP/"):
            return None
        try:
            code = int(status_line.split()[1])
        except (IndexError, ValueError):
            return None
        if code != 200:
            logger.debug(f"upnp: http {method} {url} -> {code}")
            return None
        return body_bytes
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Step 3 & 4: SOAP AddPortMapping / DeletePortMapping
# ---------------------------------------------------------------------------

_SOAP_ADD = """<?xml version="1.0"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" \
s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
 <s:Body>
  <u:AddPortMapping xmlns:u="{service_type}">
   <NewRemoteHost></NewRemoteHost>
   <NewExternalPort>{external_port}</NewExternalPort>
   <NewProtocol>{protocol}</NewProtocol>
   <NewInternalPort>{internal_port}</NewInternalPort>
   <NewInternalClient>{internal_ip}</NewInternalClient>
   <NewEnabled>1</NewEnabled>
   <NewPortMappingDescription>{description}</NewPortMappingDescription>
   <NewLeaseDuration>0</NewLeaseDuration>
  </u:AddPortMapping>
 </s:Body>
</s:Envelope>
"""

_SOAP_DELETE = """<?xml version="1.0"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" \
s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
 <s:Body>
  <u:DeletePortMapping xmlns:u="{service_type}">
   <NewRemoteHost></NewRemoteHost>
   <NewExternalPort>{external_port}</NewExternalPort>
   <NewProtocol>{protocol}</NewProtocol>
  </u:DeletePortMapping>
 </s:Body>
</s:Envelope>
"""

_SOAP_GET_EXTERNAL_IP = """<?xml version="1.0"?>
<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" \
s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
 <s:Body>
  <u:GetExternalIPAddress xmlns:u="{service_type}"/>
 </s:Body>
</s:Envelope>
"""


async def _soap_call(control_url: str, service_type: str, action: str, body: str) -> Optional[bytes]:
    """Send a SOAP action to the control URL. Returns response body bytes."""
    headers = {
        "Content-Type": 'text/xml; charset="utf-8"',
        "SOAPAction": f'"{service_type}#{action}"',
    }
    return await _http_request(
        "POST", control_url, headers=headers, body=body.encode("utf-8"),
    )


def _local_ip_for(target_ip: str) -> Optional[str]:
    """Pick the LAN IP we'd use to reach `target_ip`. Used to find what
    address to put in NewInternalClient — must be the IP the router sees
    us as coming from, not e.g. our public IP from STUN."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((target_ip, 1))  # no packet sent for UDP connect
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Public API — what __main__.py calls
# ---------------------------------------------------------------------------

async def request_mapping(
    internal_port: int,
    protocol: str = "UDP",
    description: str = "peribus",
    preferred_external_port: Optional[int] = None,
) -> Optional[UpnpMapping]:
    """
    Try to set up a UPnP port forward. Returns an UpnpMapping on success
    that the caller stashes and later passes to release_mapping().

    `preferred_external_port` defaults to `internal_port` — the natural
    choice. If the router refuses (port already mapped to someone else),
    we don't currently retry; the caller can decide.

    Strategy:
      1. Discover the gateway via SSDP.
      2. Fetch its device description, find the WAN service.
      3. Figure out our LAN IP (whichever one the gateway sees).
      4. Send AddPortMapping. On 200 OK, also send GetExternalIPAddress
         so we can return the public IP.
    """
    if protocol not in ("UDP", "TCP"):
        raise ValueError("protocol must be UDP or TCP")
    external_port = preferred_external_port or internal_port

    # 1. discover
    discovered = await _discover_gateway()
    if discovered is None:
        return None
    location_url, advertised_service = discovered

    # 2. device description
    desc = await _fetch_device_description(location_url, advertised_service)
    if desc is None:
        return None
    service_type, control_url = desc

    # 3. figure out our LAN IP. Use the gateway's host as the target —
    # whichever interface the kernel routes to it from is the one the
    # router sees us coming from.
    gateway_host = urllib.parse.urlsplit(location_url).hostname
    if not gateway_host:
        return None
    internal_ip = _local_ip_for(gateway_host)
    if internal_ip is None:
        return None
    # Sanity: it should be a private address. If not, we're not behind a
    # NAT and UPnP doesn't apply.
    try:
        ip_obj = ipaddress.ip_address(internal_ip)
        if not ip_obj.is_private:
            logger.info(
                f"upnp: local ip {internal_ip} is public — no NAT, no mapping needed"
            )
            # Return a "mapping" that records the public IP but does no
            # forwarding. Lets the caller treat the result uniformly.
            return UpnpMapping(
                gateway_url=location_url,
                service_type=service_type,
                control_url=control_url,
                external_port=external_port,
                internal_port=internal_port,
                internal_ip=internal_ip,
                protocol=protocol,
                public_ip=internal_ip,
                description=description,
            )
    except ValueError:
        return None

    # 4. AddPortMapping
    soap_body = _SOAP_ADD.format(
        service_type=service_type,
        external_port=external_port,
        internal_port=internal_port,
        protocol=protocol,
        internal_ip=internal_ip,
        description=description,
    )
    resp = await _soap_call(control_url, service_type, "AddPortMapping", soap_body)
    if resp is None:
        logger.warning(
            f"upnp: AddPortMapping refused by gateway "
            f"(external:{external_port} -> {internal_ip}:{internal_port}/{protocol})"
        )
        return None

    # 4b. GetExternalIPAddress — bonus, lets us skip STUN
    public_ip: Optional[str] = None
    ip_resp = await _soap_call(
        control_url, service_type, "GetExternalIPAddress",
        _SOAP_GET_EXTERNAL_IP.format(service_type=service_type),
    )
    if ip_resp is not None:
        m = re.search(rb"<NewExternalIPAddress>([^<]+)</NewExternalIPAddress>", ip_resp)
        if m:
            public_ip = m.group(1).decode("ascii", errors="replace").strip()

    logger.info(
        f"upnp: mapped external {external_port}/{protocol} -> "
        f"{internal_ip}:{internal_port} (public ip: {public_ip or 'unknown'})"
    )
    return UpnpMapping(
        gateway_url=location_url,
        service_type=service_type,
        control_url=control_url,
        external_port=external_port,
        internal_port=internal_port,
        internal_ip=internal_ip,
        protocol=protocol,
        public_ip=public_ip,
        description=description,
    )


async def release_mapping(mapping: UpnpMapping) -> bool:
    """Tear down a previously-created mapping. Best effort — we don't
    retry, and we don't fail loudly if the router won't talk to us
    (typical at shutdown when the network is going away anyway)."""
    if mapping.public_ip == mapping.internal_ip:
        # We didn't actually create a forward (no-NAT path), nothing to release.
        return True
    soap_body = _SOAP_DELETE.format(
        service_type=mapping.service_type,
        external_port=mapping.external_port,
        protocol=mapping.protocol,
    )
    try:
        resp = await _soap_call(
            mapping.control_url, mapping.service_type,
            "DeletePortMapping", soap_body,
        )
    except Exception as e:
        logger.debug(f"upnp: release error: {e}")
        return False
    if resp is None:
        return False
    logger.info(
        f"upnp: released mapping external {mapping.external_port}/{mapping.protocol}"
    )
    return True