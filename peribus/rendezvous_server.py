from __future__ import annotations

"""
peribus.rendezvous_server — public bootstrap node

A small, stateless service that helps peribus daemons find each other
across the internet. It does NOT relay peer-to-peer traffic, store
content, or have any view into what peers are talking about. Its job
is exactly two things:

  1. Maintain a directory of currently-connected peers, keyed by NodeID,
     with their public IP/port and current vector sketch. New peers
     register, the registration lives as long as the connection does.

  2. When a peer queries, return the most-resonant other peers from the
     directory. Optionally, when asked, ferry a "punch" coordination
     message between two peers so they can attempt NAT traversal.

State is in-memory only. If you restart the server, every peer
reconnects within their next reconnect-backoff cycle (1-60s). There is
no database, no admin panel, no metrics endpoint by default — keeping
the operational surface tiny is the whole point. If you want
observability, add it for your deployment.

Run:
    python -m peribus.rendezvous_server --port 5670

Anyone can run one. We recommend at least two in any deployment for
redundancy. Operators of public peribus deployments may want to put a
TLS terminator (caddy/nginx) in front and use TLS, but the wire format
itself is fine over plaintext TCP — every payload is signed by the
NodeID's private key, and the server has no secrets to leak.

Wire protocol (JSON-line over TCP):

  Client -> server:
    { "type": "register", "nodeid": ..., "port": ..., "pubkey": b64,
      "sketch": b64, "ts": ms, "sig": b64, "v": "peribus-rdv/0.1" }
    { "type": "query", "limit": N, "sketch": b64 }
    { "type": "punch", "target": "<nodeid>" }
    { "type": "bye" }

  Server -> client:
    { "type": "ack", "for": "register"|"punch", "ok": true }
    { "type": "peers", "peers": [{nodeid, host, port, pubkey, sketch}, ...] }
    { "type": "punch_request", "nodeid": ..., "host": ..., "port": ... }
    { "type": "error", "reason": "..." }

The server NEVER returns a peer's pubkey unless that peer registered
one — and registered pubkeys are still verified by the receiver against
the claimed nodeid. The server is fundamentally untrusted.
"""


import argparse
import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


logger = logging.getLogger(__name__)


# Tunables — modest defaults, overridable on the CLI.
DEFAULT_PORT = 5670
MAX_PEERS_PER_QUERY = 64
REGISTRATION_TTL_S = 600.0          # if a conn lives this long without re-register, we drop it
MAX_REGISTRATIONS = 50_000          # per-process safety cap
MAX_PEERS_PER_IP = 64               # cheap abuse mitigation
QUERY_RATE_PER_S = 5.0              # per-connection query budget; bursts smoothed by token bucket
PUNCH_RATE_PER_MIN = 30             # per-connection punch budget per minute


# ---------------------------------------------------------------------------
# Registration table entry
# ---------------------------------------------------------------------------

@dataclass
class _Reg:
    """One peer's current registration in the directory."""
    nodeid: str
    host: str                 # public IP we see them coming from
    port: int                 # port THEY listen on (not the one they connected with)
    pubkey: bytes
    sketch: List[float]
    last_seen: float
    writer: asyncio.StreamWriter   # so we can ferry punch_request to them
    # Local rate-limit state.
    query_tokens: float = QUERY_RATE_PER_S
    last_token_refill: float = field(default_factory=time.time)
    punch_count: int = 0
    punch_window_start: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class RendezvousServer:
    """Stateless-ish public bootstrap node. One instance per process."""

    def __init__(self, port: int = DEFAULT_PORT, require_signatures: bool = False):
        """
        port: TCP listen port.
        require_signatures: if True, registers without a valid signature are
            rejected. We keep this off by default for v0.1 because the
            cryptography lib is optional, but turning it on is recommended
            for any deployment expecting non-trivial traffic.
        """
        self.port = port
        self.require_signatures = require_signatures
        # Two indices over the same data: nodeid -> Reg, and per-IP counts
        # to bound how many peers can come from one address.
        self._regs: Dict[str, _Reg] = {}
        self._ip_counts: Dict[str, int] = {}
        # Map StreamWriter -> nodeid so we know who closed when a conn drops.
        self._conn_owner: Dict[asyncio.StreamWriter, str] = {}
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self, host: str = "0.0.0.0") -> None:
        self._server = await asyncio.start_server(self._handle, host, self.port)
        sockets = self._server.sockets
        if sockets:
            logger.info(f"rendezvous server listening on {sockets[0].getsockname()}")
        # Background task: expire stale registrations whose conns are dead.
        asyncio.create_task(self._reaper())

    async def stop(self) -> None:
        if self._server is None:
            return
        # Close per-client connections first so handler tasks unblock from
        # their readline() loops; otherwise wait_closed() hangs waiting
        # for them. Snapshot writers because _drop_reg mutates _regs.
        writers = [r.writer for r in self._regs.values()]
        for w in writers:
            try:
                w.close()
            except Exception:
                pass
        # Stop accepting new connections.
        self._server.close()
        try:
            await asyncio.wait_for(self._server.wait_closed(), timeout=3.0)
        except asyncio.TimeoutError:
            # Some handler is stuck; we've done our best.
            pass
        self._server = None
        self._regs.clear()
        self._ip_counts.clear()
        self._conn_owner.clear()

    # ------------------------------------------------------------------
    # Per-connection handler
    # ------------------------------------------------------------------

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        peer_addr = writer.get_extra_info("peername")
        client_host = peer_addr[0] if peer_addr else "?"
        nodeid: Optional[str] = None

        try:
            while True:
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=300.0)
                except asyncio.TimeoutError:
                    # Idle clients get dropped. They'll reconnect.
                    return
                if not line:
                    return

                try:
                    msg = json.loads(line.decode("utf-8"))
                except Exception:
                    await self._error(writer, "bad json")
                    continue

                mtype = msg.get("type")
                if mtype == "register":
                    new_id = await self._handle_register(msg, client_host, writer)
                    if new_id is not None:
                        # If they re-registered with a different nodeid on the
                        # same conn (unusual), drop the old binding first.
                        if nodeid is not None and nodeid != new_id:
                            self._drop_reg(nodeid, client_host)
                        nodeid = new_id
                elif mtype == "query":
                    if nodeid is None:
                        await self._error(writer, "must register before query")
                        continue
                    await self._handle_query(msg, nodeid, writer)
                elif mtype == "punch":
                    if nodeid is None:
                        await self._error(writer, "must register before punch")
                        continue
                    await self._handle_punch(msg, nodeid, writer)
                elif mtype == "bye":
                    return
                else:
                    await self._error(writer, f"unknown type: {mtype}")
        except (ConnectionError, OSError):
            pass
        except Exception as e:
            logger.warning(f"rendezvous handler error: {e}")
        finally:
            if nodeid is not None:
                self._drop_reg(nodeid, client_host)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # register
    # ------------------------------------------------------------------

    async def _handle_register(
        self, msg: dict, client_host: str, writer: asyncio.StreamWriter,
    ) -> Optional[str]:
        try:
            nodeid = msg["nodeid"]
            port = int(msg["port"])
            pubkey = base64.b64decode(msg.get("pubkey", "") or "")
            sketch_b64 = msg.get("sketch", "") or ""
            sketch_bytes = base64.b64decode(sketch_b64) if sketch_b64 else b""
            ts = int(msg.get("ts", 0))
            sig_b64 = msg.get("sig", "")
        except (KeyError, ValueError, TypeError) as e:
            await self._error(writer, f"bad register: {e}")
            return None

        # Sanity bounds.
        if not nodeid or len(nodeid) > 64 or not port or port < 1 or port > 65535:
            await self._error(writer, "bad register fields")
            return None
        if len(pubkey) > 64:
            await self._error(writer, "pubkey too long")
            return None
        if len(sketch_bytes) > 4096:
            await self._error(writer, "sketch too long")
            return None

        # Verify nodeid <-> pubkey binding using the same hash peers use.
        # If they didn't supply a pubkey, we accept the registration but the
        # lack of pubkey means peers will get an empty pubkey field and may
        # choose to ignore the entry. That's their decision, not ours.
        if pubkey:
            from peribus._foundation import nodeid_from_pubkey
            if nodeid_from_pubkey(pubkey) != nodeid:
                await self._error(writer, "nodeid does not match pubkey")
                return None

        # Optionally verify the signature on the canonical body.
        if self.require_signatures or sig_b64:
            if not sig_b64 or not pubkey:
                await self._error(writer, "signature required")
                return None
            try:
                from peribus._foundation import verify_signature
                canon = f"{nodeid}|{port}|{ts}".encode("utf-8")
                if not verify_signature(pubkey, canon, base64.b64decode(sig_b64)):
                    await self._error(writer, "signature verify failed")
                    return None
            except Exception as e:
                await self._error(writer, f"signature error: {e}")
                return None

        # Per-IP cap.
        if self._ip_counts.get(client_host, 0) >= MAX_PEERS_PER_IP:
            existing = self._regs.get(nodeid)
            # Allow re-registration from the same IP for an existing nodeid.
            if existing is None or existing.host != client_host:
                await self._error(writer, "per-ip limit exceeded")
                return None

        # Global cap.
        if len(self._regs) >= MAX_REGISTRATIONS and nodeid not in self._regs:
            await self._error(writer, "server full")
            return None

        # Decode sketch.
        from peribus._foundation import unpack_vector
        sketch = unpack_vector(sketch_bytes) if sketch_bytes else []

        # If this nodeid was already registered (e.g. on a different conn),
        # close the old conn so we don't double-deliver punch messages.
        old = self._regs.get(nodeid)
        if old is not None and old.writer is not writer:
            try:
                old.writer.close()
            except Exception:
                pass
            self._ip_counts[old.host] = max(0, self._ip_counts.get(old.host, 0) - 1)
            self._conn_owner.pop(old.writer, None)

        reg = _Reg(
            nodeid=nodeid,
            host=client_host,
            port=port,
            pubkey=pubkey,
            sketch=sketch,
            last_seen=time.time(),
            writer=writer,
        )
        self._regs[nodeid] = reg
        self._ip_counts[client_host] = self._ip_counts.get(client_host, 0) + 1
        self._conn_owner[writer] = nodeid

        await self._send(writer, {"type": "ack", "for": "register", "ok": True})

        # Notify every other registered peer that this one just appeared.
        # Cheap (one small JSON line each) and gives near-instant
        # discovery instead of waiting for the next 30s query tick on
        # each side. If we don't do this, the first peer to register
        # learns about subsequent peers slowly — see the comment on
        # RendezvousDiscovery.announce for the alternative half-fix.
        # We skip this if the new peer was already registered (a
        # re-register, not a fresh appearance).
        if old is None:
            await self._broadcast_peer_added(reg)

        return nodeid

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    async def _handle_query(
        self, msg: dict, nodeid: str, writer: asyncio.StreamWriter,
    ) -> None:
        reg = self._regs.get(nodeid)
        if reg is None:
            await self._error(writer, "not registered")
            return

        # Token bucket — refill at QUERY_RATE_PER_S.
        now = time.time()
        elapsed = now - reg.last_token_refill
        reg.query_tokens = min(
            QUERY_RATE_PER_S * 2,  # small burst allowance
            reg.query_tokens + elapsed * QUERY_RATE_PER_S,
        )
        reg.last_token_refill = now
        if reg.query_tokens < 1.0:
            await self._error(writer, "rate limit")
            return
        reg.query_tokens -= 1.0
        reg.last_seen = now

        try:
            limit = min(int(msg.get("limit", 16)), MAX_PEERS_PER_QUERY)
            asker_sketch_b64 = msg.get("sketch", "") or ""
            asker_sketch_bytes = base64.b64decode(asker_sketch_b64) if asker_sketch_b64 else b""
        except (ValueError, TypeError) as e:
            await self._error(writer, f"bad query: {e}")
            return

        from peribus._foundation import unpack_vector, cosine
        asker_sketch = unpack_vector(asker_sketch_bytes) if asker_sketch_bytes else []

        # Rank candidates by cosine to asker's sketch when provided; otherwise
        # return most-recently-seen first.
        candidates: List[_Reg] = [r for r in self._regs.values() if r.nodeid != nodeid]
        if asker_sketch:
            candidates.sort(
                key=lambda r: -cosine(asker_sketch, r.sketch) if r.sketch else 0.0,
            )
        else:
            candidates.sort(key=lambda r: -r.last_seen)
        candidates = candidates[:limit]

        peers_payload = []
        for r in candidates:
            peers_payload.append({
                "nodeid": r.nodeid,
                "host": r.host,
                "port": r.port,
                "pubkey": base64.b64encode(r.pubkey).decode("ascii") if r.pubkey else "",
                "sketch": (
                    base64.b64encode(_pack(r.sketch)).decode("ascii")
                    if r.sketch else ""
                ),
            })
        await self._send(writer, {"type": "peers", "peers": peers_payload})

    # ------------------------------------------------------------------
    # punch — coordinate a hole-punch between two peers
    # ------------------------------------------------------------------

    async def _handle_punch(
        self, msg: dict, asker: str, writer: asyncio.StreamWriter,
    ) -> None:
        target = msg.get("target")
        if not target:
            await self._error(writer, "punch: no target")
            return

        reg = self._regs.get(asker)
        if reg is None:
            await self._error(writer, "not registered")
            return

        # Per-minute rate limit on punch requests — they cost more than queries
        # because they involve two peers.
        now = time.time()
        if now - reg.punch_window_start > 60.0:
            reg.punch_window_start = now
            reg.punch_count = 0
        if reg.punch_count >= PUNCH_RATE_PER_MIN:
            await self._error(writer, "punch rate limit")
            return
        reg.punch_count += 1

        target_reg = self._regs.get(target)
        if target_reg is None:
            await self._error(writer, "punch: target not connected")
            return

        # Send the requester's public address to the target.
        try:
            await self._send(target_reg.writer, {
                "type": "punch_request",
                "nodeid": asker,
                "host": reg.host,
                "port": reg.port,
            })
        except Exception as e:
            await self._error(writer, f"punch: forward failed: {e}")
            return

        # Tell the asker the punch is on its way; their daemon will start
        # dialing the target's known address (host:port) at roughly the
        # same time the target's daemon dials theirs.
        await self._send(writer, {
            "type": "ack",
            "for": "punch",
            "ok": True,
            "target": target,
            "host": target_reg.host,
            "port": target_reg.port,
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _drop_reg(self, nodeid: str, client_host: str) -> None:
        reg = self._regs.pop(nodeid, None)
        if reg is None:
            return
        self._ip_counts[reg.host] = max(0, self._ip_counts.get(reg.host, 0) - 1)
        self._conn_owner.pop(reg.writer, None)

    async def _broadcast_peer_added(self, new_reg: _Reg) -> None:
        """Push a peers-style notification to every other registered peer.

        Format mirrors a single entry from a `peers` response, so the
        client can reuse its existing _handle_peers code path.
        """
        payload = {
            "type": "peers",
            "peers": [{
                "nodeid": new_reg.nodeid,
                "host": new_reg.host,
                "port": new_reg.port,
                "pubkey": (
                    base64.b64encode(new_reg.pubkey).decode("ascii")
                    if new_reg.pubkey else ""
                ),
                "sketch": (
                    base64.b64encode(_pack(new_reg.sketch)).decode("ascii")
                    if new_reg.sketch else ""
                ),
            }],
        }
        # Snapshot so concurrent registrations don't mutate the dict mid-iter.
        for r in list(self._regs.values()):
            if r.nodeid == new_reg.nodeid:
                continue
            try:
                await self._send(r.writer, payload)
            except Exception:
                # _send already drops broken regs; nothing more to do.
                pass

    async def _send(self, writer: asyncio.StreamWriter, msg: dict) -> None:
        try:
            line = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
            writer.write(line)
            await writer.drain()
        except Exception:
            # If a peer's writer is broken, drop the registration. The reaper
            # would catch this eventually but we'd rather not keep a dead
            # entry in the directory.
            owner = self._conn_owner.get(writer)
            if owner:
                self._drop_reg(owner, "")

    async def _error(self, writer: asyncio.StreamWriter, reason: str) -> None:
        await self._send(writer, {"type": "error", "reason": reason})

    async def _reaper(self) -> None:
        """Drop registrations whose connections went stale or silent."""
        while self._server is not None:
            await asyncio.sleep(60.0)
            now = time.time()
            for nodeid, reg in list(self._regs.items()):
                if now - reg.last_seen > REGISTRATION_TTL_S:
                    try:
                        reg.writer.close()
                    except Exception:
                        pass
                    self._drop_reg(nodeid, reg.host)


def _pack(vec: List[float]) -> bytes:
    """Local copy to avoid importing pack_vector at top-level (kept light)."""
    from peribus._foundation import pack_vector
    return pack_vector(vec)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _main() -> None:
    ap = argparse.ArgumentParser(description="peribus rendezvous bootstrap server")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--require-signatures", action="store_true",
                    help="reject registrations without a valid Ed25519 signature")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(_run(args))


async def _run(args) -> None:
    server = RendezvousServer(
        port=args.port, require_signatures=args.require_signatures,
    )
    await server.start(host=args.host)
    print(f"  rendezvous up on {args.host}:{args.port}")
    print(f"  signatures: {'required' if args.require_signatures else 'optional'}")
    print(f"  Ctrl+C to stop")
    try:
        # Block forever — the server task runs in the background.
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    _main()