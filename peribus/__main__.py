from __future__ import annotations

#!/usr/bin/env python3
"""
peribusd — the peribus daemon

Starts the peribus daemon and exposes /n/peribus over 9P.

Quick start (auto-mount, local network only):
    python -m peribus --mount

Quick start (also reach the global network):
    python -m peribus --mount --bootstrap rdv.example.org:5670

Manual (two terminals):
    # term 1
    python -m peribus
    # term 2 (one-time setup)
    sudo mkdir -p /n/peribus && sudo chown $USER /n/peribus
    python ninepfuse.py 'tcp!127.0.0.1!5661' /n/peribus

Default behaviour:
    * Identity at ~/.peribus/identity/
    * Embedder loaded from /n/llm/embed (falls back to local hash)
    * mDNS discovery on the local network
    * If --bootstrap is given: also rendezvous discovery for the global network
    * Wire server on the chosen port (default 5660)
    * 9P server on port+1 (default 5661)
    * Contacts at ~/.peribus/contacts.json

See GLOBAL.md for how invitations work.
"""


import argparse
import asyncio
import getpass
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# Make peribus and rio importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ninep.server import Server9P
from peribus._daemon import PeribusDaemon
from peribus._discovery import LocalAnnouncement, PeerInfo
from peribus._foundation import make_sketch
from peribus._discovery import RendezvousDiscovery, DEFAULT_BOOTSTRAP
from peribus._discovery import DhtDiscovery
from peribus._foundation import verify_signature
from peribus._foundation import (
    ContactBook, Invite, make_invite, verify_invite, Contact,
)
from peribus._transport import stun_lookup, punch_dial
from peribus._transport import request_mapping, release_mapping, UpnpMapping


logger = logging.getLogger(__name__)


def _run_sudo(cmd: list[str]) -> bool:
    """Run a command via sudo, printing what we're doing first."""
    print(f"    $ sudo {' '.join(cmd)}")
    try:
        subprocess.run(["sudo"] + cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"    failed: {e}")
        return False


def _ensure_mountpoint(path: str) -> bool:
    user = getpass.getuser()

    if not os.path.isdir("/n"):
        print(f"  /n does not exist, creating...")
        if not _run_sudo(["mkdir", "-p", "/n"]):
            return False

    if not os.path.isdir(path):
        print(f"  {path} does not exist, creating...")
        if not _run_sudo(["mkdir", "-p", path]):
            return False

    if not os.access(path, os.W_OK):
        print(f"  fixing ownership of {path}...")
        if not _run_sudo(["chown", user, path]):
            return False

    if os.path.ismount(path):
        print(f"  {path} already mounted, unmounting...")
        subprocess.run(["fusermount", "-u", path], check=False)
        time.sleep(0.3)

    return True


def _find_ninepfuse() -> str | None:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "ninepfuse.py",
        here.parent.parent / "ninepfuse.py",
        here / "ninepfuse.py",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    if shutil.which("ninepfuse"):
        return "ninepfuse"
    return None


def _wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


# ---------------------------------------------------------------------------
# Global-discovery wiring
# ---------------------------------------------------------------------------

class _GlobalGlue:
    """
    Bundles global discovery (DHT and/or rendezvous) plus contacts-aware
    dialing onto the existing daemon, without touching daemon.py.

    Two discovery backends, both optional, both implementing the same
    Discovery interface:

      DhtDiscovery       — Kademlia + vector-resonance overlay, no servers.
                           Requires at least one bootstrap peer to join.
      RendezvousDiscovery — public-server-based fallback. Easier to set up
                           but introduces an operational dependency.

    You can run either, both, or neither. mDNS is handled by the daemon
    independently and stays on for LAN peers.
    """

    def __init__(self, daemon: PeribusDaemon, contacts: ContactBook):
        self.daemon = daemon
        self.contacts = contacts
        self.rendezvous: RendezvousDiscovery | None = None
        self.dht: DhtDiscovery | None = None
        self._public_host: str | None = None  # learned via STUN or UPnP
        self._upnp_mapping: UpnpMapping | None = None  # for cleanup at shutdown
        self._tasks: list[asyncio.Task] = []
        # Wrap the daemon's _on_peer_appeared so we can apply the
        # contacts-always-dial rule.
        self._orig_on_peer_appeared = daemon._on_peer_appeared
        daemon._on_peer_appeared = self._on_peer_appeared  # type: ignore[assignment]

    async def start(
        self,
        rendezvous_bootstrap: list[str],
        dht_bootstrap: list[str],
        kad_port: int,
        stun_servers: list[str] | None,
        upnp: bool = False,
    ) -> None:
        if not rendezvous_bootstrap and not dht_bootstrap and kad_port == 0:
            return

        # UPnP — try first, since a successful mapping gives us the public IP
        # for free and we'd rather skip STUN if so. Only meaningful when the
        # DHT is running.
        if upnp and (kad_port != 0 or dht_bootstrap):
            # If kad_port is 0, the daemon will pick a random port. We don't
            # know it yet, which makes UPnP forwarding pointless (we'd need
            # to bind first). So require an explicit kad_port for UPnP.
            if kad_port == 0:
                logger.warning(
                    "upnp: requires --kad-port to be set explicitly; skipping"
                )
            else:
                logger.info(f"upnp: requesting mapping for udp/{kad_port}...")
                self._upnp_mapping = await request_mapping(
                    internal_port=kad_port,
                    protocol="UDP",
                    description=f"peribus dht ({self.daemon.identity.nodeid[:8]})",
                )
                if self._upnp_mapping is not None:
                    if self._upnp_mapping.public_ip:
                        self._public_host = self._upnp_mapping.public_ip
                        logger.info(
                            f"upnp: public ip is {self._public_host} "
                            f"(skipping STUN)"
                        )
                else:
                    logger.warning(
                        "upnp: no mapping established. Falling back to "
                        "STUN for public-ip discovery."
                    )

        # STUN — diagnostic, but if it succeeds we use the public IP
        # for the DHT publish so peers across NATs can reach us. Skip if
        # UPnP already gave us one.
        if self._public_host is None and (stun_servers is None or stun_servers):
            try:
                mapping = await stun_lookup(stun_servers)
                if mapping is not None:
                    if mapping.is_natted:
                        logger.info(
                            f"NAT detected: local {mapping.local_ip}:{mapping.local_port} "
                            f"-> public {mapping.public_ip}:{mapping.public_port}"
                        )
                    else:
                        logger.info(f"no NAT: public {mapping.public_ip}:{mapping.public_port}")
                    self._public_host = mapping.public_ip
            except Exception as e:
                logger.debug(f"stun lookup failed: {e}")

        # DHT discovery (preferred path).
        if kad_port != 0 or dht_bootstrap:
            self.dht = DhtDiscovery(
                nodeid=self.daemon.identity.nodeid,
                wire_port=self.daemon.listen_port,
                pubkey=self.daemon.identity.public_key_bytes(),
                sign=self.daemon.identity.sign,
                verify=verify_signature,
                pubkey_provider=self.daemon.identity.public_key_bytes,
                sketch_provider=lambda: make_sketch(
                    self.daemon.identity_vector.snapshot()
                ),
                bootstrap_peers=dht_bootstrap,
                kad_port=kad_port if kad_port > 0 else 0,
                host_provider=lambda: self._public_host,
            )
            self.dht._our_nodeid = self.daemon.identity.nodeid
            self.dht.on_peer_appeared = self.daemon._on_peer_appeared
            self.dht.on_peer_disappeared = self.daemon._on_peer_disappeared
            await self.dht.start()
            logger.info(
                f"dht: bootstrap-self URL is {self.dht.bootstrap_self_url()} "
                f"(give this to peers who want to bootstrap from you)"
            )

            # Filesystem-level controls. These let callers script the daemon
            # by writing to /n/peribus/ctl and reading /n/peribus/bootstrap,
            # which is the Plan-9-native idiom — no extra CLI, no restart,
            # no out-of-band socket. Registered only when the DHT is up,
            # because both pieces are about the DHT.
            self.daemon.register_ctl("connect", self._ctl_connect)
            self.daemon.register_info(
                "bootstrap",
                lambda: self.dht.bootstrap_self_url() if self.dht else None,
            )

        # Rendezvous discovery (fallback path).
        if rendezvous_bootstrap:
            self.rendezvous = RendezvousDiscovery(
                bootstrap=rendezvous_bootstrap,
                identity_signer=self.daemon.identity.sign,
            )
            self.rendezvous._our_nodeid = self.daemon.identity.nodeid
            self.rendezvous.on_peer_appeared = self.daemon._on_peer_appeared
            self.rendezvous.on_peer_disappeared = self.daemon._on_peer_disappeared
            self.rendezvous.on_punch_request = self._handle_punch_request
            await self.rendezvous.start()
            await self._announce_rendezvous_now()

        # Periodic announce — drives both backends.
        self._tasks.append(asyncio.create_task(self._announce_loop()))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        if self.rendezvous is not None:
            await self.rendezvous.stop()
            self.rendezvous = None
        if self.dht is not None:
            await self.dht.stop()
            self.dht = None
        # Release the UPnP forward last, so it isn't torn down before the
        # daemon stops listening (a race that can briefly orphan the port).
        if self._upnp_mapping is not None:
            try:
                await release_mapping(self._upnp_mapping)
            except Exception as e:
                logger.debug(f"upnp release at shutdown failed: {e}")
            self._upnp_mapping = None

    async def _announce_rendezvous_now(self) -> None:
        if self.rendezvous is None:
            return
        sketch = make_sketch(self.daemon.identity_vector.snapshot())
        await self.rendezvous.announce(LocalAnnouncement(
            nodeid=self.daemon.identity.nodeid,
            port=self.daemon.listen_port,
            pubkey=self.daemon.identity.public_key_bytes(),
            sketch=sketch,
        ))

    async def _announce_dht_now(self) -> None:
        if self.dht is None:
            return
        # DHT picks up the current sketch via the closure we passed in;
        # we just trigger a publish.
        await self.dht.announce(LocalAnnouncement(
            nodeid=self.daemon.identity.nodeid,
            port=self.daemon.listen_port,
            pubkey=self.daemon.identity.public_key_bytes(),
            sketch=make_sketch(self.daemon.identity_vector.snapshot()),
        ))

    async def _announce_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(60.0)
                await self._announce_rendezvous_now()
                await self._announce_dht_now()
        except asyncio.CancelledError:
            pass

    async def _on_peer_appeared(self, info: PeerInfo) -> None:
        """Wrapper that applies contacts-always-dial after the daemon's normal handling."""
        await self._orig_on_peer_appeared(info)
        # If this peer is a contact, dial them regardless of resonance.
        if self.contacts.is_contact(info.nodeid):
            existing = self.daemon.wire.get_conn(info.nodeid)
            if existing is None:
                logger.info(f"contact {info.nodeid} appeared, dialing")
                conn = await self.daemon.wire.dial(info.nodeid, info.host, info.port)
                # If the direct dial failed and rendezvous is up, request a punch.
                if conn is None and self.rendezvous is not None:
                    logger.info(f"direct dial to {info.nodeid} failed, requesting punch")
                    await self.rendezvous.request_punch(info.nodeid)

    async def lookup_via_dht(self, nodeid: str) -> PeerInfo | None:
        """Resolve a NodeID to PeerInfo via the DHT (used for invitation imports)."""
        if self.dht is None:
            return None
        return await self.dht.lookup_peer(nodeid)

    async def _ctl_connect(self, arg: str) -> None:
        """Handler for `connect <NODEID@host:port>` written to /n/peribus/ctl.

        Bootstraps the running DHT against a new peer without restarting
        anything. The intended user flow:

            # On machine A, after start.py finishes, read your URL:
            cat /n/peribus/bootstrap
            # → z7yk...@198.51.100.42:5670
            # Send that to machine B out-of-band (text, email, paper).

            # On machine B:
            echo 'connect z7yk...@198.51.100.42:5670' > /n/peribus/ctl
            # The daemon does a Kademlia bootstrap; both views fill in;
            # the overlay starts gossiping; mutual discovery happens.

        Only one direction needs to do this — once one peer joins the
        other's DHT, the routing tables on both sides converge.
        """
        from peribus._discovery import Contact
        from peribus._discovery import parse_bootstrap_peer

        if self.dht is None:
            logger.warning("ctl connect: DHT is not enabled on this daemon")
            return
        if not arg:
            logger.warning("ctl connect: missing argument (expected NODEID@host:port)")
            return
        try:
            nodeid, host, port = parse_bootstrap_peer(arg)
        except ValueError as e:
            logger.warning(f"ctl connect: {e}")
            return
        if nodeid == self.daemon.identity.nodeid:
            logger.warning("ctl connect: refusing to bootstrap from self")
            return
        seed = Contact(nodeid=nodeid, host=host, port=port)
        logger.info(f"ctl connect: bootstrapping DHT from {arg}")
        n_alive = await self.dht.dht.bootstrap([seed])
        if n_alive > 0:
            logger.info(
                f"ctl connect: bootstrap succeeded, routing table now has "
                f"{self.dht.dht.routing.size()} contacts"
            )
            # Push our sketch out so they see us right away.
            try:
                await self.dht.overlay.publish_now()
                await self.dht.overlay.publish_view()
            except Exception as e:
                logger.debug(f"ctl connect: post-bootstrap publish failed: {e}")
        else:
            logger.warning(
                f"ctl connect: peer at {host}:{port} did not respond — "
                f"check the address, the port, and any NAT/firewall in between"
            )

    async def _handle_punch_request(self, nodeid: str, host: str, port: int) -> None:
        """A peer wants to talk. Dial them in the punch-friendly way."""
        if self.daemon.wire.get_conn(nodeid) is not None:
            return  # already connected
        logger.info(f"punch_request from {nodeid}, dialing {host}:{port}")
        result = await punch_dial(host, port)
        if result is None:
            logger.debug(f"hole-punch to {nodeid} at {host}:{port} failed")
            return
        # We have a TCP connection but haven't done the peribus handshake.
        # The simplest path: close this socket and let the wire layer's
        # normal dial() do the handshake. The pinhole stays open briefly
        # in most NATs, long enough for the second dial to walk through.
        reader, writer = result
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        # Now do the real, handshaking dial through the wire layer.
        conn = await self.daemon.wire.dial(nodeid, host, port)
        if conn is None:
            logger.debug(f"post-punch handshake dial to {nodeid} failed")


# ---------------------------------------------------------------------------
# Sub-commands: invite generation/import (run before/separate from daemon)
# ---------------------------------------------------------------------------

def _cmd_invite_create(args) -> None:
    """Generate a peribus:// invite URL for a target NodeID."""
    from peribus._foundation import Identity

    identity_dir = Path(args.identity_dir) if args.identity_dir else None
    identity = Identity.load_or_create(identity_dir)

    invite = make_invite(identity, args.target, ttl_s=args.ttl)
    url = invite.to_url()
    print(url)
    print(f"# from:    {invite.from_nodeid}", file=sys.stderr)
    print(f"# to:      {invite.to_nodeid}", file=sys.stderr)
    print(f"# expires: {time.strftime('%Y-%m-%d %H:%M', time.localtime(invite.expires_at))}",
          file=sys.stderr)


def _cmd_invite_import(args) -> None:
    """Verify an invite URL and add the issuer to your contacts."""
    try:
        invite = Invite.from_url(args.url)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    err = verify_invite(invite)
    if err is not None:
        print(f"invite rejected: {err}", file=sys.stderr)
        sys.exit(1)

    # The "to" NodeID should be us; warn if not.
    from peribus._foundation import Identity
    identity_dir = Path(args.identity_dir) if args.identity_dir else None
    identity = Identity.load_or_create(identity_dir)
    if invite.to_nodeid != identity.nodeid:
        print(
            f"warning: invite is addressed to {invite.to_nodeid}, "
            f"but our nodeid is {identity.nodeid}",
            file=sys.stderr,
        )
        if not args.force:
            print("re-run with --force to import anyway", file=sys.stderr)
            sys.exit(1)

    # Add the issuer (the person who sent us the invite) to contacts.
    contacts = ContactBook()
    import base64
    contact = Contact(
        nodeid=invite.from_nodeid,
        label=args.label or "",
        pubkey_b64=base64.b64encode(invite.issuer_pubkey).decode("ascii"),
        introduced_by="",  # self-imported
    )
    contacts.add(contact)
    print(f"added contact: {invite.from_nodeid}" + (f" ({args.label})" if args.label else ""))


def _cmd_contacts_list(args) -> None:
    contacts = ContactBook()
    for c in contacts.all():
        label = f"  ({c.label})" if c.label else ""
        print(f"{c.nodeid}{label}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="peribusd — the peribus mycelium daemon",
    )
    sub = ap.add_subparsers(dest="cmd")

    # The default (no subcommand) runs the daemon.
    ap.add_argument("--wire-port", type=int, default=5660,
                    help="port for peer-to-peer wire protocol (default 5660)")
    ap.add_argument("--ninep-port", type=int, default=5661,
                    help="port for 9P filesystem (default 5661)")
    ap.add_argument("--no-discovery", action="store_true",
                    help="don't start mDNS local discovery")
    ap.add_argument("--bootstrap", action="append", default=None,
                    metavar="HOST:PORT",
                    help="rendezvous server for global discovery (repeatable). "
                         "Falls back to DEFAULT_BOOTSTRAP if omitted.")
    ap.add_argument("--dht-bootstrap", action="append", default=None,
                    metavar="NODEID@HOST:PORT",
                    help="DHT bootstrap peer for server-free global discovery "
                         "(repeatable). The recommended global path.")
    ap.add_argument("--kad-port", type=int, default=0,
                    metavar="PORT",
                    help="UDP port for the Kademlia DHT (default: random). "
                         "Set this to a fixed port if you want others to "
                         "bootstrap from you reliably across restarts.")
    ap.add_argument("--no-dht", action="store_true",
                    help="disable the DHT entirely (only mDNS + rendezvous)")
    ap.add_argument("--upnp", action="store_true",
                    help="ask the router to forward our DHT port automatically "
                         "via UPnP. Eliminates manual port-forwarding for most "
                         "consumer routers. Mutually exclusive with --no-upnp.")
    ap.add_argument("--no-upnp", action="store_true",
                    help="explicitly disable UPnP even if a router would respond. "
                         "Use this if your network has UPnP enabled but you don't "
                         "trust it, or if you've forwarded the port manually.")
    ap.add_argument("--stun", action="append", default=None,
                    metavar="HOST:PORT",
                    help="STUN server for NAT detection (repeatable). "
                         "Use --stun '' to disable STUN entirely.")
    ap.add_argument("--llm-mount", default="/n/llm",
                    help="path to llmfs mount, for the embedder (default /n/llm)")
    ap.add_argument("--identity-dir", default=None,
                    help="override default ~/.peribus/identity location")
    ap.add_argument("--mount", action="store_true",
                    help="auto-mount the 9P FS")
    ap.add_argument("--mountpoint", default="/n/peribus",
                    help="where to mount the 9P FS (default /n/peribus)")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # `peribus invite create <nodeid>` -> prints peribus://invite/... URL
    p_inv = sub.add_parser("invite", help="manage invitations")
    inv_sub = p_inv.add_subparsers(dest="invite_cmd", required=True)
    p_create = inv_sub.add_parser("create", help="generate an invite for a NodeID")
    p_create.add_argument("target", help="NodeID of the person you're inviting")
    p_create.add_argument("--ttl", type=float, default=7 * 24 * 3600,
                          help="invite lifetime in seconds (default 7 days)")
    p_create.add_argument("--identity-dir", default=None)

    p_import = inv_sub.add_parser("import", help="import an invite you received")
    p_import.add_argument("url", help="peribus://invite/... URL")
    p_import.add_argument("--label", default="", help="human alias for this contact")
    p_import.add_argument("--force", action="store_true",
                          help="import even if addressed to a different nodeid")
    p_import.add_argument("--identity-dir", default=None)

    p_contacts = sub.add_parser("contacts", help="manage contacts")
    c_sub = p_contacts.add_subparsers(dest="contacts_cmd", required=True)
    c_sub.add_parser("list", help="list known contacts")

    return ap


def main() -> None:
    ap = _build_parser()
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(args, "log_level", "INFO"),
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.cmd == "invite":
        if args.invite_cmd == "create":
            _cmd_invite_create(args)
        elif args.invite_cmd == "import":
            _cmd_invite_import(args)
        return
    if args.cmd == "contacts":
        if args.contacts_cmd == "list":
            _cmd_contacts_list(args)
        return

    asyncio.run(_run_daemon(args))


async def _run_daemon(args) -> None:
    identity_dir = Path(args.identity_dir) if args.identity_dir else None

    daemon = PeribusDaemon(
        listen_port=args.wire_port,
        llm_mount=args.llm_mount,
        identity_dir=identity_dir,
    )
    contacts = ContactBook()
    glue = _GlobalGlue(daemon, contacts)

    # Resolve rendezvous bootstrap list: CLI flag > built-in defaults > nothing.
    rendezvous_bootstrap = list(args.bootstrap) if args.bootstrap else list(DEFAULT_BOOTSTRAP)

    # Resolve DHT bootstrap list. The DHT is on by default unless --no-dht;
    # an empty bootstrap list still starts the DHT (it just waits for
    # inbound connections from peers who bootstrap from us).
    dht_bootstrap = list(args.dht_bootstrap) if args.dht_bootstrap else []
    dht_enabled = not args.no_dht
    kad_port = args.kad_port if dht_enabled else 0

    # Resolve STUN servers: --stun '' disables; otherwise CLI > defaults.
    stun_servers: list[str] | None
    if args.stun is None:
        stun_servers = None
    elif args.stun == [""]:
        stun_servers = []
    else:
        stun_servers = [s for s in args.stun if s]

    loop = asyncio.get_running_loop()
    stopping = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stopping.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stopping.set())

    await daemon.start(with_discovery=not args.no_discovery)
    if rendezvous_bootstrap or dht_enabled:
        # Resolve UPnP: --upnp on, --no-upnp off, otherwise off (don't
        # surprise users by phoning their router without being asked).
        if args.no_upnp:
            upnp_enabled = False
        else:
            upnp_enabled = bool(args.upnp)
        await glue.start(
            rendezvous_bootstrap=rendezvous_bootstrap,
            dht_bootstrap=dht_bootstrap,
            kad_port=kad_port if dht_enabled else 0,
            stun_servers=stun_servers if stun_servers != [] else None,
            upnp=upnp_enabled,
        )

    server_9p = Server9P(daemon.fs_root)
    serve_task = asyncio.create_task(server_9p.serve_tcp("0.0.0.0", args.ninep_port))

    fuse_proc: subprocess.Popen | None = None
    if args.mount:
        print()
        print(f"  ── preparing mountpoint ──")
        if not _ensure_mountpoint(args.mountpoint):
            print(f"  ✗ could not prepare {args.mountpoint}; daemon still running.")
        else:
            ninepfuse = _find_ninepfuse()
            if ninepfuse is None:
                print(f"  ✗ ninepfuse.py not found; daemon still running.")
            else:
                if not _wait_for_port(args.ninep_port):
                    print(f"  ✗ 9P server not responding on port {args.ninep_port}")
                else:
                    addr = f"tcp!127.0.0.1!{args.ninep_port}"
                    cmd = [sys.executable, ninepfuse, addr, args.mountpoint]
                    print(f"    $ {' '.join(cmd)}")
                    fuse_proc = subprocess.Popen(cmd)

    n_contacts = len(contacts.all())
    print()
    print(f"  peribusd up")
    print(f"    nodeid:    {daemon.identity.nodeid}")
    print(f"    wire:      tcp!0.0.0.0!{args.wire_port}")
    print(f"    9p:        tcp!0.0.0.0!{args.ninep_port}")
    discovery_parts = []
    if not args.no_discovery:
        discovery_parts.append("mdns")
    if glue.dht is not None:
        discovery_parts.append(f"dht (udp!{glue.dht.kad_port})")
    if glue.rendezvous is not None:
        discovery_parts.append(
            f"rendezvous ({len(rendezvous_bootstrap)} server"
            f"{'s' if len(rendezvous_bootstrap) != 1 else ''})"
        )
    print(f"    discovery: {' + '.join(discovery_parts) if discovery_parts else 'off'}")
    if glue.dht is not None:
        # Print our bootstrap-self URL so the user can share it.
        print(f"    bootstrap: {glue.dht.bootstrap_self_url()}")
    print(f"    contacts:  {n_contacts}")
    if args.mount and fuse_proc:
        print(f"    mounted:   {args.mountpoint}")
    print()
    if not args.mount:
        print(f"  Mount manually with:")
        print(f"    sudo mkdir -p /n/peribus && sudo chown $USER /n/peribus")
        print(f"    python ninepfuse.py 'tcp!127.0.0.1!{args.ninep_port}' /n/peribus")
        print()
    if glue.dht is not None and not dht_bootstrap:
        print(f"  DHT is up but you provided no --dht-bootstrap peers.")
        print(f"  You'll be reachable to anyone who bootstraps from you, but")
        print(f"  you won't find anyone yourself until they connect to you,")
        print(f"  or you add a --dht-bootstrap peer. See GLOBAL.md.")
        print()
    elif not rendezvous_bootstrap and glue.dht is None:
        print(f"  No global discovery configured; only the local network is reachable.")
        print(f"  See GLOBAL.md to connect to the wider rhizome.")
        print()

    try:
        await stopping.wait()
    finally:
        print("\n  shutting down peribusd...")
        if fuse_proc is not None:
            print(f"  unmounting {args.mountpoint}...")
            subprocess.run(["fusermount", "-u", args.mountpoint], check=False)
            try:
                fuse_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                fuse_proc.terminate()
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass
        await glue.stop()
        await daemon.stop()
        print("  done.")


if __name__ == "__main__":
    main()