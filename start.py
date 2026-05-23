#!/usr/bin/env python3
"""
Rio Mux — Start Tool (GUI by default, CLI fallback)

This is the merged successor of start_peribus.py + start_gui.py. It
provides a single entry point that:

  • Opens the PySide6 GUI by default (frameless paper card).
  • Falls back to the interactive CLI when PySide6 is unavailable,
    or when run with --cli, or when stdin is a TTY and --no-tty-cli
    isn't set... wait, simpler than that:
        --gui   → force GUI; error if PySide6 missing
        --cli   → force CLI (interactive prompts)
        (none)  → try GUI first; on ImportError fall back to CLI

What it does:
  1. Ensures /n exists (and /n/llm, /n/rio for standalone mounts)
  2. Lets you pick a mode:
     a) Create a new mux  → starts llmfs, rio, riomux, mounts /n
     b) Connect to a mux  → mounts remote riomux via ninepfuse
     c) Standalone         → starts llmfs + rio, mounts each separately

Authentication (single shared token):
  A single auth token is generated (or provided) and passed everywhere:

      llmfs   --auth-token <tok>                    (backend auth)
      rio     --auth-token <tok>                    (backend auth)
      riomux  --auth-token <tok>                    (mux client auth)
              --backend rio=127.0.0.1:5641:<tok>    (mux→rio Tauth)
              --backend llm=127.0.0.1:5640:<tok>    (mux→llmfs Tauth)
              --backend peribus=127.0.0.1:5661      (peribus is unauthed)

  Same token mounts the FUSE side:
      python ninepfuse.py 'tcp!127.0.0.1!5642' /n -t <tok>

  Plan 9 interop (factotum + p9any/pass):
      key proto=pass user=glenda !password=<tok> dom=ninep
      mount -a tcp!host!5642 /n/mux

The mux exposes:
    /n/
    ├── <name>/     → rio (display server + scene)
    ├── llm/        → llmfs (agents, providers)
    └── peribus/    → peribusd (optional social/feed layer)

Routing example (works correctly through the mux, blocking-safe):
    while true; do cat /n/llm/claude/output > /n/<name>/scene/parse; done
"""

import argparse
import getpass
import os
import secrets
import shutil
import signal
import socket
import string
import subprocess
import sys
import time


# ── Detect whether the GUI is even possible ──────────────────────
#
# We probe PySide6 once at import time. The result is consulted both
# by the CLI/GUI dispatcher in main() and (lazily) by the GUI
# widget classes below. When PySide6 isn't present, every GUI symbol
# we reference is replaced by a stub that raises a clear error —
# this lets us define the GUI classes unconditionally without
# crashing the CLI fallback path.

try:
    from PySide6.QtCore import (
        Qt, QTimer, Signal, QObject, QThread, QPoint, Slot,
        QVariantAnimation, QEasingCurve,
    )
    from PySide6.QtGui import (
        QFont, QTextCursor, QColor, QPalette, QPainter, QBrush, QPen,
    )
    from PySide6.QtWidgets import (
        QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
        QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox,
        QFrame, QPlainTextEdit, QSpinBox,
        QMessageBox, QScrollArea, QGraphicsDropShadowEffect,
    )
    GUI_AVAILABLE = True
    GUI_IMPORT_ERROR = None
except ImportError as _gui_err:
    GUI_AVAILABLE = False
    GUI_IMPORT_ERROR = _gui_err


# ── Resolve project root ─────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # one level up from riomux/

# Prepend PROJECT_ROOT to PYTHONPATH for every subprocess we spawn,
# so `python -m llmfs.main` etc. find their packages.
_env = os.environ.copy()
_existing_pp = _env.get("PYTHONPATH", "")
_env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + _existing_pp if _existing_pp else "")


# ── Defaults ─────────────────────────────────────────────────────

LLMFS_PORT         = 5640
RIO_PORT           = 5641
MUX_PORT           = 5642
PERIBUS_WIRE_PORT  = 5660   # peer-to-peer wire protocol
PERIBUS_NINEP_PORT = 5661   # 9P backend exposed to the mux
PERIBUS_KAD_PORT   = 5670   # UDP, Kademlia DHT — fixed so our bootstrap URL is stable across restarts

MOUNT_BASE = "/n"


# ─────────────────────────────────────────────────────────────────
# ── Shared utilities (used by both CLI and GUI paths) ────────────
# ─────────────────────────────────────────────────────────────────


def prompt(msg, default=None):
    """Prompt with optional default."""
    if default:
        raw = input(f"  {msg} [{default}]: ").strip()
        return raw if raw else default
    return input(f"  {msg}: ").strip()


def prompt_choice(msg, choices, default=None):
    """Prompt the user to pick from numbered choices."""
    print(f"\n  {msg}")
    for i, (key, label) in enumerate(choices, 1):
        marker = " (default)" if key == default else ""
        print(f"    {i}) {label}{marker}")

    while True:
        raw = input(f"  Choice [1-{len(choices)}]: ").strip()
        if not raw and default:
            return default
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx][0]
        except ValueError:
            pass
        print(f"    Please enter 1-{len(choices)}")


def prompt_yes_no(msg, default=True):
    """Prompt for yes/no."""
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"  {msg} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def run_sudo(cmd, check=True):
    """Run a command with sudo."""
    print(f"    $ sudo {' '.join(cmd)}")
    return subprocess.run(["sudo"] + cmd, check=check)


def check_binary(name):
    """Check if a binary is on PATH."""
    return shutil.which(name) is not None


def wait_for_port(port, host="127.0.0.1", timeout=10):
    """Wait for a TCP port to become available."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.3)
    return False


def generate_token(length=32):
    """Generate a cryptographically secure random token."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


# ── Setup directories ────────────────────────────────────────────

def ensure_dir(path):
    """Ensure a directory exists and is writable."""
    if not os.path.isdir("/n"):
        print("  /n does not exist, creating...")
        run_sudo(["mkdir", "-p", "/n"])

    if not os.path.isdir(path):
        print(f"  {path} does not exist, creating...")
        run_sudo(["mkdir", "-p", path])

    user = getpass.getuser()
    if not os.access(path, os.W_OK):
        print(f"  Fixing ownership of {path}...")
        run_sudo(["chown", user, path])

    return True


def ensure_mount_base():
    """Ensure /n exists and is writable."""
    print("\n── Checking filesystem ──")
    ensure_dir(MOUNT_BASE)
    print(f"  ✓ {MOUNT_BASE} ready")


# ── Auth token management ────────────────────────────────────────

def setup_auth_token(provided_token=None):
    """
    Set up auth token for the mux session.

    If a token is provided (CLI arg or env), use it.
    Otherwise, offer to generate one or skip auth.

    Returns the token string, or None if auth is disabled.
    """
    # Check environment
    env_token = os.environ.get("RIOMUX_AUTH_TOKENS", "").strip()

    if provided_token:
        print(f"  Using provided auth token")
        return provided_token

    if env_token:
        # Use first token from env
        token = env_token.split(",")[0].strip()
        print(f"  Using auth token from RIOMUX_AUTH_TOKENS")
        return token

    # Interactive prompt
    enable = prompt_yes_no("Enable authentication?", default=True)
    if not enable:
        print("  ⚠ Auth disabled — anyone on the network can connect")
        return None

    choice = prompt_choice(
        "Auth token:",
        [
            ("generate", "Generate a random token"),
            ("enter",    "Enter a token manually"),
        ],
        default="generate",
    )

    if choice == "generate":
        token = generate_token()
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  Auth Token: {token}  │")
        print(f"  └─────────────────────────────────────────┘")
        print(f"  Save this token — you'll need it to connect remotely.")
        print(f"  Plan 9: key proto=pass dom=riomux !password={token}")
        return token
    else:
        token = prompt("Enter auth token")
        if not token:
            print("  ⚠ Empty token — auth disabled")
            return None
        return token


# ── Mount helpers ────────────────────────────────────────────────

def find_ninepfuse():
    """Find our ninepfuse.py client."""
    # Check in project root
    candidates = [
        os.path.join(PROJECT_ROOT, "ninepfuse.py"),
        os.path.join(SCRIPT_DIR, "..", "ninepfuse.py"),
        os.path.join(SCRIPT_DIR, "ninepfuse.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    # Check if it's importable
    if shutil.which("ninepfuse"):
        return "ninepfuse"
    return None


def mount_ninepfuse(host, port, mountpoint, auth_token=None, user=None):
    """
    Mount a 9P server via our ninepfuse FUSE client.
    Returns the Popen process or None on failure.
    """
    os.makedirs(mountpoint, exist_ok=True)

    # Check if already mounted
    if os.path.ismount(mountpoint):
        print(f"  ⚠ {mountpoint} already mounted, unmounting...")
        subprocess.run(["fusermount", "-u", mountpoint], check=False)
        time.sleep(0.5)

    ninepfuse = find_ninepfuse()
    if ninepfuse is None:
        print("  ✗ ninepfuse.py not found")
        return None

    addr = f"tcp!{host}!{port}"
    cmd = [sys.executable, ninepfuse, addr, mountpoint]

    if auth_token:
        cmd.extend(["-t", auth_token])
    if user:
        cmd.extend(["-u", user])

    print(f"  Mounting {host}:{port} → {mountpoint}")
    proc = subprocess.Popen(cmd, env=_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)

    # Give it a moment to mount
    time.sleep(2)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode()
        print(f"  ✗ ninepfuse failed: {stderr}")
        return None

    # Verify mount
    if os.path.isdir(mountpoint):
        try:
            os.listdir(mountpoint)
            print(f"  ✓ Mounted {mountpoint}")
            return proc
        except OSError:
            pass

    print(f"  ✗ Mount verification failed for {mountpoint}")
    proc.terminate()
    return None


def mount_9pfuse(host, port, mountpoint):
    """Mount a 9P server via plan9port 9pfuse (no auth). Returns Popen or None."""
    os.makedirs(mountpoint, exist_ok=True)

    if os.path.ismount(mountpoint):
        print(f"  ⚠ {mountpoint} already mounted, unmounting...")
        subprocess.run(["fusermount", "-u", mountpoint], check=False)
        time.sleep(0.5)

    addr = f"{host}:{port}"
    print(f"  Mounting {addr} → {mountpoint} (9pfuse, no auth)")
    proc = subprocess.Popen(
        ["9pfuse", addr, mountpoint],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    time.sleep(1)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode()
        print(f"  ✗ 9pfuse failed: {stderr}")
        return None

    if os.path.isdir(mountpoint):
        try:
            os.listdir(mountpoint)
            print(f"  ✓ Mounted {mountpoint}")
            return proc
        except OSError:
            pass

    print(f"  ✗ Mount verification failed for {mountpoint}")
    proc.terminate()
    return None


def unmount(mountpoint):
    """Unmount a FUSE mount."""
    if os.path.ismount(mountpoint):
        subprocess.run(["fusermount", "-u", mountpoint], check=False)


# ── Mode: Create new mux ────────────────────────────────────────

def nuke_stale_mounts():
    """
    Kill ALL existing 9pfuse/ninepfuse processes and unmount everything under /n/.
    """
    print("\n── Cleaning stale mounts ──")

    # Kill stale mount processes
    for procname in ["9pfuse", "ninepfuse"]:
        result = subprocess.run(
            ["pgrep", "-a", procname], capture_output=True, text=True
        )
        if result.stdout.strip():
            print(f"  Killing stale {procname} processes:")
            for line in result.stdout.strip().splitlines():
                print(f"    {line}")
            subprocess.run(["pkill", "-9", procname], check=False)
            time.sleep(0.5)

    # Kill stale attachment scripts
    subprocess.run(["pkill", "-f", "llmfs_attach"], capture_output=True)
    subprocess.run(["pkill", "-f", "acme_attach"], capture_output=True)

    # Unmount everything under /n/
    stale_mounts = []
    try:
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[1].startswith("/n"):
                    stale_mounts.append(parts[1])
    except Exception:
        pass

    if stale_mounts:
        stale_mounts.sort(key=len, reverse=True)
        print(f"  Unmounting {len(stale_mounts)} stale mount(s):")
        for mp in stale_mounts:
            print(f"    fusermount -u {mp}")
            subprocess.run(["fusermount", "-u", mp], check=False)
        time.sleep(0.5)
    else:
        print("  No stale mounts found under /n/")



# ─────────────────────────────────────────────────────────────────
# ── Mode functions (create / connect / standalone) ───────────────
# ─────────────────────────────────────────────────────────────────

def mode_create_mux(name, auth_token, peribus_cfg, processes, mounts):
    """Start llmfs, rio, riomux, and mount under /n.

    peribus_cfg is the dict returned by setup_peribus_config() — collected
    before any service starts so prompts don't race with debug output.
    """
    print(f"\n── Creating mux as '{name}' ──")

    nuke_stale_mounts()

    # 1. Start LLMFS
    print(f"\n  Starting LLMFS on port {LLMFS_PORT}...")
    llmfs_cmd = [
        sys.executable, "-m", "llmfs.main",
        "--port", str(LLMFS_PORT),
    ]
    if auth_token:
        llmfs_cmd.extend(["--auth-token", auth_token])
    llmfs = subprocess.Popen(llmfs_cmd, env=_env, start_new_session=True)
    processes.append(("llmfs", llmfs))

    if not wait_for_port(LLMFS_PORT):
        print("  ✗ LLMFS failed to start")
        return False
    print(f"  ✓ LLMFS running on port {LLMFS_PORT}")

    # 2. Start Rio
    print(f"\n  Starting Rio on port {RIO_PORT}...")
    rio_cmd = [
        sys.executable, "-m", "rio.main",
        "--port", str(RIO_PORT),
        "--workspace", name,
        "--mux-mount", MOUNT_BASE,
        "--fullscreen",
    ]
    if auth_token:
        rio_cmd.extend(["--auth-token", auth_token])
    rio = subprocess.Popen(rio_cmd, env=_env, start_new_session=True)
    processes.append(("rio", rio))

    if not wait_for_port(RIO_PORT):
        print("  ✗ Rio failed to start")
        return False
    print(f"  ✓ Rio running on port {RIO_PORT}")

    # 2b. Start peribusd unless the user opted out entirely. The mux exposes
    # it as /n/peribus alongside llm/ and the workspace.
    peribus_ok = False
    scope = peribus_cfg.get("scope", "lan")
    if scope == "off":
        print(f"\n  Peribus: disabled (per user choice)")
    else:
        print(f"\n  Starting peribusd on ports {PERIBUS_WIRE_PORT}/{PERIBUS_NINEP_PORT}...")
        peribus_cmd = [
            sys.executable, "-m", "peribus",
            "--wire-port",  str(PERIBUS_WIRE_PORT),
            "--ninep-port", str(PERIBUS_NINEP_PORT),
            # No --mount: the mux exposes us. mDNS still on for LAN peers.
        ]
        if scope == "public":
            peribus_cmd.extend(["--kad-port", str(PERIBUS_KAD_PORT)])
            for spec in peribus_cfg.get("bootstrap_peers", []):
                peribus_cmd.extend(["--dht-bootstrap", spec])
            if peribus_cfg.get("upnp"):
                peribus_cmd.append("--upnp")
        else:
            # LAN-only: keep mDNS, drop the DHT (no UDP listener, no public
            # publish, no chance of accidentally exposing yourself).
            peribus_cmd.append("--no-dht")

        peribus = subprocess.Popen(peribus_cmd, env=_env, start_new_session=True)
        processes.append(("peribusd", peribus))

        if wait_for_port(PERIBUS_NINEP_PORT, timeout=5):
            peribus_ok = True
            print(f"  ✓ peribusd running on port {PERIBUS_NINEP_PORT}")
        else:
            print(f"  ⚠ peribusd did not respond on port {PERIBUS_NINEP_PORT} — continuing without it")

    # 3. Start riomux.
    #
    # Auth flows in two directions:
    #
    #   • Mux side  (--auth-token <tok>): clients connecting to the mux
    #     (i.e. our local FUSE mount, and any remote peer) must present
    #     the token.
    #
    #   • Backend side (per-backend ":<tok>" in --backend specs): the
    #     mux itself Tauth's against the llmfs/rio backends, because
    #     they were started with --auth-token above. Peribus is left
    #     unauthed; its --backend spec has no trailing token.
    print(f"\n  Starting riomux on port {MUX_PORT}...")
    print(f"    Backend: {name} → 127.0.0.1:{RIO_PORT}"
          f"{' (auth)' if auth_token else ''}")
    print(f"    Backend: llm → 127.0.0.1:{LLMFS_PORT}"
          f"{' (auth)' if auth_token else ''}")
    if peribus_ok:
        print(f"    Backend: peribus → 127.0.0.1:{PERIBUS_NINEP_PORT}")

    # Build backend specs. The colon-form host:port:token is parsed by
    # riomux/__main__.py — anything past the last colon, if non-numeric,
    # is treated as the token. Empty token → plain host:port.
    def _backend_spec(label, host, port, token):
        if token:
            return f"{label}={host}:{port}:{token}"
        return f"{label}={host}:{port}"

    mux_cmd = [
        sys.executable, "-m", "riomux",
        "--port", str(MUX_PORT),
        "--backend", _backend_spec(name, "127.0.0.1", RIO_PORT, auth_token),
        "--backend", _backend_spec("llm", "127.0.0.1", LLMFS_PORT, auth_token),
    ]
    if peribus_ok:
        # Peribus has no auth; pass it bare.
        mux_cmd.extend([
            "--backend",
            _backend_spec("peribus", "127.0.0.1", PERIBUS_NINEP_PORT, None),
        ])

    if auth_token:
        mux_cmd.extend(["--auth-token", auth_token])
        print(f"    Auth: enabled (mux clients + backends)")
    else:
        print(f"    Auth: disabled")

    mux = subprocess.Popen(mux_cmd, env=_env, start_new_session=True)
    processes.append(("riomux", mux))

    if not wait_for_port(MUX_PORT):
        print("  ✗ riomux failed to start")
        return False
    print(f"  ✓ riomux running on port {MUX_PORT}")

    # 4. Mount via ninepfuse (with auth)
    print(f"\n  Mounting mux at {MOUNT_BASE}...")
    fuse = mount_ninepfuse(
        "127.0.0.1", MUX_PORT, MOUNT_BASE,
        auth_token=auth_token,
    )
    if fuse:
        processes.append(("ninepfuse", fuse))
        mounts.append(MOUNT_BASE)
    else:
        # Fallback to 9pfuse if ninepfuse fails and no auth
        if not auth_token and check_binary("9pfuse"):
            print("  Falling back to 9pfuse (no auth)...")
            fuse = mount_9pfuse("127.0.0.1", MUX_PORT, MOUNT_BASE)
            if fuse:
                processes.append(("9pfuse", fuse))
                mounts.append(MOUNT_BASE)
        if not fuse:
            print("  ⚠ Mount failed — servers are running but not mounted.")
            print(f"    Mount manually:")
            if auth_token:
                print(f"    python ninepfuse.py 'tcp!127.0.0.1!{MUX_PORT}' {MOUNT_BASE} -t <token>")
            else:
                print(f"    9pfuse 127.0.0.1:{MUX_PORT} {MOUNT_BASE}")

    return True


# ── Mode: Connect to existing mux ───────────────────────────────

def mode_connect_mux(processes, mounts):
    """Connect to an existing riomux and mount it."""
    print("\n── Connect to existing mux ──")

    host = prompt("Mux host", "192.168.1.10")
    port = int(prompt("Mux port", str(MUX_PORT)))

    # Auth
    auth_token = prompt("Auth token (empty for none)", "")
    if not auth_token:
        auth_token = None

    # Mount via ninepfuse
    fuse = mount_ninepfuse(host, port, MOUNT_BASE, auth_token=auth_token)
    if fuse:
        processes.append(("ninepfuse-remote", fuse))
        mounts.append(MOUNT_BASE)
    else:
        # Fallback
        if not auth_token and check_binary("9pfuse"):
            print("  Trying 9pfuse fallback (no auth)...")
            fuse = mount_9pfuse(host, port, MOUNT_BASE)
            if fuse:
                processes.append(("9pfuse-remote", fuse))
                mounts.append(MOUNT_BASE)

        if not fuse:
            print("  ✗ Could not mount remote mux.")
            return False

    # List what's available
    print(f"\n  Available on mux:")
    try:
        for entry in sorted(os.listdir(MOUNT_BASE)):
            print(f"    /n/{entry}/")
    except OSError as e:
        print(f"    (could not list: {e})")

    return True


# ── Mode: Standalone (no mux) ───────────────────────────────────

def mode_standalone(processes, mounts):
    """Start llmfs and rio without muxing, mount each separately."""
    print("\n── Standalone mode (separate mounts) ──")

    # Start LLMFS
    print(f"\n  Starting LLMFS on port {LLMFS_PORT}...")
    llmfs = subprocess.Popen([
        sys.executable, "-m", "llmfs.main",
        "--port", str(LLMFS_PORT),
    ], env=_env, start_new_session=True)
    processes.append(("llmfs", llmfs))

    if not wait_for_port(LLMFS_PORT):
        print("  ✗ LLMFS failed to start")
        return False
    print(f"  ✓ LLMFS running on port {LLMFS_PORT}")

    # Start Rio
    print(f"\n  Starting Rio on port {RIO_PORT}...")
    rio = subprocess.Popen([
        sys.executable, "-m", "rio.main",
        "--port", str(RIO_PORT),
    ], env=_env, start_new_session=True)
    processes.append(("rio", rio))

    if not wait_for_port(RIO_PORT):
        print("  ✗ Rio failed to start")
        return False
    print(f"  ✓ Rio running on port {RIO_PORT}")

    # Mount each separately — no auth for standalone backends
    ninepfuse = find_ninepfuse()
    has_9pfuse = check_binary("9pfuse")

    if ninepfuse or has_9pfuse:
        ensure_dir("/n/llm")
        fuse_llm = mount_ninepfuse("127.0.0.1", LLMFS_PORT, "/n/llm")
        if not fuse_llm and has_9pfuse:
            fuse_llm = mount_9pfuse("127.0.0.1", LLMFS_PORT, "/n/llm")
        if fuse_llm:
            processes.append(("fuse-llm", fuse_llm))
            mounts.append("/n/llm")

        ensure_dir("/n/rio")
        fuse_rio = mount_ninepfuse("127.0.0.1", RIO_PORT, "/n/rio")
        if not fuse_rio and has_9pfuse:
            fuse_rio = mount_9pfuse("127.0.0.1", RIO_PORT, "/n/rio")
        if fuse_rio:
            processes.append(("fuse-rio", fuse_rio))
            mounts.append("/n/rio")
    else:
        print(f"\n  No FUSE client found — mount manually:")
        print(f"    python ninepfuse.py 'tcp!127.0.0.1!{LLMFS_PORT}' /n/llm")
        print(f"    python ninepfuse.py 'tcp!127.0.0.1!{RIO_PORT}' /n/rio")

    return True


# ─────────────────────────────────────────────────────────────────
# ── Cleanup (rogue-loop killer, parallel SIGTERM, lazy unmount) ──
# ─────────────────────────────────────────────────────────────────

# ── Cleanup ──────────────────────────────────────────────────────

# Patterns of "rogue" processes that aren't tracked in `processes[]` but
# should die when we shut down. These are typically routing loops the user
# (or an LLM agent) started detached from us, e.g. by writing
#     while true; do cat /n/llm/claude/output > /n/<name>/scene/parse; done
# to /tmp/llm and running it. The example in this script's docstring is
# itself the recipe for this — see the `while true` block in main().
#
# We match by command-line substring with `pkill -f`. Keep these patterns
# specific enough not to nuke unrelated user shells.
_ROGUE_PATTERNS = (
    "/tmp/llm",
    "/tmp/route",
    "/tmp/scene",
    "cat /n/llm/.*/output",
    "cat /n/.*/scene/parse",
)


def _kill_rogue_loops():
    """Kill routing loops that escaped our process tree.

    These are the bash processes the user sees in `ps` after Ctrl-C
    eating ~30% CPU each — typically `bash /tmp/llm` running a tight
    `while true; do cat ... ; done` with no sleep.
    """
    killed_any = False
    for pattern in _ROGUE_PATTERNS:
        # -f matches against full command line, -9 to be sure (these are
        # not our processes so SIGTERM-then-wait-then-SIGKILL is overkill).
        result = subprocess.run(
            ["pkill", "-9", "-f", pattern],
            capture_output=True,
        )
        # pkill exits 0 if it killed something, 1 if not.
        if result.returncode == 0:
            killed_any = True
    if killed_any:
        print("  Killed rogue routing loop(s)")


def _killpg_safe(proc, sig):
    """Send `sig` to `proc`'s entire process group, falling back to the
    process itself if it isn't a session leader.

    We launch every child with start_new_session=True so the whole
    subtree (e.g. a Python server that forked a helper) goes down at
    once. If that didn't take for some reason, fall back to plain kill.
    """
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


def cleanup(processes, mounts):
    """Stop all processes and unmount.

    Old behaviour: unmount first, then loop terminating processes one at
    a time with proc.wait(timeout=5) between each. With 5 children and
    any of them blocked on a 9P read tied to the FUSE mount, this stacked
    into a multi-minute hang.

    New behaviour:
      0. Kill orphan routing loops (`bash /tmp/llm` etc.) — these are
         outside our process tree and would otherwise survive.
      1. Send SIGTERM to every child's process group, in parallel.
      2. Lazy-unmount FUSE (`fusermount -uz`) so blocked reads don't
         hold us hostage; the kernel detaches now and cleans up later.
      3. Wait for all children with a SHARED 3-second deadline, not
         5 seconds per child.
      4. SIGKILL any stragglers.
    """
    print("\n── Shutting down ──")

    # 0. Rogue loops first — before anything else, because they keep the
    #    9P pipes hot and we want them to stop generating traffic.
    _kill_rogue_loops()

    # 1. Send SIGTERM to every child IN PARALLEL.
    #    No waiting between sends — we want them all to start dying
    #    concurrently so their I/O winds down together.
    alive = []
    for name, proc in reversed(processes):
        if proc.poll() is None:
            print(f"  Stopping {name} (pid {proc.pid})...")
            _killpg_safe(proc, signal.SIGTERM)
            alive.append((name, proc))

    # 2. Lazy-unmount FUSE.
    #    Plain `fusermount -u` blocks until in-flight reads return, which
    #    they won't if our 9P server is mid-shutdown. `-z` (lazy) detaches
    #    the mount immediately; the kernel cleans up references as fids
    #    are released. Combined with the SIGTERMs above, the servers
    #    notice their sockets close and exit cleanly.
    for mp in reversed(mounts):
        print(f"  Unmounting {mp}...")
        subprocess.run(["fusermount", "-uz", mp], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 3. Wait with a SHARED deadline.
    #    Total budget is 3s for everyone, not 5s each. In practice every
    #    well-behaved child exits within a few hundred ms once its
    #    sockets close.
    deadline = time.time() + 3.0
    for name, proc in alive:
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass  # Handled in step 4.

    # 4. SIGKILL the stubborn.
    for name, proc in alive:
        if proc.poll() is None:
            print(f"  Force-killing {name} (pid {proc.pid})...")
            _killpg_safe(proc, signal.SIGKILL)
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass

    # Belt and braces: one more pass at orphan loops in case a child
    # respawned one on its way down.
    _kill_rogue_loops()

    print("  ✓ All stopped.")


# ─────────────────────────────────────────────────────────────────
# ── Peribus config & summary helpers ─────────────────────────────
# ─────────────────────────────────────────────────────────────────

# ── Main ─────────────────────────────────────────────────────────

def setup_peribus_config():
    """
    Collect all peribus-related decisions up front, before any service starts.

    Done this way deliberately: starting llmfs/rio first means their startup
    chatter races with our prompts, and the user sees stack traces from a
    failed embedder load while trying to type a bootstrap URL. Answer
    everything first, then bring services up in a single quiet sequence.

    Returns a dict the caller hands to mode_create_mux. Keys:
        scope:           "off" | "lan" | "public"
        upnp:            bool        (only meaningful when scope == "public")
        bootstrap_peers: list[str]   (NODEID@host:port, may be empty)
    """
    print("\n── Peribus discovery ──")
    print("  Peribus is the social/feed layer that lives at /n/peribus.")
    print("  Choose how it connects to other peers.")

    scope = prompt_choice(
        "Network scope:",
        [
            ("off",
             "Off  — don't run peribus at all"),
            ("lan",
             "LAN  — find peers on the local network only (mDNS, no DHT)"),
            ("public",
             "Public  — also discover peers on the open internet (DHT)"),
        ],
        default="lan",
    )

    cfg = {"scope": scope, "upnp": False, "bootstrap_peers": []}

    if scope != "public":
        return cfg

    # UPnP: ask the router to open the port for us. Big quality-of-life
    # win for users behind NAT, which is most of them. Off by default
    # because phoning the router is something the user should opt into.
    cfg["upnp"] = prompt_yes_no(
        "Try UPnP to open your router automatically?", default=True,
    )
    if not cfg["upnp"]:
        print(f"  ⚠ Without UPnP or manual port-forward, you'll be reachable")
        print(f"    only for replies to your own queries. You can still find")
        print(f"    peers but peers can't initiate contact with you.")

    # Bootstrap peers — optional. Empty list means standalone bootstrap node.
    print(f"\n  Bootstrap peers connect you to an existing peribus network.")
    print(f"  Format: NODEID@host:port (one per line, blank to finish).")
    print(f"  Skip if you're the first node — others can bootstrap from you.")
    while True:
        spec = prompt("    bootstrap peer", default="").strip()
        if not spec:
            break
        if "@" not in spec or ":" not in spec:
            print(f"      ✗ bad format, expected NODEID@host:port")
            continue
        cfg["bootstrap_peers"].append(spec)
    if cfg["bootstrap_peers"]:
        print(f"  ✓ {len(cfg['bootstrap_peers'])} bootstrap peer(s) configured")
    else:
        print(f"  ✓ No bootstrap peers — you'll be a standalone DHT node")
        print(f"    (others can join by connecting to your bootstrap URL)")

    return cfg

def _get_local_ip():
    """Best-effort local IP for remote connection hints."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "YOUR_IP"


def _peribus_nodeid():
    """Read peribusd's NodeID from its identity dir. Returns None if peribusd
    has not yet written it (e.g. it's still starting, or never started)."""
    nodeid_path = os.path.expanduser("~/.peribus/identity/nodeid")
    try:
        with open(nodeid_path) as f:
            nodeid = f.read().strip()
    except OSError:
        return None
    return nodeid or None


def _peribus_bootstrap_url(processes):
    """
    Build the NODEID@host:port string for our local peribusd, if it's running
    with the DHT. Returns None if peribusd isn't up or the DHT wasn't enabled.

    We detect "DHT enabled" by checking whether the peribusd Popen was launched
    with --kad-port (start.py only passes it when the user opted in). The
    NodeID is read from the identity dir, which the daemon writes on first
    start. The host is our LAN address; for internet reach the user needs to
    use their public IP.
    """
    # Find peribusd among the running processes.
    peribus_proc = None
    for name, proc in processes:
        if name == "peribusd" and proc.poll() is None:
            peribus_proc = proc
            break
    if peribus_proc is None:
        return None
    cmdline = peribus_proc.args if hasattr(peribus_proc, "args") else []
    if "--kad-port" not in cmdline:
        return None  # DHT was disabled

    nodeid = _peribus_nodeid()
    if not nodeid:
        return None
    return f"{nodeid}@{_get_local_ip()}:{PERIBUS_KAD_PORT}"



# ─────────────────────────────────────────────────────────────────
# ── CLI entry point (interactive prompts on stdin) ───────────────
# ─────────────────────────────────────────────────────────────────


def run_cli():
    """
    Interactive CLI flow — exactly the old start_peribus.py main()
    behaviour: ask the user via stdin, drive the mode functions,
    print a summary, wait until Ctrl-C, then clean up.
    """
    processes = []  # [(name, Popen), ...]
    mounts = []     # [mountpoint, ...]

    # Register cleanup
    def signal_handler(sig, frame):
        cleanup(processes, mounts)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 56)
    print("  riomux — 9P Multiplexer Start")
    print("=" * 56)

    # Check dependencies
    ninepfuse = find_ninepfuse()
    has_9pfuse = check_binary("9pfuse")

    if not ninepfuse and not has_9pfuse:
        print("\n  ⚠ No FUSE client found.")
        print("    ninepfuse.py not found in project.")
        print("    9pfuse not found on PATH.")
        print("    Mounting will be unavailable.\n")
    elif ninepfuse:
        print(f"\n  Using ninepfuse: {ninepfuse}")
    else:
        print(f"\n  Using 9pfuse (no auth support)")

    # Ensure /n
    ensure_mount_base()

    # Choose mode
    mode = prompt_choice(
        "How do you want to start?",
        [
            ("create",     "Create a new mux (start llmfs + rio + riomux, mount /n)"),
            ("connect",    "Connect to an existing mux (mount remote riomux)"),
            ("standalone", "Standalone (start llmfs + rio, mount separately under /n/)"),
        ],
        default="create",
    )

    ok = False
    auth_token = None  # populated by create mode for the remote-instructions block

    if mode == "create":
        name = prompt("Workspace name (e.g. your name)", getpass.getuser())
        name = name.strip().lower().replace(" ", "_")
        if not name:
            name = "default"

        # Auth setup
        print("\n── Authentication ──")
        auth_token = setup_auth_token()

        # Peribus setup — collect ALL decisions before we start any service.
        # Otherwise log output from llmfs/rio races with the bootstrap prompt
        # and the user has to type a URL through scrolling debug spam.
        peribus_cfg = setup_peribus_config()

        ok = mode_create_mux(name, auth_token, peribus_cfg, processes, mounts)

    elif mode == "connect":
        ok = mode_connect_mux(processes, mounts)

    elif mode == "standalone":
        ok = mode_standalone(processes, mounts)

    if not ok:
        print("\n  ⚠ Some services failed to start. Check output above.")

    # Summary
    running = [(n, p) for n, p in processes if p.poll() is None]
    if running:
        print(f"\n{'=' * 56}")
        print(f"  Running services:")
        for nm, proc in running:
            print(f"    • {nm:16s}  pid {proc.pid}")
        if mounts:
            print(f"\n  Mounts:")
            for mp in mounts:
                print(f"    • {mp}")
                try:
                    for entry in sorted(os.listdir(mp)):
                        full = os.path.join(mp, entry)
                        if os.path.isdir(full):
                            print(f"      └── {entry}/")
                except OSError:
                    pass

        # Auth was captured up-front (in mode == "create") rather than dug
        # out of proc.args, which was fragile and silently failed when the
        # token wasn't included in the cmdline.
        if auth_token:
            local_ip = _get_local_ip()
            print(f"\n  Remote connection (Linux/Mac):")
            print(f"    python ninepfuse.py 'tcp!{local_ip}!{MUX_PORT}' /n -t {auth_token}")
            print(f"\n  Remote connection (Plan 9):")
            print(f"    echo 'key proto=pass dom=ninep user=$user !password={auth_token}' >/mnt/factotum/ctl")
            print(f"    mount -a /net/tcp!{local_ip}!{MUX_PORT} /n/mux")

        # If peribusd is running with the DHT, surface our bootstrap-self URL
        # so the user has something concrete to share with relatives.
        peribus_url = _peribus_bootstrap_url(processes)
        peribus_nodeid = _peribus_nodeid()
        if peribus_url and peribus_nodeid:
            print(f"\n  Peribus DHT is up.")
            print(f"    nodeid:    {peribus_nodeid}")
            print(f"    bootstrap: {peribus_url}")
            print()
            print(f"  To connect with another machine that's also running peribus:")
            print(f"    1. Both of you read your bootstrap URL:")
            print(f"         cat /n/peribus/bootstrap")
            print(f"    2. Either of you connects to the other:")
            print(f"         echo 'connect <their-bootstrap-url>' > /n/peribus/ctl")
            print(f"    3. The DHT does the rest. Verify with:")
            print(f"         cat /n/peribus/ctl")
            print()
            print(f"  Note: bootstrap URL above uses your LAN IP. For internet")
            print(f"  reach, replace it with your public IP and ensure UDP port")
            print(f"  {PERIBUS_KAD_PORT} is forwarded.")

        print(f"\n  Example routing (streaming, blocking-safe):")
        print(f"    while true; do")
        print(f"      cat /n/llm/claude/output > /n/*/scene/parse")
        print(f"    done")
        print(f"\n  Press Ctrl+C to stop everything.")
        print(f"{'=' * 56}\n")

        # Wait for processes
        try:
            while True:
                for nm, proc in processes:
                    if proc.poll() is not None:
                        rc = proc.returncode
                        if nm == "rio":
                            print(f"\n  Rio exited (code {rc}) — shutting down mux stack...")
                            raise KeyboardInterrupt
                        elif rc != 0 and rc != -15:
                            print(f"  ⚠ {nm} exited with code {rc}")
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        print("\n  No services running.")

    cleanup(processes, mounts)


# ─────────────────────────────────────────────────────────────────
# ── GUI (PySide6) ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
#
# Defined only when PySide6 imported successfully at top of file.
# Otherwise the GUI symbols don't exist and run_gui() refuses to run
# with a clear message.

if GUI_AVAILABLE:

    # Helper aliases — the GUI was originally written against its own
    # _generate_token / _local_ip / _default_workspace_name; after the
    # merge those map onto the existing CLI helpers.
    _generate_token         = generate_token
    _local_ip               = _get_local_ip
    def _default_workspace_name() -> str:
        return getpass.getuser().lower().replace(" ", "_") or "default"

    # ── Palette (matches theme.PAPER) ────────────────────────────────────
    
    CARD_BG        = "rgb(250, 247, 240)"          # off-white paper
    INK            = "rgb(26, 26, 26)"             # near-black
    INK_MUTED      = "rgb(120, 115, 105)"
    HAIRLINE       = "rgb(42, 42, 42)"
    HAIRLINE_SOFT  = "rgba(42, 42, 42, 80)"
    SAGE_SEL       = "rgba(180, 200, 180, 140)"
    AMBER          = "rgb(212, 142, 60)"           # warning
    RED_INK        = "rgb(170, 50, 50)"            # double warning
    GREEN_OK       = "rgb(80, 140, 80)"
    
    UI_FAMILY      = "'IBM Plex Sans', 'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
    MONO_FAMILY    = "'IBM Plex Mono', 'JetBrains Mono', 'Consolas', 'Menlo', monospace"
    
    
    # Shadow palette — the drop-shadow doubles as a status indicator that
    # animates between these tones.
    SHADOW_IDLE    = QColor(15, 15, 15, 140)       # near-black, neutral resting
    SHADOW_DANGER  = QColor(190, 50, 50, 160)      # red — public DHT selected
    SHADOW_RUNNING = QColor(212, 142, 60, 150)     # warm amber — mux is up
    
    
    QSS = f"""
    /* The paper card — the entire window. Frameless, translucent OS
       surface so the drop-shadow projects directly onto the desktop. */
    QFrame#card {{
        background-color: {CARD_BG};
        border: 1px solid {HAIRLINE};
        border-radius: 2px;
    }}
    
    /* Header strip inside the card */
    QLabel#title {{
        font-family: {MONO_FAMILY};
        font-size: 13px;
        color: {INK};
        background: transparent;
    }}
    QLabel#muxId {{
        font-family: {MONO_FAMILY};
        font-size: 13px;
        color: {INK};
        background: transparent;
    }}
    QLabel#breadcrumb {{
        font-family: {MONO_FAMILY};
        font-size: 12px;
        color: rgb(120, 145, 175);
        background: transparent;
    }}
    
    /* Section labels — small, lowercased, monospace, like "colors" */
    QLabel#sectionLabel {{
        font-family: {MONO_FAMILY};
        font-size: 11px;
        color: {INK_MUTED};
        background: transparent;
        padding-right: 6px;
    }}
    
    QLabel#bodyLabel {{
        font-family: {MONO_FAMILY};
        font-size: 12px;
        color: {INK};
        background: transparent;
    }}
    QLabel#muted {{
        font-family: {MONO_FAMILY};
        font-size: 11px;
        color: {INK_MUTED};
        background: transparent;
    }}
    QLabel#hint {{
        font-family: {MONO_FAMILY};
        font-size: 11px;
        color: {INK_MUTED};
        background: transparent;
    }}
    
    /* Warnings */
    QLabel#warn {{
        font-family: {MONO_FAMILY};
        font-size: 11px;
        color: {AMBER};
        background: transparent;
        padding: 6px 8px;
        border: 1px solid {AMBER};
        border-radius: 2px;
    }}
    QLabel#warn2 {{
        font-family: {MONO_FAMILY};
        font-size: 11px;
        color: {RED_INK};
        background: rgba(170, 50, 50, 18);
        border: 1px solid {RED_INK};
        border-radius: 2px;
        padding: 6px 8px;
        font-weight: 600;
    }}
    QLabel#ok {{
        font-family: {MONO_FAMILY};
        font-size: 11px;
        color: {GREEN_OK};
        background: transparent;
        padding: 4px 0;
    }}
    
    /* Pill buttons — match the color-picker chips in the screenshot */
    QPushButton[pill="true"] {{
        background-color: {CARD_BG};
        border: 1px solid {HAIRLINE_SOFT};
        border-radius: 12px;
        padding: 3px 12px 3px 10px;
        color: {INK};
        font-family: {UI_FAMILY};
        font-size: 11px;
    }}
    QPushButton[pill="true"]:hover {{
        border: 1px solid {HAIRLINE};
    }}
    QPushButton[pill="true"][selected="true"] {{
        border: 1.5px solid {AMBER};
    }}
    
    /* Text inputs — underlined editorial style */
    QLineEdit, QSpinBox {{
        background-color: {CARD_BG};
        border: none;
        border-bottom: 1px solid {HAIRLINE_SOFT};
        padding: 4px 2px;
        color: {INK};
        font-family: {MONO_FAMILY};
        font-size: 12px;
        selection-background-color: {SAGE_SEL};
    }}
    QLineEdit:focus, QSpinBox:focus {{
        border-bottom: 1px solid {INK};
    }}
    QLineEdit:disabled, QSpinBox:disabled {{
        color: rgba(26, 26, 26, 80);
    }}
    
    QPlainTextEdit {{
        background-color: {CARD_BG};
        border: 1px solid {HAIRLINE_SOFT};
        border-radius: 2px;
        padding: 6px 8px;
        color: {INK};
        font-family: {MONO_FAMILY};
        font-size: 11px;
        selection-background-color: {SAGE_SEL};
    }}
    QPlainTextEdit:focus {{
        border: 1px solid {INK};
    }}
    QPlainTextEdit#log {{
        background-color: rgb(252, 250, 244);
        border: 1px solid {HAIRLINE_SOFT};
        color: {INK};
        font-family: {MONO_FAMILY};
        font-size: 11px;
    }}
    
    /* Combobox — minimal underlined */
    QComboBox {{
        background-color: {CARD_BG};
        border: none;
        border-bottom: 1px solid {HAIRLINE_SOFT};
        padding: 4px 22px 4px 4px;
        color: {INK};
        font-family: {MONO_FAMILY};
        font-size: 12px;
    }}
    QComboBox:focus {{
        border-bottom: 1px solid {INK};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 18px;
    }}
    QComboBox::down-arrow {{
        image: none;
        width: 0; height: 0;
        border-left:  4px solid transparent;
        border-right: 4px solid transparent;
        border-top:   4px solid {INK_MUTED};
        margin-right: 6px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {CARD_BG};
        border: 1px solid {HAIRLINE};
        color: {INK};
        selection-background-color: {INK};
        selection-color: {CARD_BG};
        font-family: {MONO_FAMILY};
        outline: 0;
    }}
    
    /* Checkboxes */
    QCheckBox {{
        font-family: {MONO_FAMILY};
        font-size: 12px;
        color: {INK};
        background: transparent;
        spacing: 6px;
        padding: 2px 0;
    }}
    QCheckBox::indicator {{
        width: 12px;
        height: 12px;
    }}
    QCheckBox::indicator:unchecked {{
        background-color: {CARD_BG};
        border: 1px solid {HAIRLINE_SOFT};
        border-radius: 2px;
    }}
    QCheckBox::indicator:checked {{
        background-color: {INK};
        border: 1px solid {INK};
        border-radius: 2px;
    }}
    
    /* Generic ghost / subtle buttons */
    QPushButton#ghost {{
        background-color: transparent;
        border: none;
        color: {INK_MUTED};
        font-family: {MONO_FAMILY};
        font-size: 11px;
        padding: 2px 4px;
        text-align: left;
    }}
    QPushButton#ghost:hover {{
        color: {INK};
    }}
    
    QPushButton#subtle {{
        background-color: {CARD_BG};
        border: 1px solid {HAIRLINE_SOFT};
        border-radius: 2px;
        padding: 4px 10px;
        color: {INK};
        font-family: {MONO_FAMILY};
        font-size: 11px;
    }}
    QPushButton#subtle:hover {{
        border: 1px solid {INK};
    }}
    
    /* Primary Start button — flat black, paper-theme primary */
    QPushButton#primary {{
        background-color: {INK};
        border: none;
        color: {CARD_BG};
        font-family: {UI_FAMILY};
        font-size: 12px;
        font-weight: 600;
        padding: 7px 18px;
        border-radius: 2px;
    }}
    QPushButton#primary:hover {{
        background-color: rgb(50, 50, 50);
    }}
    QPushButton#primary:pressed {{
        background-color: rgb(10, 10, 10);
    }}
    QPushButton#primary:disabled {{
        background-color: rgba(26, 26, 26, 80);
        color: {CARD_BG};
    }}
    
    QPushButton#danger {{
        background-color: {CARD_BG};
        border: 1px solid {RED_INK};
        color: {RED_INK};
        font-family: {UI_FAMILY};
        font-size: 12px;
        padding: 7px 14px;
        border-radius: 2px;
    }}
    QPushButton#danger:hover {{
        background-color: {RED_INK};
        color: {CARD_BG};
    }}
    QPushButton#danger:disabled {{
        border: 1px solid rgba(170, 50, 50, 80);
        color: rgba(170, 50, 50, 100);
    }}
    
    /* Close button in the card's top-right corner */
    QPushButton#closeBtn {{
        background-color: transparent;
        border: none;
        color: {INK_MUTED};
        font-family: {UI_FAMILY};
        font-size: 13px;
        padding: 0 4px;
    }}
    QPushButton#closeBtn:hover {{
        color: {INK};
    }}
    
    /* Subtle separator rule */
    QFrame#rule {{
        background-color: {HAIRLINE_SOFT};
        max-height: 1px;
        min-height: 1px;
        border: none;
    }}
    
    /* Scrollbars — paper style: barely there */
    QScrollArea, QScrollArea > QWidget > QWidget {{
        background: transparent;
        border: none;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: rgba(42, 42, 42, 60);
        border-radius: 3px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: rgba(42, 42, 42, 120);
    }}
    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{
        background: transparent;
    }}
    """
    

    # ── Worker thread driving start_peribus.py ───────────────────────────
    
    class _StreamRedirector(QObject):
        line = Signal(str)
        def write(self, s):
            if s:
                self.line.emit(str(s))
        def flush(self):
            pass
    
    
    class StarterWorker(QObject):
        log = Signal(str)
        finished = Signal(bool, str)
    
        def __init__(self, config: dict):
            super().__init__()
            self.config = config
            self._processes = []
            self._mounts = []
    
        @Slot()
        def run(self):
            redirector = _StreamRedirector()
            redirector.line.connect(self.log.emit)
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = redirector
            sys.stderr = redirector
    
            ok = False
            summary = ""
            try:
                # After the merge, ensure_mount_base / mode_create_mux /
                # mode_connect_mux / mode_standalone all live in this same
                # module. We grab a reference to it via sys.modules so the
                # connect-mode monkey-patch of `sp.prompt = _fake_prompt`
                # still works the same way it did when this code lived in
                # start_gui.py and imported start_peribus.py separately.
                sp = sys.modules[__name__]
    
                cfg = self.config
                mode = cfg["mode"]
                sp.ensure_mount_base()
    
                if mode == "create":
                    name = cfg["workspace"] or _default_workspace_name()
                    auth_token = cfg["auth_token"] if cfg["auth_enabled"] else None
                    peribus_cfg = {
                        "scope":           cfg["peribus_scope"],
                        "upnp":            cfg["peribus_upnp"],
                        "bootstrap_peers": cfg["peribus_bootstrap"],
                    }
                    ok = sp.mode_create_mux(
                        name, auth_token, peribus_cfg, self._processes, self._mounts,
                    )
                    if ok:
                        summary = self._summary_create(name, auth_token)
    
                elif mode == "connect":
                    answers = iter([
                        cfg["connect_host"],
                        str(cfg["connect_port"]),
                        cfg["connect_token"] or "",
                    ])
                    def _fake_prompt(msg, default=None):
                        try:
                            v = next(answers)
                            return v if v else (default or "")
                        except StopIteration:
                            return default or ""
                    real_prompt = sp.prompt
                    sp.prompt = _fake_prompt
                    try:
                        ok = sp.mode_connect_mux(self._processes, self._mounts)
                    finally:
                        sp.prompt = real_prompt
                    if ok:
                        summary = f"connected to {cfg['connect_host']}:{cfg['connect_port']} → /n"
    
                elif mode == "standalone":
                    ok = sp.mode_standalone(self._processes, self._mounts)
                    if ok:
                        summary = "standalone: /n/llm + /n/rio"
    
            except Exception as e:
                self.log.emit(f"\n  ✗ Error: {e}\n")
                ok = False
                summary = f"failed: {e}"
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
    
            self.finished.emit(ok, summary)
    
        def _summary_create(self, name, auth_token):
            ip = _local_ip()
            lines = [
                f"mux running on {ip}:{MUX_PORT}",
                f"  workspace: /n/{name}",
                f"  llmfs:     /n/llm",
            ]
            if auth_token:
                lines.append(f"  auth:      enabled  (token: {auth_token})")
            else:
                lines.append(f"  auth:      DISABLED")
            return "\n".join(lines)
    
        def stop_all(self):
            try:
                sp = sys.modules[__name__]
                sp.cleanup(self._processes, self._mounts)
            except Exception as e:
                self.log.emit(f"\n  ✗ Cleanup error: {e}\n")
    
    
    # ── Pill button — matches the color-chip style ───────────────────────
    
    class ModePill(QPushButton):
        """A pill-shaped toggle with an optional colored dot, like the
        color-picker chips in the screenshot."""
    
        def __init__(self, label: str, dot_color: str | None = None, parent=None):
            super().__init__(parent)
            self._dot = QColor(dot_color) if dot_color else None
            self.setCheckable(True)
            self.setCursor(Qt.PointingHandCursor)
            self.setProperty("pill", True)
            self.setProperty("selected", False)
            if self._dot is not None:
                self.setText("        " + label)   # space for the dot
            else:
                self.setText(label)
            self.setMinimumHeight(24)
    
        def setSelected(self, sel: bool):
            self.setProperty("selected", bool(sel))
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()
    
        def paintEvent(self, ev):
            super().paintEvent(ev)
            if self._dot is None:
                return
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(QColor(42, 42, 42, 160), 1))
            p.setBrush(QBrush(self._dot))
            d = 9
            x = 10
            y = (self.height() - d) // 2
            p.drawEllipse(x, y, d, d)
            p.end()
    
    
    # ── Frameless paper card (draggable) ─────────────────────────────────
    
    class Card(QFrame):
        """The off-white paper panel holding the form. Drag dead card area
        to move the frameless window."""
    
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setObjectName("card")
            self.setFrameShape(QFrame.NoFrame)
            self._drag_pos: QPoint | None = None
    
        def mousePressEvent(self, ev):
            if ev.button() == Qt.LeftButton:
                self._drag_pos = (
                    ev.globalPosition().toPoint()
                    - self.window().frameGeometry().topLeft()
                )
                ev.accept()
            else:
                super().mousePressEvent(ev)
    
        def mouseMoveEvent(self, ev):
            if self._drag_pos is not None and ev.buttons() & Qt.LeftButton:
                self.window().move(ev.globalPosition().toPoint() - self._drag_pos)
                ev.accept()
            else:
                super().mouseMoveEvent(ev)
    
        def mouseReleaseEvent(self, ev):
            self._drag_pos = None
            super().mouseReleaseEvent(ev)
    
    
    # ── Main window (frameless) ──────────────────────────────────────────
    
    class StarterWindow(QWidget):
        def __init__(self):
            super().__init__()
            # Frameless + translucent OS surface, like a /pop'd terminal.
            # The top-level widget paints nothing; the card inside owns the
            # visible surface and its drop-shadow spills onto whatever is
            # behind the window on the desktop.
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WA_NoSystemBackground, True)
            self.setStyleSheet(QSS)
            self.setWindowTitle("riomux")
            self.resize(820, 820)
            # Centering needs the final geometry, so it's done after resize().
            # We do it here as well as on showEvent — some WMs ignore the
            # pre-show move(), but doing it now avoids a visible jump on
            # WMs that do honor it.
            self._center_on_screen()
    
            self._worker_thread: QThread | None = None
            self._worker: StarterWorker | None = None
    
            # A thin transparent margin around the card lets the shadow
            # bleed outside it without being clipped by the window's own
            # geometry — same trick /pop uses for a popped terminal.
            outer = QVBoxLayout(self)
            outer.setContentsMargins(24, 18, 36, 40)   # asymmetric → shadow on lower-right
            outer.setSpacing(0)
    
            self.card = Card(self)
            outer.addWidget(self.card)
    
            # Shadow projects onto the OS desktop. Its colour doubles as a
            # status indicator: black when idle, red when public peribus is
            # selected (live preview of the danger), amber when the mux is
            # actually running.
            self._shadow = QGraphicsDropShadowEffect(self)
            self._shadow.setBlurRadius(50)
            self._shadow.setOffset(14, 18)
            self._shadow.setColor(SHADOW_IDLE)
            self.card.setGraphicsEffect(self._shadow)
    
            # One reusable animation drives every shadow-colour change.
            self._shadow_anim = QVariantAnimation(self)
            self._shadow_anim.setDuration(420)
            self._shadow_anim.setEasingCurve(QEasingCurve.InOutCubic)
            self._shadow_anim.valueChanged.connect(self._on_shadow_anim_step)
    
            card_layout = QVBoxLayout(self.card)
            card_layout.setContentsMargins(28, 22, 22, 22)
            card_layout.setSpacing(14)
    
            # ── Header (Terminal ID style) ──────────────────────────────
            header_row = QHBoxLayout()
            header_row.setSpacing(8)
            header_col = QVBoxLayout()
            header_col.setSpacing(2)
            title_lbl = QLabel("Mux ID:")
            title_lbl.setObjectName("title")
            header_col.addWidget(title_lbl)
            self.mux_id_lbl = QLabel(f"riomux_{secrets.token_hex(4)}")
            self.mux_id_lbl.setObjectName("muxId")
            header_col.addWidget(self.mux_id_lbl)
            crumb = QLabel("/start")
            crumb.setObjectName("breadcrumb")
            header_col.addWidget(crumb)
            header_row.addLayout(header_col, 1)
    
            self.close_btn = QPushButton("✕")
            self.close_btn.setObjectName("closeBtn")
            self.close_btn.setCursor(Qt.PointingHandCursor)
            self.close_btn.setFixedSize(24, 24)
            self.close_btn.clicked.connect(self.close)
            header_row.addWidget(self.close_btn, 0, Qt.AlignTop)
            card_layout.addLayout(header_row)
    
            # ── Mode picker (pills, mirroring the color row) ────────────
            mode_row = QHBoxLayout()
            mode_row.setSpacing(8)
            mode_label = QLabel("mode")
            mode_label.setObjectName("sectionLabel")
            mode_row.addWidget(mode_label)
    
            self.pill_create     = ModePill("Create",     "#7AB890")
            self.pill_connect    = ModePill("Connect",    "#6E9BC2")
            self.pill_standalone = ModePill("Standalone", "#A0A0A0")
            self.mode_pills = [self.pill_create, self.pill_connect, self.pill_standalone]
            for pill in self.mode_pills:
                pill.clicked.connect(lambda _=False, p=pill: self._set_mode(p))
                mode_row.addWidget(pill)
            mode_row.addStretch()
            card_layout.addLayout(mode_row)
    
            rule = QFrame()
            rule.setObjectName("rule")
            card_layout.addWidget(rule)
    
            # ── Inline body (scrollable so the card height stays stable) ─
            scroll = QScrollArea(self.card)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            card_layout.addWidget(scroll, 1)
    
            body_host = QWidget()
            scroll.setWidget(body_host)
            body = QVBoxLayout(body_host)
            body.setContentsMargins(0, 4, 6, 4)
            body.setSpacing(18)
    
            # ── Workspace (create only) ─────────────────────────────────
            self.workspace_panel = QWidget()
            ws = QGridLayout(self.workspace_panel)
            ws.setContentsMargins(0, 0, 0, 0)
            ws.setHorizontalSpacing(10)
            ws.setVerticalSpacing(4)
    
            ws_lbl = QLabel("workspace")
            ws_lbl.setObjectName("sectionLabel")
            ws.addWidget(ws_lbl, 0, 0, Qt.AlignTop)
    
            self.workspace_edit = QLineEdit(_default_workspace_name())
            self.workspace_edit.setPlaceholderText("lowercased, underscores")
            ws.addWidget(self.workspace_edit, 0, 1)
    
            ws_hint = QLabel("will be mounted at /n/<name>/")
            ws_hint.setObjectName("hint")
            ws.addWidget(ws_hint, 1, 1)
            body.addWidget(self.workspace_panel)
    
            # ── Authentication (create only) ────────────────────────────
            self.auth_panel = QWidget()
            au = QGridLayout(self.auth_panel)
            au.setContentsMargins(0, 0, 0, 0)
            au.setHorizontalSpacing(10)
            au.setVerticalSpacing(6)
    
            au_lbl = QLabel("auth")
            au_lbl.setObjectName("sectionLabel")
            au.addWidget(au_lbl, 0, 0, Qt.AlignTop)
    
            self.auth_enabled = QCheckBox("require auth token  (recommended)")
            self.auth_enabled.setChecked(True)
            au.addWidget(self.auth_enabled, 0, 1, 1, 2)
    
            token_lbl = QLabel("token")
            token_lbl.setObjectName("hint")
            au.addWidget(token_lbl, 1, 0, Qt.AlignTop)
            self.auth_token_edit = QLineEdit(_generate_token())
            au.addWidget(self.auth_token_edit, 1, 1)
            self.auth_regen_btn = QPushButton("regenerate")
            self.auth_regen_btn.setObjectName("subtle")
            self.auth_regen_btn.setCursor(Qt.PointingHandCursor)
            self.auth_regen_btn.clicked.connect(
                lambda: self.auth_token_edit.setText(_generate_token())
            )
            au.addWidget(self.auth_regen_btn, 1, 2)
    
            self.auth_warning = QLabel(
                "⚠  without a token, anyone on the network can connect"
            )
            self.auth_warning.setObjectName("warn")
            self.auth_warning.setVisible(False)
            self.auth_warning.setWordWrap(True)
            au.addWidget(self.auth_warning, 2, 1, 1, 2)
            body.addWidget(self.auth_panel)
    
            # ── Peribus / network (create only) ─────────────────────────
            self.peribus_panel = QWidget()
            pe = QGridLayout(self.peribus_panel)
            pe.setContentsMargins(0, 0, 0, 0)
            pe.setHorizontalSpacing(10)
            pe.setVerticalSpacing(6)
    
            pe_lbl = QLabel("network")
            pe_lbl.setObjectName("sectionLabel")
            pe.addWidget(pe_lbl, 0, 0, Qt.AlignTop)
    
            self.peribus_scope = QComboBox()
            self.peribus_scope.addItem("None    — peribus disabled", "off")
            self.peribus_scope.addItem("LAN     — local network peers (mDNS)", "lan")
            self.peribus_scope.addItem("Public  — open internet peers (DHT)", "public")
            self.peribus_scope.setCurrentIndex(0)   # default: None
            pe.addWidget(self.peribus_scope, 0, 1, 1, 2)
    
            from PySide6.QtWidgets import QSizePolicy
            self.peribus_status = QLabel("")
            self.peribus_status.setWordWrap(True)
            # Tell Qt this label's height depends on its width — required for
            # word-wrap to actually grow vertically inside a QGridLayout.
            sp = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
            sp.setHeightForWidth(True)
            self.peribus_status.setSizePolicy(sp)
            pe.addWidget(self.peribus_status, 1, 1, 1, 2)
    
            self.peribus_upnp = QCheckBox("try UPnP to open router automatically")
            self.peribus_upnp.setChecked(True)
            pe.addWidget(self.peribus_upnp, 2, 1, 1, 2)
    
            bs_lbl = QLabel("bootstrap")
            bs_lbl.setObjectName("hint")
            pe.addWidget(bs_lbl, 3, 0, Qt.AlignTop)
            self.peribus_bootstrap = QPlainTextEdit()
            self.peribus_bootstrap.setPlaceholderText(
                "NODEID@host:port, one per line.\nempty if you're the first node."
            )
            self.peribus_bootstrap.setFixedHeight(56)
            pe.addWidget(self.peribus_bootstrap, 3, 1, 1, 2)
            self._pe_bs_lbl = bs_lbl
            body.addWidget(self.peribus_panel)
    
            # ── Connect parameters (connect mode only) ─────────────────
            self.connect_panel = QWidget()
            cn = QGridLayout(self.connect_panel)
            cn.setContentsMargins(0, 0, 0, 0)
            cn.setHorizontalSpacing(10)
            cn.setVerticalSpacing(6)
    
            cn_lbl = QLabel("remote")
            cn_lbl.setObjectName("sectionLabel")
            cn.addWidget(cn_lbl, 0, 0, Qt.AlignTop)
    
            host_lbl = QLabel("host")
            host_lbl.setObjectName("hint")
            cn.addWidget(host_lbl, 0, 1)
            self.connect_host = QLineEdit("192.168.1.10")
            cn.addWidget(self.connect_host, 1, 1)
    
            port_lbl = QLabel("port")
            port_lbl.setObjectName("hint")
            cn.addWidget(port_lbl, 0, 2)
            self.connect_port = QSpinBox()
            self.connect_port.setRange(1, 65535)
            self.connect_port.setValue(MUX_PORT)
            cn.addWidget(self.connect_port, 1, 2)
    
            tk_lbl = QLabel("token")
            tk_lbl.setObjectName("hint")
            cn.addWidget(tk_lbl, 2, 1)
            self.connect_token = QLineEdit("")
            self.connect_token.setPlaceholderText("paste the host's auth token (empty for none)")
            cn.addWidget(self.connect_token, 3, 1, 1, 2)
            body.addWidget(self.connect_panel)
            body.addStretch()
    
            # ── Action bar ─────────────────────────────────────────────
            bar = QHBoxLayout()
            bar.setSpacing(8)
            self.start_btn = QPushButton("Start")
            self.start_btn.setObjectName("primary")
            self.start_btn.setCursor(Qt.PointingHandCursor)
            self.start_btn.clicked.connect(self._on_start)
            bar.addWidget(self.start_btn)
    
            self.stop_btn = QPushButton("Stop")
            self.stop_btn.setObjectName("danger")
            self.stop_btn.setCursor(Qt.PointingHandCursor)
            self.stop_btn.setEnabled(False)
            self.stop_btn.clicked.connect(self._on_stop)
            bar.addWidget(self.stop_btn)
            bar.addStretch()
    
            self.status_label = QLabel("idle")
            self.status_label.setObjectName("muted")
            bar.addWidget(self.status_label)
            card_layout.addLayout(bar)
    
            # ── Output log ─────────────────────────────────────────────
            log_lbl = QLabel("output")
            log_lbl.setObjectName("sectionLabel")
            card_layout.addWidget(log_lbl)
    
            self.log_view = QPlainTextEdit()
            self.log_view.setObjectName("log")
            self.log_view.setReadOnly(True)
            self.log_view.setMinimumHeight(140)
            card_layout.addWidget(self.log_view)
    
            # Wiring
            self.auth_enabled.toggled.connect(self._on_auth_toggle)
            self.peribus_scope.currentIndexChanged.connect(self._refresh_peribus_status)
    
            # Initial state
            self._set_mode(self.pill_create)
            self._on_auth_toggle(self.auth_enabled.isChecked())
            self._refresh_peribus_status()
    
        # ── Mode switching ─────────────────────────────────────────────
    
        def _set_mode(self, picked: ModePill):
            for p in self.mode_pills:
                p.setSelected(p is picked)
            is_create  = (picked is self.pill_create)
            is_connect = (picked is self.pill_connect)
            self.workspace_panel.setVisible(is_create)
            self.auth_panel.setVisible(is_create)
            self.peribus_panel.setVisible(is_create)
            self.connect_panel.setVisible(is_connect)
            # Leaving create mode hides the peribus combo — so the public
            # risk no longer applies. Re-resolve shadow accordingly.
            self._refresh_shadow()
    
        def _refresh_shadow(self):
            """Pick the shadow colour from current state and animate to it.
    
            Priority:
              1. mux is running             → amber
              2. public peribus selected    → red
              3. everything else            → black
            """
            if getattr(self, "_running", False):
                self._animate_shadow_to(SHADOW_RUNNING)
                return
            is_create = self.pill_create.property("selected")
            if is_create and self.peribus_scope.currentData() == "public":
                self._animate_shadow_to(SHADOW_DANGER)
            else:
                self._animate_shadow_to(SHADOW_IDLE)
    
        def _current_mode(self) -> str:
            if self.pill_create.property("selected"):
                return "create"
            if self.pill_connect.property("selected"):
                return "connect"
            return "standalone"
    
        # ── Reactivity ─────────────────────────────────────────────────
    
        def _on_auth_toggle(self, enabled: bool):
            self.auth_token_edit.setEnabled(enabled)
            self.auth_regen_btn.setEnabled(enabled)
            self.auth_warning.setVisible(not enabled)
    
        def _refresh_peribus_status(self):
            scope = self.peribus_scope.currentData()
            is_public = (scope == "public")
            self.peribus_upnp.setVisible(is_public)
            self.peribus_bootstrap.setVisible(is_public)
            self._pe_bs_lbl.setVisible(is_public)
    
            if scope == "off":
                self.peribus_status.setText("✓  no peer discovery, no network exposure")
                self.peribus_status.setObjectName("ok")
            elif scope == "lan":
                self.peribus_status.setText(
                    "⚠  WARNING — LAN mode advertises this node via mDNS. "
                    "Anyone on the same Wi-Fi / LAN can discover you and read "
                    "your peribus feed."
                )
                self.peribus_status.setObjectName("warn")
            else:
                self.peribus_status.setText(
                    "⚠⚠  DOUBLE WARNING — PUBLIC mode joins the open Kademlia "
                    "DHT and may forward your IP to internet peers. Posts you "
                    "publish become discoverable by anyone running peribus, "
                    "anywhere. Only enable if you understand the privacy "
                    "implications."
                )
                self.peribus_status.setObjectName("warn2")
            # Re-polish so the objectName change re-applies QSS rules.
            self.peribus_status.setStyleSheet("")
            self.peribus_status.style().unpolish(self.peribus_status)
            self.peribus_status.style().polish(self.peribus_status)
            # Recompute size after the style and text changes so word-wrap
            # actually grows the label vertically.
            self.peribus_status.updateGeometry()
            self.peribus_status.adjustSize()
            # Drive the card's shadow off the same signal.
            self._refresh_shadow()
    
        # ── Start / Stop ───────────────────────────────────────────────
    
        def _collect_config(self) -> dict:
            bootstrap = [
                line.strip()
                for line in self.peribus_bootstrap.toPlainText().splitlines()
                if line.strip()
            ]
            return {
                "mode":              self._current_mode(),
                "workspace":         self.workspace_edit.text().strip(),
                "auth_enabled":      self.auth_enabled.isChecked(),
                "auth_token":        self.auth_token_edit.text().strip(),
                "peribus_scope":     self.peribus_scope.currentData(),
                "peribus_upnp":      self.peribus_upnp.isChecked(),
                "peribus_bootstrap": bootstrap,
                "connect_host":      self.connect_host.text().strip(),
                "connect_port":      self.connect_port.value(),
                "connect_token":     self.connect_token.text().strip(),
            }
    
        def _on_start(self):
            cfg = self._collect_config()
    
            # Public peribus → extra confirmation. The double warning earned it.
            if cfg["mode"] == "create" and cfg["peribus_scope"] == "public":
                box = QMessageBox(self)
                box.setWindowTitle("Confirm — public DHT")
                box.setIcon(QMessageBox.Warning)
                box.setText("Join the OPEN peribus DHT?")
                box.setInformativeText(
                    f"Your node will be reachable from the open internet on UDP "
                    f"port {PERIBUS_KAD_PORT}. Posts you publish become "
                    f"discoverable by anyone running peribus."
                )
                box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
                box.setDefaultButton(QMessageBox.Cancel)
                if box.exec() != QMessageBox.Yes:
                    return
    
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("starting…")
            self.log_view.clear()
    
            # Flip the shadow to amber the instant the user commits to
            # starting — gives immediate visual feedback without waiting for
            # the worker thread to actually come up. The worker_finished
            # handler will hold or reset it from here.
            self._running = True
            self._refresh_shadow()
    
            self._worker_thread = QThread(self)
            self._worker = StarterWorker(cfg)
            self._worker.moveToThread(self._worker_thread)
            self._worker.log.connect(self._append_log)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker_thread.started.connect(self._worker.run)
            self._worker_thread.start()
    
        def _on_stop(self):
            if self._worker is None:
                return
            self.status_label.setText("stopping…")
            self._append_log("\n── User requested stop ──\n")
            QTimer.singleShot(0, self._worker.stop_all)
            # Once cleanup is in flight, we're no longer "running" — reset
            # the controls and let the shadow fall back to its idle / risk
            # state based on the current form values.
            self._running = False
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("stopped")
            self._refresh_shadow()
    
        @Slot(str)
        def _append_log(self, text: str):
            self.log_view.moveCursor(QTextCursor.End)
            self.log_view.insertPlainText(text)
            self.log_view.moveCursor(QTextCursor.End)
    
        @Slot(bool, str)
        def _on_worker_finished(self, ok: bool, summary: str):
            if ok:
                self.status_label.setText("running")
                self._append_log(f"\n── Ready ──\n{summary}\n")
                self._running = True
            else:
                self.status_label.setText("stopped")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self._running = False
                if self._worker_thread:
                    self._worker_thread.quit()
                    self._worker_thread.wait(2000)
                    self._worker_thread = None
                    self._worker = None
            # Re-resolve shadow off the new running state. Success → amber,
            # failure → black (or red if public peribus is still selected).
            self._refresh_shadow()
    
        def closeEvent(self, event):
            if self._worker is not None:
                self._append_log("\n── Window closing, shutting down services ──\n")
                try:
                    self._worker.stop_all()
                except Exception:
                    pass
                if self._worker_thread:
                    self._worker_thread.quit()
                    self._worker_thread.wait(3000)
            event.accept()
    
        # ── Shadow animation ───────────────────────────────────────────
    
        def _animate_shadow_to(self, target: QColor):
            """Tween the card's drop-shadow colour to `target`.
    
            QVariantAnimation can interpolate QColor directly — we don't have
            to break it into RGBA components ourselves.
            """
            start = self._shadow.color()
            if start == target:
                return
            # Stop any animation in flight; otherwise its trailing frames
            # will fight whatever we set next.
            self._shadow_anim.stop()
            self._shadow_anim.setStartValue(start)
            self._shadow_anim.setEndValue(target)
            self._shadow_anim.start()
    
        @Slot(object)
        def _on_shadow_anim_step(self, value):
            # QVariantAnimation between two QColors gives us a QColor at each
            # step — assign it straight onto the effect.
            if isinstance(value, QColor):
                self._shadow.setColor(value)
    
        # ── Centering ──────────────────────────────────────────────────
    
        def _center_on_screen(self):
            """Center this window on the screen the user is currently on.
    
            We pick the screen under the mouse cursor — on a multi-monitor
            setup that's the one the user is actually looking at when they
            launched us. Falls back to the primary screen if there's no
            cursor screen (some headless / remote setups).
            """
            from PySide6.QtGui import QCursor
            screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
            if screen is None:
                return
            # availableGeometry excludes taskbars / docks, so we end up
            # actually centered in the usable area, not under a panel.
            avail = screen.availableGeometry()
            frame = self.frameGeometry()
            frame.moveCenter(avail.center())
            self.move(frame.topLeft())
    
        def showEvent(self, event):
            # Re-center on first show — pre-show move() is a hint and some
            # window managers (notably tiling ones, and a few X11 WMs)
            # override it until the window is actually mapped. Doing it
            # here too means the window lands centered on first paint.
            super().showEvent(event)
            if not getattr(self, "_centered_on_show", False):
                self._center_on_screen()
                self._centered_on_show = True
    
        # ── ESC closes (frameless friendliness) ────────────────────────
        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Escape:
                self.close()
            else:
                super().keyPressEvent(event)


    def run_gui():
        """
        Launch the PySide6 GUI. Called only when GUI_AVAILABLE.
        """
        app = QApplication(sys.argv)
        app.setApplicationName("riomux-start")
        # Match the paper theme as the app-wide fallback palette.
        pal = app.palette()
        pal.setColor(QPalette.Window, QColor(240, 237, 230))
        pal.setColor(QPalette.WindowText, QColor(26, 26, 26))
        pal.setColor(QPalette.Base, QColor(250, 247, 240))
        pal.setColor(QPalette.Text, QColor(26, 26, 26))
        pal.setColor(QPalette.Button, QColor(250, 247, 240))
        pal.setColor(QPalette.ButtonText, QColor(26, 26, 26))
        app.setPalette(pal)

        win = StarterWindow()
        win.show()
        sys.exit(app.exec())

else:

    def run_gui():
        """
        Stub used when PySide6 isn't installed.
        """
        print("Error: PySide6 is required for the GUI but not installed.",
              file=sys.stderr)
        print(f"  Import error: {GUI_IMPORT_ERROR}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Install with: pip install PySide6", file=sys.stderr)
        print("Or run with --cli to use the terminal interface.",
              file=sys.stderr)
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# ── Top-level dispatcher ─────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────


def main():
    """
    Decide which front-end to run.

    Default: try GUI; on missing PySide6, fall back to CLI.
    --gui:   force GUI; error if PySide6 missing.
    --cli:   force CLI (interactive stdin prompts).
    """
    parser = argparse.ArgumentParser(
        description="riomux start — GUI by default, CLI fallback",
        add_help=True,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--gui", action="store_true",
        help="Force the PySide6 GUI (default if PySide6 is installed).",
    )
    mode.add_argument(
        "--cli", action="store_true",
        help="Force the interactive terminal flow.",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli()
        return

    if args.gui:
        # User explicitly asked for GUI — error out clearly if it isn't here.
        run_gui()  # stub prints and exits when GUI_AVAILABLE is False
        return

    # No explicit flag: prefer GUI, fall back to CLI on ImportError.
    if GUI_AVAILABLE:
        run_gui()
    else:
        # Don't surprise the user — say what's going on once, then proceed.
        print("(PySide6 not available — falling back to CLI. "
              "Install PySide6 or pass --gui to require it.)",
              file=sys.stderr)
        run_cli()


if __name__ == "__main__":
    main()