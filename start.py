#!/usr/bin/env python3
"""
Rio Mux — Interactive Start Script

Walks the user through:
  1. Ensuring /n exists (and /n/llm, /n/rio for standalone mounts)
  2. Choosing a mode:
     a) Create a new mux  → starts llmfs, rio, riomux, mounts /n
     b) Connect to a mux  → mounts remote riomux via 9pfuse
     c) Standalone         → starts llmfs + rio, mounts each separately

The mux exposes:
    /n/
    ├── <n>/     → rio (display server + scene)
    └── llm/        → llmfs (agents, providers)

Key routing example (works correctly through the mux):
    while true; do cat /n/llm/claude/output > /n/<n>/scene/parse; done

All blocking reads, clunk semantics, and streaming are preserved
because riomux operates at the 9P wire level.
"""

import subprocess
import sys
import os
import signal
import time
import shutil
import getpass
import socket

# ── Resolve project root ─────────────────────────────────────────
# start_mux.py lives in riomux/ inside the project root.
# We need the project root on PYTHONPATH so `python -m riomux`,
# `python -m llmfs.main`, and `python -m rio.main` all resolve.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # one level up from riomux/

# Build an env dict that prepends PROJECT_ROOT to PYTHONPATH
_env = os.environ.copy()
_existing_pp = _env.get("PYTHONPATH", "")
_env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + _existing_pp if _existing_pp else "")

# ── Defaults ─────────────────────────────────────────────────────

LLMFS_PORT = 5640
RIO_PORT   = 5641
MUX_PORT   = 5642

MOUNT_BASE = "/n"

# ── Helpers ──────────────────────────────────────────────────────

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


# ── Mount helpers ────────────────────────────────────────────────

def mount_9pfuse(host, port, mountpoint):
    """Mount a 9P server via 9pfuse. Returns the Popen process."""
    os.makedirs(mountpoint, exist_ok=True)

    # Check if already mounted
    if os.path.ismount(mountpoint):
        print(f"  ⚠ {mountpoint} already mounted, unmounting...")
        subprocess.run(["fusermount", "-u", mountpoint], check=False)
        time.sleep(0.5)

    addr = f"{host}:{port}"
    print(f"  Mounting {addr} → {mountpoint}")
    proc = subprocess.Popen(
        ["9pfuse", addr, mountpoint],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Give it a moment
    time.sleep(1)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode()
        print(f"  ✗ 9pfuse failed: {stderr}")
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


def unmount(mountpoint):
    """Unmount a FUSE mount."""
    if os.path.ismount(mountpoint):
        subprocess.run(["fusermount", "-u", mountpoint], check=False)


# ── Mode: Create new mux ────────────────────────────────────────

def nuke_stale_mounts():
    """
    Kill ALL existing 9pfuse processes and unmount everything under /n/.

    Previous runs may have left stale FUSE mounts (e.g. /n/llm, /n/rioa,
    /n/mux) that block new mounts or cause 'Transport endpoint is not
    connected' errors.  This cleans the slate.
    """
    print("\n── Cleaning stale mounts ──")

    # 1. Kill all 9pfuse processes (except our own, but we haven't started any yet)
    result = subprocess.run(
        ["pgrep", "-a", "9pfuse"], capture_output=True, text=True
    )
    if result.stdout.strip():
        print("  Killing stale 9pfuse processes:")
        for line in result.stdout.strip().splitlines():
            print(f"    {line}")
        subprocess.run(["pkill", "-9", "9pfuse"], check=False)
        time.sleep(0.5)
    else:
        print("  No stale 9pfuse processes found")

    # 2. Also kill stale attachment scripts (cat loops from previous runs)
    subprocess.run(["pkill", "-f", "llmfs_attach"], capture_output=True)
    subprocess.run(["pkill", "-f", "acme_attach"], capture_output=True)

    # 3. Unmount everything under /n/ (both direct mounts and nested mux mounts)
    #    Read /proc/mounts to find all FUSE mounts under /n/
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
        # Unmount deepest paths first (e.g. /n/mux/llm before /n/mux)
        stale_mounts.sort(key=len, reverse=True)
        print(f"  Unmounting {len(stale_mounts)} stale mount(s):")
        for mp in stale_mounts:
            print(f"    fusermount -u {mp}")
            subprocess.run(["fusermount", "-u", mp], check=False)
        time.sleep(0.5)
    else:
        print("  No stale mounts found under /n/")

    print("  ✓ Clean slate")


def mode_create_mux(name, processes, mounts):
    """Start llmfs, rio, riomux, and mount under /n."""
    print(f"\n── Creating mux as '{name}' ──")

    # Clean up any stale mounts/processes from previous runs
    nuke_stale_mounts()

    # 1. Start LLMFS
    print(f"\n  Starting LLMFS on port {LLMFS_PORT}...")
    llmfs = subprocess.Popen([
        sys.executable, "-m", "llmfs.main",
        "--port", str(LLMFS_PORT),
    ], env=_env)
    processes.append(("llmfs", llmfs))

    if not wait_for_port(LLMFS_PORT):
        print("  ✗ LLMFS failed to start")
        return False
    print(f"  ✓ LLMFS running on port {LLMFS_PORT}")

    # 2. Start Rio
    print(f"\n  Starting Rio on port {RIO_PORT}...")
    rio = subprocess.Popen([
        sys.executable, "-m", "rio.main",
        "--port", str(RIO_PORT),
        "--workspace", name,
        "--mux-mount", MOUNT_BASE,
    ], env=_env)
    processes.append(("rio", rio))

    if not wait_for_port(RIO_PORT):
        print("  ✗ Rio failed to start")
        return False
    print(f"  ✓ Rio running on port {RIO_PORT}")

    # 3. Start riomux
    print(f"\n  Starting riomux on port {MUX_PORT}...")
    print(f"    Backend: {name} → 127.0.0.1:{RIO_PORT}")
    print(f"    Backend: llm → 127.0.0.1:{LLMFS_PORT}")
    mux = subprocess.Popen([
        sys.executable, "-m", "riomux",
        "--port", str(MUX_PORT),
        "--backend", f"{name}=127.0.0.1:{RIO_PORT}",
        "--backend", f"llm=127.0.0.1:{LLMFS_PORT}",
    ], env=_env)
    processes.append(("riomux", mux))

    if not wait_for_port(MUX_PORT):
        print("  ✗ riomux failed to start")
        return False
    print(f"  ✓ riomux running on port {MUX_PORT}")

    # 4. Mount via 9pfuse
    print(f"\n  Mounting mux at {MOUNT_BASE}...")
    fuse = mount_9pfuse("127.0.0.1", MUX_PORT, MOUNT_BASE)
    if fuse:
        processes.append(("9pfuse", fuse))
        mounts.append(MOUNT_BASE)
    else:
        print("  ⚠ Mount failed — servers are running but not mounted.")
        print(f"    You can mount manually: 9pfuse 127.0.0.1:{MUX_PORT} {MOUNT_BASE}")

    return True


# ── Mode: Connect to existing mux ───────────────────────────────

def mode_connect_mux(processes, mounts):
    """Connect to an existing riomux and mount it."""
    print("\n── Connect to existing mux ──")

    host = prompt("Mux host", "192.168.1.10")
    port = int(prompt("Mux port", str(MUX_PORT)))

    # Mount the remote mux
    fuse = mount_9pfuse(host, port, MOUNT_BASE)
    if fuse:
        processes.append(("9pfuse-remote", fuse))
        mounts.append(MOUNT_BASE)
    else:
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
    ], env=_env)
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
    ], env=_env)
    processes.append(("rio", rio))

    if not wait_for_port(RIO_PORT):
        print("  ✗ Rio failed to start")
        return False
    print(f"  ✓ Rio running on port {RIO_PORT}")

    # Mount each separately
    has_9pfuse = check_binary("9pfuse")

    if has_9pfuse:
        ensure_dir("/n/llm")
        fuse_llm = mount_9pfuse("127.0.0.1", LLMFS_PORT, "/n/llm")
        if fuse_llm:
            processes.append(("9pfuse-llm", fuse_llm))
            mounts.append("/n/llm")

        ensure_dir("/n/rio")
        fuse_rio = mount_9pfuse("127.0.0.1", RIO_PORT, "/n/rio")
        if fuse_rio:
            processes.append(("9pfuse-rio", fuse_rio))
            mounts.append("/n/rio")
    else:
        print(f"\n  No 9pfuse — mount manually:")
        print(f"    9pfuse 127.0.0.1:{LLMFS_PORT} /n/llm")
        print(f"    9pfuse 127.0.0.1:{RIO_PORT} /n/rio")

    return True


# ── Cleanup ──────────────────────────────────────────────────────

def cleanup(processes, mounts):
    """Stop all processes and unmount."""
    print("\n── Shutting down ──")

    # Unmount first
    for mp in reversed(mounts):
        print(f"  Unmounting {mp}...")
        unmount(mp)

    # Terminate processes in reverse order
    for name, proc in reversed(processes):
        if proc.poll() is None:
            print(f"  Stopping {name} (pid {proc.pid})...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    print("  ✓ All stopped.")


# ── Main ─────────────────────────────────────────────────────────

def main():
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
    if not check_binary("9pfuse"):
        print("\n  ⚠ 9pfuse not found on PATH.")
        print("    Install plan9port or add it to PATH for mounting.")
        print("    Mounting will be unavailable.\n")

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

    if mode == "create":
        name = prompt("Workspace name (e.g. your name)", getpass.getuser())
        name = name.strip().lower().replace(" ", "_")
        if not name:
            name = "default"
        ok = mode_create_mux(name, processes, mounts)

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
        for name, proc in running:
            print(f"    • {name:16s}  pid {proc.pid}")
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

        print(f"\n  Example routing (streaming, blocking-safe):")
        print(f"    while true; do")
        print(f"      cat /n/llm/claude/output > /n/*/scene/parse")
        print(f"    done")
        print(f"\n  Press Ctrl+C to stop everything.")
        print(f"{'=' * 56}\n")

        # Wait for processes — if Rio exits (window closed / Exit menu),
        # tear everything else down automatically.
        try:
            while True:
                for name, proc in processes:
                    if proc.poll() is not None:
                        rc = proc.returncode
                        if name == "rio":
                            # Rio closed (user chose Exit or closed window)
                            # → bring down the whole mux stack
                            print(f"\n  Rio exited (code {rc}) — shutting down mux stack...")
                            raise KeyboardInterrupt  # reuse the cleanup path
                        elif rc != 0 and rc != -15:  # not SIGTERM
                            print(f"  ⚠ {name} exited with code {rc}")
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        print("\n  No services running.")

    cleanup(processes, mounts)


if __name__ == "__main__":
    main()