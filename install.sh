#!/usr/bin/env bash
#
# Peribus installer (self-contained)
# Usage:
#   bash install.sh
#
# Assumes you are ALREADY inside the cloned `peribus` repository directory.
#
# This script performs the complete installation, replacing the old two-stage
# (bootstrap + graphical install.py) flow. It:
#   1. Finds a compatible Python (3.11 or 3.12)
#   2. Creates and activates a virtual environment
#   3. Upgrades pip tooling
#   4. Installs all system (apt) prerequisites:
#        - everything listed in pre.txt
#        - fuse3 + build-essential (required by the in-house ninepfuse.py
#          client and to build C extensions)
#   5. Installs every Python dependency from requirements.txt, one at a time,
#      so a single failing/missing package does not abort the whole run
#   6. Creates the /n mount point and hands it to the current user
#
# All of the work formerly done by the PySide6 graphical installer (install.py)
# is done here in plain shell — no GUI, no PySide6 bootstrap required.
#
set -uo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
VENV_DIR=".venv"
# Acceptable Python minor versions, in order of preference.
PY_CANDIDATES=("python3.11" "python3.12" "python3.13")

REQUIREMENTS="requirements.txt"
PRE_TXT="pre.txt"
MOUNT_POINT="/n"

# Fallback system prerequisites if pre.txt is missing (mirrors install.py).
PRE_FALLBACK=("libminizip-dev" "libxcb-cursor0" "portaudio19-dev")

# Always-required system packages (added on top of pre.txt in install.py).
#   fuse3          -> libfuse3 + fusermount3, used by the pyfuse3-based client.
#                     NOTE: install fuse3 ONLY; the legacy `fuse` package
#                     conflicts with fuse3 on Ubuntu 22.04+.
#   build-essential-> lets pip build C extensions.
ALWAYS_PKGS=("fuse3" "build-essential")

# Tallies, mirroring install.py's succeeded/failed accounting.
SUCCEEDED=0
FAILED=0

# ----------------------------------------------------------------------------
# Pretty output helpers
# ----------------------------------------------------------------------------
if [ -t 1 ]; then
    BOLD="$(printf '\033[1m')"
    RED="$(printf '\033[31m')"; GRN="$(printf '\033[32m')"
    YLW="$(printf '\033[33m')"; CYN="$(printf '\033[36m')"
    RST="$(printf '\033[0m')"
else
    BOLD=""; RED=""; GRN=""; YLW=""; CYN=""; RST=""
fi

info()  { printf "%s==>%s %s\n" "${CYN}${BOLD}" "${RST}" "$*"; }
ok()    { printf "%s ok %s %s\n" "${GRN}${BOLD}" "${RST}" "$*"; }
warn()  { printf "%swarn%s %s\n" "${YLW}${BOLD}" "${RST}" "$*"; }
die()   { printf "%sERROR%s %s\n" "${RED}${BOLD}" "${RST}" "$*" >&2; exit 1; }
step()  { printf "\n%s=== %s ===%s\n" "${CYN}${BOLD}" "$*" "${RST}"; }

banner() {
    printf "%s" "${CYN}${BOLD}"
    cat <<'EOF'
   ___  ___ ___ ___ ___ _   _ ___
  | _ \| __| _ \_ _| _ ) | | / __|
  |  _/| _||   /| || _ \ |_| \__ \
  |_|  |___|_|_\___|___/\___/|___/
        Plan 9 inspired AI graphics engine
EOF
    printf "%s\n" "${RST}"
}

# ----------------------------------------------------------------------------
# Pre-flight checks
# ----------------------------------------------------------------------------
banner

# When piped through `curl | bash`, stdin is the script, so any interactive
# prompt (e.g. a sudo password) must read from the terminal.
if [ ! -t 0 ] && [ -e /dev/tty ]; then
    exec < /dev/tty || true
fi

# Sanity: we should already be inside the cloned peribus repo.
if [ ! -f "$REQUIREMENTS" ] && [ ! -f "$PRE_TXT" ] && [ ! -f "start.py" ]; then
    warn "This doesn't look like the peribus repository directory."
    warn "Expected to find one of: $REQUIREMENTS, $PRE_TXT, start.py"
    warn "Run this script from inside the cloned 'peribus' directory."
fi

# ----------------------------------------------------------------------------
# Determine a sudo prefix only if we are not already root and sudo exists.
#   (equivalent of install.py's _sudo_prefix)
# ----------------------------------------------------------------------------
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi
fi

# ----------------------------------------------------------------------------
# Find a compatible Python interpreter
# ----------------------------------------------------------------------------
PYTHON=""
for cand in "${PY_CANDIDATES[@]}"; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYTHON="$(command -v "$cand")"
        break
    fi
done

# Fall back to `python3` only if it reports 3.11 or 3.12.
if [ -z "$PYTHON" ] && command -v python3 >/dev/null 2>&1; then
    ver="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")"
    case "$ver" in
        3.11|3.12) PYTHON="$(command -v python3)" ;;
    esac
fi

if [ -z "$PYTHON" ]; then
    die "No compatible Python found. Peribus needs Python 3.11 or 3.12.
     On Ubuntu you can install 3.11 with:
       sudo add-apt-repository ppa:deadsnakes/ppa
       sudo apt update
       sudo apt install python3.11 python3.11-venv python3.11-dev"
fi

PYVER="$("$PYTHON" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
ok "Using Python $PYVER ($PYTHON)"

# Ensure the venv module is available for this interpreter.
if ! "$PYTHON" -c 'import venv' >/dev/null 2>&1; then
    PYMM="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    die "The venv module is missing for $PYTHON.
     Install it with: sudo apt install python${PYMM}-venv"
fi

# ----------------------------------------------------------------------------
# Create and activate the virtual environment
# ----------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment in ./$VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR" || die "Failed to create virtual environment."
else
    info "Virtual environment already exists, reusing it."
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "Virtual environment activated"

# The venv's python — equivalent of install.py's VENV_PYTHON (sys.executable).
VENV_PYTHON="$(command -v python)"

# ----------------------------------------------------------------------------
# Upgrade pip tooling
# ----------------------------------------------------------------------------
info "Upgrading pip / setuptools / wheel..."
if python -m pip install --upgrade pip setuptools wheel >/dev/null; then
    ok "pip tooling upgraded"
else
    warn "Could not upgrade pip tooling; continuing anyway."
fi

# ----------------------------------------------------------------------------
# Helper: install a batch of apt packages.
#   Mirrors install.py's install_apt(): try the whole batch first, and if that
#   fails, retry each package individually so one bad name doesn't block the
#   rest. Individual failures are non-fatal.
# ----------------------------------------------------------------------------
install_apt() {
    local packages=("$@")
    if ! command -v apt-get >/dev/null 2>&1; then
        warn "apt-get not found — skipping system packages."
        warn "Install these manually: ${packages[*]}"
        return 1
    fi

    if [ -z "$SUDO" ] && [ "$(id -u)" -ne 0 ]; then
        warn "Not running as root and sudo is unavailable."
        warn "If apt installs fail, install these manually: ${packages[*]}"
    fi

    $SUDO apt-get update || warn "apt-get update failed; continuing."

    # Fast path: install the whole batch at once.
    if $SUDO apt-get install -y "${packages[@]}"; then
        return 0
    fi

    # Batch failed — likely one unavailable/renamed package on this distro.
    # Retry each package individually.
    warn "batch install failed; retrying packages individually..."
    local failed=()
    local pkg
    for pkg in "${packages[@]}"; do
        if ! $SUDO apt-get install -y "$pkg"; then
            failed+=("$pkg")
        fi
    done
    if [ "${#failed[@]}" -gt 0 ]; then
        warn "could not install: ${failed[*]}"
        warn "(continuing anyway; install these manually if needed)"
    fi
    # Return success so optional system packages don't abort the run, exactly
    # as install.py does.
    return 0
}

# ----------------------------------------------------------------------------
# Read pre.txt (system prerequisites). Falls back to documented defaults.
#   Mirrors install.py's read_pre_txt().
# ----------------------------------------------------------------------------
read_pre_txt() {
    if [ -f "$PRE_TXT" ]; then
        # Strip blank lines and comments.
        grep -vE '^\s*(#|$)' "$PRE_TXT" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
    else
        printf "%s\n" "${PRE_FALLBACK[@]}"
    fi
}

# ----------------------------------------------------------------------------
# Read requirements.txt (pip packages). Skips comments and option lines.
#   Mirrors install.py's read_requirements().
# ----------------------------------------------------------------------------
read_requirements() {
    [ -f "$REQUIREMENTS" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        # trim leading/trailing whitespace
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -z "$line" ] && continue
        case "$line" in
            \#*|-*) continue ;;   # skip comments and -r/-e/--option lines
        esac
        printf "%s\n" "$line"
    done < "$REQUIREMENTS"
}

# ============================================================================
# STEP 1 — System apt prerequisites
# ============================================================================
step "Installing system packages (apt)"

# Collect pre.txt entries + always-required packages, de-duped, order preserved.
declare -a APT_PKGS=()
declare -A APT_SEEN=()
add_apt_pkg() {
    local p="$1"
    [ -z "$p" ] && return
    if [ -z "${APT_SEEN[$p]:-}" ]; then
        APT_PKGS+=("$p")
        APT_SEEN[$p]=1
    fi
}

while IFS= read -r p; do
    add_apt_pkg "$p"
done < <(read_pre_txt)

for p in "${ALWAYS_PKGS[@]}"; do
    add_apt_pkg "$p"
done

if [ "${#APT_PKGS[@]}" -gt 0 ]; then
    info "Packages: ${APT_PKGS[*]}"
    if install_apt "${APT_PKGS[@]}"; then
        ok "System packages step complete"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        warn "System packages step had problems (non-fatal)"
        FAILED=$((FAILED + 1))
    fi
else
    warn "No system packages to install."
fi

# ============================================================================
# STEP 2 — Python dependencies from requirements.txt, one at a time
# ============================================================================
step "Installing Python dependencies (pip)"

PIP_TOTAL=0
PIP_FAILED=()
while IFS= read -r pkg; do
    [ -z "$pkg" ] && continue
    PIP_TOTAL=$((PIP_TOTAL + 1))
    info "Installing $pkg"
    if "$VENV_PYTHON" -m pip install "$pkg"; then
        ok "installed $pkg"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        # Individual pip failures are non-fatal (the list may be messy /
        # incomplete), exactly as in install.py (Step optional=True).
        warn "could not install $pkg (continuing)"
        PIP_FAILED+=("$pkg")
        FAILED=$((FAILED + 1))
    fi
done < <(read_requirements)

if [ "$PIP_TOTAL" -eq 0 ]; then
    warn "No pip packages found in $REQUIREMENTS (skipping)."
elif [ "${#PIP_FAILED[@]}" -gt 0 ]; then
    warn "pip packages that failed: ${PIP_FAILED[*]}"
fi

# ============================================================================
# STEP 3 — Create the /n mount point
#   Mirrors install.py's create_mount_point().
# ============================================================================
step "Creating $MOUNT_POINT mount point"

create_mount_point() {
    if [ -e "$MOUNT_POINT" ]; then
        info "$MOUNT_POINT already exists."
        return 0
    fi
    if ! $SUDO mkdir -p "$MOUNT_POINT"; then
        return 1
    fi
    local user="${USER:-${LOGNAME:-}}"
    if [ -n "$user" ]; then
        $SUDO chown "$user" "$MOUNT_POINT" || warn "Could not chown $MOUNT_POINT to $user."
    fi
    return 0
}

if create_mount_point; then
    ok "$MOUNT_POINT ready"
    SUCCEEDED=$((SUCCEEDED + 1))
else
    # Non-fatal, as in install.py (Step optional=True).
    warn "Could not create $MOUNT_POINT (non-fatal)."
    FAILED=$((FAILED + 1))
fi

# ============================================================================
# Summary
# ============================================================================
echo
if [ "$FAILED" -eq 0 ]; then
    ok "Complete — ${SUCCEEDED} step(s) succeeded."
    info "Launch Peribus with:"
    printf "    %ssource %s/bin/activate%s\n" "${BOLD}" "$VENV_DIR" "${RST}"
    printf "    %spython start.py%s\n" "${BOLD}" "${RST}"
else
    warn "Finished with ${FAILED} issue(s); ${SUCCEEDED} succeeded. Check the log above."
    info "You can still try launching with:"
    printf "    %ssource %s/bin/activate%s\n" "${BOLD}" "$VENV_DIR" "${RST}"
    printf "    %spython start.py%s\n" "${BOLD}" "${RST}"
fi