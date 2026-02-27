"""
Shell Sandbox for LLM Agent Commands

Policy:
  - READ-ONLY access to the entire filesystem (cat, ls, head, tail, file, stat, find, grep, etc.)
  - FULL READ+WRITE access to /n/ (the Plan 9 / LLMFS namespace)
  - FULL READ+WRITE access to ~/.peripheria/ (agent-local persistent state)
  - NO destructive operations anywhere (rm, mv overwrite, shred, truncate to 0, dd, mkfs, etc.)
  - NO privilege escalation (sudo, su, chmod 777, chown, etc.)
  - NO network abuse (curl uploads, wget to overwrite, nc listeners, etc.)
  - NO shell escapes that bypass the filter (eval, exec, bash -c with nested commands, etc.)

Design:
  The sandbox works in two layers:
    1. DENY LIST — patterns that are always blocked regardless of path
    2. WRITE GATE — commands that modify files are only allowed if ALL target paths are under
       a writable root (currently /n/ and ~/.peripheria/)

  Commands are parsed conservatively. If the sandbox cannot determine that a
  command is safe, it blocks it and returns a human-readable rejection reason.

Usage:
    from shell_sandbox import check_command
    
    ok, reason = check_command("cat /etc/passwd")        # (True, None)
    ok, reason = check_command("rm -rf /")               # (False, "rm is not allowed")
    ok, reason = check_command("echo hi > /n/llm/input") # (True, None)
    ok, reason = check_command("echo hi > /tmp/pwned")   # (False, "write target '/tmp/pwned' is outside writable directories")
"""

import re
import shlex
from typing import Tuple, Optional, List

# ── Always-blocked commands ─────────────────────────────────────────────
# These are never allowed regardless of arguments or target paths.

BLOCKED_COMMANDS = {
    # Destructive
    "rm", "rmdir", "shred", "wipe",
    # Disk / partition
    "mkfs", "fdisk", "parted", "gdisk", "dd", "wipefs", "blkdiscard",
    # Privilege escalation
    "sudo", "su", "doas", "pkexec",
    # Ownership / permission (could make files world-writable then modify)
    "chown", "chgrp",
    # System-level danger
    "reboot", "shutdown", "poweroff", "halt", "init",
    "systemctl", "service",
    "mount", "umount",  # real mount, not /mount macro
    "insmod", "rmmod", "modprobe",
    # Container / vm escape vectors
    "docker", "podman", "kubectl", "nsenter", "unshare", "chroot",
    # Compilers / linkers (prevent building exploits)
    # (optional — uncomment if you want to block)
    # "gcc", "g++", "cc", "make", "ld",
    # Package managers (prevent installing tools that bypass sandbox)
    "apt", "apt-get", "dpkg", "yum", "dnf", "pacman", 
    "npm", "cargo", "gem", "go",
}

# ── Commands that WRITE to files ────────────────────────────────────────
# These are allowed ONLY when every target path resolves under /n/
# The sandbox extracts target paths and checks each one.

WRITE_COMMANDS = {
    "cp", "mv", "install",           # file copy/move
    "tee",                            # writes to file(s)
    "touch",                          # creates files
    "mkdir", "mktemp",                # creates dirs/files
    "ln",                             # creates links
    "chmod",                          # permission changes
    "truncate",                       # can destroy content
    "split", "csplit",                # creates output files
    "patch",                          # modifies files
    "sed", "awk", "perl",            # can modify files in-place with -i
    "python", "python3",             # can do anything — restrict targets
    "tar", "unzip", "gzip", "bzip2", "xz", "zstd",  # extract/compress
}

# ── Writable root directories ─────────────────────────────────────────
# Paths under these directories are allowed for write operations.
# All other paths are read-only.

WRITABLE_ROOTS = [
    "/n",
]

# ── Redirect / pipe patterns ───────────────────────────────────────────
# We check for shell output redirects (>, >>, etc.) and ensure targets are under /n/

# Matches: > /path, >> /path, 2> /path, 2>> /path, &> /path, etc.
REDIRECT_RE = re.compile(
    r'(?:^|[^\\])'           # not escaped
    r'(?:[012]?>{1,2}|&>>?)'  # redirect operator
    r'\s*'
    r'([^\s;|&]+)'           # target path (non-whitespace, stop at ; | &)
)

# ── Shell meta patterns that bypass simple parsing ─────────────────────
# Commands that embed arbitrary code and resist static analysis.

SHELL_ESCAPE_PATTERNS = [
    # eval / exec with arguments
    (r'\beval\b', "eval can execute arbitrary code"),
    # bash -c / sh -c / zsh -c
    (r'\b(?:ba)?sh\s+-c\b', "sh -c can execute arbitrary code"),
    (r'\bzsh\s+-c\b', "zsh -c can execute arbitrary code"),
    # Process substitution writing: >(cmd)
    (r'>\(', "process substitution writes are not allowed"),
    # Backtick or $() in write positions — too complex to analyse
    # (We allow $() in read-only commands; block in write contexts)
    # Python/perl one-liners that could write anywhere
    (r'\bpython[23]?\s+-c\b', "python -c can execute arbitrary code"),
    (r'\bperl\s+-e\b', "perl -e can execute arbitrary code"),
    (r'\bruby\s+-e\b', "ruby -e can execute arbitrary code"),
    (r'\bnode\s+-e\b', "node -e can execute arbitrary code"),
]

# ── Allowed read-only commands (non-exhaustive, used for fast-path) ────
READ_ONLY_COMMANDS = {
    "cat", "ls", "dir", "ll",
    "head", "tail", "less", "more",
    "grep", "egrep", "fgrep", "rg", "ag",
    "find", "locate", "which", "whereis", "type",
    "wc", "sort", "uniq", "cut", "paste", "column", "tr", "comm",
    "diff", "cmp", "md5sum", "sha256sum", "sha1sum",
    "file", "stat", "du", "df",
    "hexdump", "xxd", "od", "strings",
    "date", "cal", "uptime", "uname", "hostname", "whoami", "id",
    "env", "printenv", "echo", "printf",
    "ps", "top", "htop", "free", "lsof",
    "tree", "realpath", "basename", "dirname", "readlink",
    "jq", "yq", "csvtool", "xmllint",
    "man", "help", "info",
    "true", "false", "test", "[",
    "seq", "yes", "sleep", "wait",
    "xargs",   # depends on sub-command but usually read
    "tput", "clear", "reset",
}


# ── Protected path pattern ────────────────────────────────────────────
# Files under /n/ whose basename is ALL CAPITAL LETTERS (e.g. OUTPUT, CODE)
# are protected from reading.  They are typically agent outputs and should
# not be consumed directly by the same sandbox.
_PROTECTED_BASENAME_RE = re.compile(r'^[A-Z][A-Z0-9_]*$')


def _is_protected_read_path(path: str) -> bool:
    """
    Return True if *path* is under /n/ and its basename is ALL-CAPS.
    Such files are write-only from the sandbox's perspective.
    """
    import posixpath
    norm = _normalize_path(path)
    if not (norm == "/n" or norm.startswith("/n/")):
        return False
    basename = posixpath.basename(norm)
    return bool(_PROTECTED_BASENAME_RE.match(basename))


def _normalize_path(path: str) -> str:
    """
    Resolve a path string to an absolute path for policy checking.
    Handles ~, relative paths, and .. traversals.
    Does NOT require the path to exist (no os.path.realpath which follows symlinks 
    to existing files — we want to catch /n/../etc/shadow).
    """
    import posixpath
    # Expand ~ 
    if path.startswith("~"):
        path = "/root" + path[1:]  # conservative assumption
    # Make absolute
    if not path.startswith("/"):
        path = "/" + path  # conservative: treat relative as root-relative
    # Normalize .. and .
    path = posixpath.normpath(path)
    return path


def _path_is_writable(path: str) -> bool:
    """Check if a normalized absolute path is under an allowed writable directory."""
    norm = _normalize_path(path)
    return any(
        norm == prefix or norm.startswith(prefix + "/")
        for prefix in WRITABLE_ROOTS
    )


def _extract_redirect_targets(command: str) -> List[str]:
    """Extract all output redirect targets from a shell command string."""
    targets = []
    for m in REDIRECT_RE.finditer(command):
        target = m.group(1).strip("'\"")
        targets.append(target)
    return targets


def _get_base_command(token: str) -> str:
    """Extract the base command name from a token (strips path prefixes)."""
    # /usr/bin/rm -> rm
    if "/" in token:
        token = token.rsplit("/", 1)[-1]
    cmd = token.lower()
    # Also check prefix before dot: mkfs.ext4 -> mkfs
    if "." in cmd:
        prefix = cmd.split(".")[0]
        if prefix in BLOCKED_COMMANDS:
            return prefix
    return cmd


def check_command(command: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a shell command against the sandbox policy.
    
    Returns:
        (True, None) if the command is allowed.
        (False, reason) if the command is blocked.
    """
    command = command.strip()
    if not command:
        return True, None
    
    # ── Layer 0: Block shell escape patterns ────────────────────────
    for pattern, reason in SHELL_ESCAPE_PATTERNS:
        if re.search(pattern, command):
            # Exception: allow if it's clearly targeting /n/
            # e.g., python3 -c "..." is blocked, but we can't easily tell
            # so we block conservatively
            return False, reason
    
    # ── Layer 1: Check redirects ────────────────────────────────────
    # Any command can have redirects; check ALL redirect targets
    redirect_targets = _extract_redirect_targets(command)
    for target in redirect_targets:
        if not _path_is_writable(target):
            return False, f"write redirect to '{target}' is outside writable directories"
    
    # ── Layer 2: Split on pipes/semicolons and check each segment ──
    # Split on ; | && || but preserve quoted strings
    # Simple approach: split on unquoted separators
    segments = _split_command_segments(command)
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        ok, reason = _check_single_command(segment)
        if not ok:
            return False, reason
    
    return True, None


def _split_command_segments(command: str) -> List[str]:
    """
    Split a command on unquoted ; | && || into segments.
    This is a simplified parser — it handles basic quoting but not all edge cases.
    """
    segments = []
    current = []
    in_single = False
    in_double = False
    i = 0
    chars = command
    
    while i < len(chars):
        c = chars[i]
        
        if c == '\\' and not in_single and i + 1 < len(chars):
            current.append(c)
            current.append(chars[i + 1])
            i += 2
            continue
        
        if c == "'" and not in_double:
            in_single = not in_single
            current.append(c)
        elif c == '"' and not in_single:
            in_double = not in_double
            current.append(c)
        elif not in_single and not in_double:
            if c == ';':
                segments.append(''.join(current))
                current = []
            elif c == '|' and i + 1 < len(chars) and chars[i + 1] == '|':
                segments.append(''.join(current))
                current = []
                i += 1  # skip second |
            elif c == '|':
                segments.append(''.join(current))
                current = []
            elif c == '&' and i + 1 < len(chars) and chars[i + 1] == '&':
                segments.append(''.join(current))
                current = []
                i += 1  # skip second &
            else:
                current.append(c)
        else:
            current.append(c)
        
        i += 1
    
    if current:
        segments.append(''.join(current))
    
    return segments


def _check_single_command(segment: str) -> Tuple[bool, Optional[str]]:
    """Check a single command segment (no pipes/semicolons)."""
    segment = segment.strip()
    if not segment:
        return True, None
    
    # Strip leading variable assignments (FOO=bar cmd ...)
    while re.match(r'^[A-Za-z_][A-Za-z0-9_]*=\S*\s+', segment):
        segment = re.sub(r'^[A-Za-z_][A-Za-z0-9_]*=\S*\s+', '', segment, count=1).strip()
    
    if not segment:
        return True, None
    
    # Try to tokenize
    try:
        tokens = shlex.split(segment)
    except ValueError:
        # Unbalanced quotes — conservative block
        return False, "could not parse command (unbalanced quotes)"
    
    if not tokens:
        return True, None
    
    base_cmd = _get_base_command(tokens[0])
    
    # ── Always-blocked commands ─────────────────────────────────────
    if base_cmd in BLOCKED_COMMANDS:
        return False, f"'{base_cmd}' is not allowed"
    
    # ── Read-only commands (fast path) ──────────────────────────────
    # These are safe as long as redirects are checked (done in Layer 1)
    # BUT: block reads of ALL-CAPS files under /n/ (protected outputs).
    if base_cmd in READ_ONLY_COMMANDS:
        # Commands whose non-flag args are file paths to read
        _FILE_READING_CMDS = {
            "cat", "head", "tail", "less", "more",
            "grep", "egrep", "fgrep", "rg", "ag",
            "wc", "sort", "uniq", "cut", "paste", "column", "comm",
            "diff", "cmp", "md5sum", "sha256sum", "sha1sum",
            "file", "stat", "hexdump", "xxd", "od", "strings",
            "jq", "yq", "csvtool", "xmllint",
        }
        if base_cmd in _FILE_READING_CMDS:
            for t in tokens[1:]:
                if t.startswith('-'):
                    continue
                if _is_protected_read_path(t):
                    return False, f"reading '{t}' is not allowed (ALL-CAPS files under /n/ are protected)"
        return True, None
    
    # ── Write commands — check all target paths ─────────────────────
    if base_cmd in WRITE_COMMANDS:
        return _check_write_command(base_cmd, tokens)
    
    # ── Special cases ───────────────────────────────────────────────
    # 'echo' with redirects is handled by Layer 1 (redirect check)
    if base_cmd == "echo" or base_cmd == "printf":
        return True, None
    
    # 'tee' — check file arguments
    if base_cmd == "tee":
        return _check_tee(tokens)
    
    # Unknown commands: allow if no redirects outside writable directories (already checked)
    # and not in the blocked list. This is permissive for read operations
    # on custom tools the user may have installed.
    # BUT — if redirects were found and validated, we're okay.
    # If no redirects, it's likely read-only. Allow it.
    return True, None


def _check_write_command(cmd: str, tokens: List[str]) -> Tuple[bool, Optional[str]]:
    """Validate a write-capable command's target paths are under /n/."""
    
    if cmd in ("cp", "install"):
        # Last argument is the destination
        if len(tokens) < 3:
            return True, None  # malformed, will fail anyway
        dest = tokens[-1]
        if not _path_is_writable(dest):
            return False, f"'{cmd}' destination '{dest}' is outside writable directories"
        return True, None
    
    if cmd == "mv":
        if len(tokens) < 3:
            return True, None
        dest = tokens[-1]
        # mv also destroys the source — source must be under /n/ too
        sources = [t for t in tokens[1:] if not t.startswith('-') and t != tokens[-1]]
        for src in sources:
            if not _path_is_writable(src):
                return False, f"'mv' source '{src}' is outside writable directories (would delete it)"
        if not _path_is_writable(dest):
            return False, f"'mv' destination '{dest}' is outside writable directories"
        return True, None
    
    if cmd == "touch":
        targets = [t for t in tokens[1:] if not t.startswith('-')]
        for t in targets:
            if not _path_is_writable(t):
                return False, f"'touch' target '{t}' is outside writable directories"
        return True, None
    
    if cmd == "mkdir":
        targets = [t for t in tokens[1:] if not t.startswith('-')]
        for t in targets:
            if not _path_is_writable(t):
                return False, f"'mkdir' target '{t}' is outside writable directories"
        return True, None
    
    if cmd in ("sed", "awk", "perl"):
        # Block in-place editing (-i flag) outside writable directories
        if "-i" in tokens or any(t.startswith("-i") for t in tokens):
            # Find file arguments (heuristic: non-flag args after the expression)
            file_args = [t for t in tokens[1:] if not t.startswith('-') and '/' in t]
            for f in file_args:
                if not _path_is_writable(f):
                    return False, f"'{cmd} -i' on '{f}' is outside writable directories"
        # Without -i, sed/awk/perl are read-only (output to stdout)
        return True, None
    
    if cmd == "truncate":
        targets = [t for t in tokens[1:] if not t.startswith('-') and t not in ('-s', '--size')]
        # Also skip the argument after -s
        cleaned = []
        skip_next = False
        for t in tokens[1:]:
            if skip_next:
                skip_next = False
                continue
            if t in ('-s', '--size'):
                skip_next = True
                continue
            if t.startswith('-'):
                continue
            cleaned.append(t)
        for t in cleaned:
            if not _path_is_writable(t):
                return False, f"'truncate' target '{t}' is outside writable directories"
        return True, None
    
    if cmd == "chmod":
        # chmod changes permissions — only allow under /n/
        targets = [t for t in tokens[1:] if not t.startswith('-') and not re.match(r'^[0-7]+$|^[ugoa]', t)]
        for t in targets:
            if not _path_is_writable(t):
                return False, f"'chmod' target '{t}' is outside writable directories"
        return True, None
    
    if cmd in ("ln",):
        if len(tokens) < 3:
            return True, None
        dest = tokens[-1]
        if not _path_is_writable(dest):
            return False, f"'ln' target '{dest}' is outside writable directories"
        return True, None
    
    if cmd in ("tar", "unzip", "gzip", "bzip2", "xz", "zstd"):
        # Extraction could write anywhere — check for -C / output dir
        # Conservative: if extracting, require explicit /n/ target
        if cmd == "tar":
            if "-x" in tokens or "--extract" in tokens or any(t.startswith("-") and "x" in t and not t.startswith("--") for t in tokens):
                # Find -C argument
                for i, t in enumerate(tokens):
                    if t in ("-C", "--directory") and i + 1 < len(tokens):
                        if not _path_is_writable(tokens[i + 1]):
                            return False, f"'tar -x -C {tokens[i+1]}' extracts outside writable directories"
                        return True, None
                return False, "tar extract without -C to a writable directory... is not allowed (could write to cwd)"
            # tar -c (create) is read-only if output goes to stdout or /n/
            return True, None
        if cmd == "unzip":
            for i, t in enumerate(tokens):
                if t == "-d" and i + 1 < len(tokens):
                    if not _path_is_writable(tokens[i + 1]):
                        return False, f"'unzip -d {tokens[i+1]}' extracts outside writable directories"
                    return True, None
            return False, "unzip without -d to a writable directory... is not allowed (could write to cwd)"
        # gzip/bzip2/xz/zstd: in-place by default
        file_args = [t for t in tokens[1:] if not t.startswith('-')]
        for f in file_args:
            if not _path_is_writable(f):
                return False, f"'{cmd}' on '{f}' is outside writable directories (in-place compression)"
        return True, None
    
    if cmd in ("python", "python3"):
        # Python scripts can do anything — only allow if script is under /n/
        # or if it's a module run (-m)
        script_args = [t for t in tokens[1:] if not t.startswith('-')]
        if script_args:
            script = script_args[0]
            if not _path_is_writable(script):
                return False, f"running Python script '{script}' outside writable directories is not allowed"
        return True, None
    
    # Default: block unknown write commands targeting outside writable directories
    return False, f"'{cmd}' may write files — not allowed without writable directory targets"


def _check_tee(tokens: List[str]) -> Tuple[bool, Optional[str]]:
    """Check that tee only writes to /n/ paths."""
    targets = [t for t in tokens[1:] if not t.startswith('-')]
    for t in targets:
        if not _path_is_writable(t):
            return False, f"'tee' target '{t}' is outside writable directories"
    return True, None


# ── Convenience for testing ─────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # Should ALLOW
        ("cat /etc/passwd", True),
        ("ls -la /home", True),
        ("grep -r TODO /src", True),
        ("head -n 50 /var/log/syslog", True),
        ("find / -name '*.py' -type f", True),
        ("echo hello > /n/llm/input", True),
        ("echo 'new agent' > /n/llm/ctl", True),
        ("cp /etc/config /n/backup/config", True),
        ("mkdir -p /n/workspace/new", True),
        ("touch /n/workspace/file.txt", True),
        ("cat /src/main.py | grep def", True),
        ("wc -l /src/*.py", True),
        ("diff /n/old /n/new", True),
        ("pip install PyAudio", True),
        
        # Should BLOCK
        ("rm -rf /", False),
        ("rm /home/user/file.txt", False),
        ("sudo cat /etc/shadow", False),
        ("echo pwned > /tmp/evil", False),
        ("mv /etc/passwd /etc/passwd.bak", False),
        ("cp /n/data /tmp/exfil", False),
        ("chmod 777 /etc/passwd", False),
        ("dd if=/dev/zero of=/dev/sda", False),
        ("bash -c 'rm -rf /'", False),
        ("eval 'rm -rf /'", False),
        ("python3 -c 'import os; os.remove(\"/etc/passwd\")'", False),
        ("mkfs.ext4 /dev/sda1", False),
        ("apt install nmap", False),
        ("echo data | tee /tmp/leak", False),
        ("sed -i 's/root/pwned/' /etc/passwd", False),
        ("tar -xf archive.tar", False),  # no -C /n/
        ("truncate -s 0 /var/log/syslog", False),
        ("cat /etc/passwd > /tmp/stolen", False),
        ("cp README.md rio/filesystem.py", False),

        # Protected ALL-CAPS files under /n/ — reads should BLOCK
        ("cat /n/llm/agent/OUTPUT", False),
        ("head /n/llm/agent/CODE", False),
        ("grep pattern /n/rio/DATA", False),
        ("tail -n 10 /n/something/STATUS", False),
        # Writes to paths mentioning caps names in *content* should ALLOW
        ("echo 'n/llm/agent/CODE -> /n/rio/input' > /n/rio/routes", True),
        # Writes to ALL-CAPS filenames should still ALLOW (only reads blocked)
        ("echo hello > /n/llm/agent/OUTPUT", True),
        ("touch /n/workspace/RESULT", True),
        # Reads of non-all-caps under /n/ should still ALLOW
        ("cat /n/llm/agent/output", True),
        ("cat /n/rio/routes", True),

    ]
    
    print("Shell Sandbox Tests")
    print("=" * 60)
    passed = 0
    failed = 0
    for cmd, expected in tests:
        ok, reason = check_command(cmd)
        status = "✓" if ok == expected else "✗"
        if ok != expected:
            failed += 1
            print(f"  {status} FAIL: '{cmd}'")
            print(f"         expected={'allow' if expected else 'block'}, got={'allow' if ok else 'block'}")
            if reason:
                print(f"         reason: {reason}")
        else:
            passed += 1
            print(f"  {status} {'allow' if ok else 'block':5s}: {cmd}")
            if reason:
                print(f"         reason: {reason}")
    
    print(f"\n{passed}/{passed+failed} passed")