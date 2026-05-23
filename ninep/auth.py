"""
ninep.auth — 9P2000 token authentication for any Server9P-based server.

Self-contained copy of the auth machinery used by riomux. Lives in the
`ninep` package so any 9P server built on top of `ninep.server.Server9P`
(rio, llmfs, peribusd, etc.) can opt in without depending on riomux.

The protocol matches riomux exactly — same two flows (raw token and
p9any/pass) — so the same factotum entries and the same ninepfuse
'-t <token>' flag work against rio directly:

    1. Client sends Tauth(afid, uname, aname)
       → Server creates an AuthFid, returns Rauth(aqid)

    2. Client writes a token to the afid via Twrite(afid, token)
       → Server validates the token

    3. Client reads the afid via Tread(afid, ...)
       → Server returns "ok\\n" or "err: <reason>\\n"

    4. Client sends Tattach(fid, afid, uname, aname)
       → Server checks afid is authenticated, then allows attach

If auth is disabled (no secrets configured), Tauth returns Rerror
("authentication not required") and Tattach proceeds without an afid,
preserving backward compatibility with existing clients.

Env vars (for rio-side use):
    RIO_AUTH_TOKENS  — comma-separated tokens
    RIO_AUTH_FILE    — path to a secrets file
    RIO_AUTH_TIMEOUT — auth fid timeout in seconds (default 60)

Usage from a 9P server:
    from ninep.auth import AuthManager, AuthContext, AuthFid

    auth = AuthManager(secrets=["mytoken123"])
    auth = AuthManager(secrets_file="/etc/rio/tokens")
    auth = AuthManager.from_env()

    # Per connection:
    ctx = AuthContext(auth)
    afid_state = ctx.handle_tauth(fid, uname, aname)  # → AuthFid
    ctx.handle_auth_write(fid, token_bytes)
    status = ctx.handle_auth_read(fid, 0, 64)         # → b"ok\\n" / b"err: ...\\n"
    err = ctx.check_attach(afid, uname)               # None if OK
"""

import hmac
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger("ninep.auth")

NOFID = 0xFFFFFFFF


@dataclass
class AuthFid:
    """
    State for a single auth fid.

    Supports two auth protocols, auto-detected from the first write:

    1. Raw token (ninepfuse / custom clients):
       Client writes token directly, reads "ok\n" or "err: ...\n".

    2. p9any + pass (Plan 9 factotum):
       Read  → "v.2 role=server proto=pass@ninep\0"
       Write → "proto=pass@ninep\0" (or "pass\0")
       Read  → "OK"
       Write → password bytes
       Read  → "ok\n" or "err: ...\n"
    """
    fid: int
    uname: str
    aname: str
    authenticated: bool = False
    error: str = ""
    _phase: str = "init"       # init, p9any_offer, p9any_ok, pass_wait, raw, done
    _token_buf: bytes = b""
    _read_buf: bytes = b""
    _created_at: float = field(default_factory=time.time)

    # The p9any offer string: version, role, and available protocols.
    # We keep the domain generic ("ninep") so the same factotum entry
    # works against any ninep-based server.
    P9ANY_OFFER = b"v.2 role=server proto=pass@ninep\0"
    P9ANY_OK = b"OK"

    def write(self, data: bytes) -> int:
        """
        Handle a write to the auth fid.

        Auto-detects raw token vs p9any based on the first write.
        Returns the number of bytes accepted.
        """
        if self._phase == "init":
            # First write — detect protocol
            text = data.rstrip(b'\0').decode('utf-8', errors='replace').strip()

            if text.startswith("proto=") or text.startswith("pass"):
                # Client is doing p9any — they already read our offer
                # and are selecting a protocol. This means they read
                # the offer in a prior Tread.
                self._phase = "p9any_ok"
                self._read_buf = self.P9ANY_OK
                return len(data)
            else:
                # Raw token mode (ninepfuse and simple clients).
                self._phase = "raw"
                self._token_buf = data
                return len(data)

        elif self._phase == "p9any_offer":
            # Client selecting protocol after reading the offer
            text = data.rstrip(b'\0').decode('utf-8', errors='replace').strip()
            if "pass" in text.lower():
                self._phase = "p9any_ok"
                self._read_buf = self.P9ANY_OK
            else:
                self._phase = "done"
                self.error = f"unsupported protocol: {text}"
                self._read_buf = f"err: {self.error}\n".encode()
            return len(data)

        elif self._phase == "p9any_ok":
            # Client writing password after reading "OK"
            self._phase = "pass_wait"
            self._token_buf = data
            return len(data)

        elif self._phase == "pass_wait":
            self._token_buf += data
            return len(data)

        elif self._phase == "raw":
            self._token_buf += data
            return len(data)

        elif self._phase == "done":
            return len(data)

        return len(data)

    def get_token(self) -> str:
        """Return the accumulated token/password as a string (stripped)."""
        return self._token_buf.rstrip(b'\0').decode('utf-8', errors='replace').strip()

    def read(self, offset: int, count: int) -> bytes:
        """
        Handle a read from the auth fid.

        In p9any mode the first read returns the protocol offer.
        Subsequent reads return status after authentication.
        In raw mode reads return auth status.
        """
        if self._phase == "init":
            # First read before any write — send p9any offer (factotum path)
            self._phase = "p9any_offer"
            self._read_buf = self.P9ANY_OFFER
            return self._read_buf[offset:offset + count]

        if self._read_buf:
            return self._read_buf[offset:offset + count]

        # After authentication attempt
        if self.authenticated:
            status = b"ok\n"
        elif self.error:
            status = f"err: {self.error}\n".encode('utf-8')
        elif not self._token_buf:
            return b""
        else:
            status = b"err: not authenticated\n"

        return status[offset:offset + count]

    def clear_read_buf(self):
        """Clear the read buffer after client has consumed it."""
        self._read_buf = b""

    @property
    def age(self) -> float:
        """Seconds since this auth fid was created."""
        return time.time() - self._created_at


class AuthManager:
    """
    Manages 9P authentication for a Server9P-based server.

    Holds a set of valid tokens and validates client-submitted tokens
    against them using constant-time comparison. If no secrets are
    configured, auth is disabled (backward compatible with unauthenticated
    clients).

    Auth fids expire after auth_timeout seconds (default 60).
    """

    # Env var names — distinct from riomux so rio can have its own tokens.
    ENV_TOKENS  = "RIO_AUTH_TOKENS"
    ENV_FILE    = "RIO_AUTH_FILE"
    ENV_TIMEOUT = "RIO_AUTH_TIMEOUT"

    def __init__(
        self,
        secrets: List[str] = None,
        secrets_file: str = None,
        auth_timeout: float = 60.0,
    ):
        self._secrets: Set[str] = set()
        self._auth_timeout = auth_timeout

        if secrets:
            for s in secrets:
                s = s.strip()
                if s:
                    self._secrets.add(s)

        if secrets_file:
            self._load_secrets_file(secrets_file)

        env_tokens = os.environ.get(self.ENV_TOKENS, '')
        if env_tokens:
            for token in env_tokens.split(','):
                token = token.strip()
                if token:
                    self._secrets.add(token)

        if self._secrets:
            logger.info(f"Auth enabled with {len(self._secrets)} token(s)")
        else:
            logger.info("Auth disabled (no secrets configured)")

    @classmethod
    def from_env(cls) -> 'AuthManager':
        """
        Build an AuthManager from environment variables:
            RIO_AUTH_TOKENS  — comma-separated tokens
            RIO_AUTH_FILE    — path to a secrets file
            RIO_AUTH_TIMEOUT — auth fid timeout in seconds
        """
        secrets_file = os.environ.get(cls.ENV_FILE)
        timeout = float(os.environ.get(cls.ENV_TIMEOUT, '60'))
        return cls(secrets_file=secrets_file, auth_timeout=timeout)

    @property
    def enabled(self) -> bool:
        """Whether auth is active (secrets are configured)."""
        return len(self._secrets) > 0

    @property
    def secret_count(self) -> int:
        return len(self._secrets)

    def _load_secrets_file(self, path: str):
        """Load tokens from a file, one per line. Lines starting with # are comments."""
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self._secrets.add(line)
            logger.info(f"Loaded secrets from {path}")
        except FileNotFoundError:
            logger.warning(f"Secrets file not found: {path}")
        except Exception as e:
            logger.error(f"Error reading secrets file {path}: {e}")

    def validate_token(self, token: str) -> bool:
        """
        Validate a token against the known secrets using constant-time
        comparison. Returns True if the token matches any configured secret.
        """
        if not self._secrets:
            return False

        token_bytes = token.encode('utf-8')

        for secret in self._secrets:
            secret_bytes = secret.encode('utf-8')
            if hmac.compare_digest(token_bytes, secret_bytes):
                return True

        return False

    def create_auth_fid(self, fid: int, uname: str, aname: str) -> AuthFid:
        """Create a new auth fid for a Tauth request."""
        return AuthFid(fid=fid, uname=uname, aname=aname)

    def authenticate(self, auth_fid: AuthFid) -> bool:
        """
        Attempt to authenticate an auth fid after the client has written
        a token. Updates the auth fid state. Returns True on success.
        """
        if auth_fid.age > self._auth_timeout:
            auth_fid.error = "auth fid expired"
            auth_fid.authenticated = False
            logger.warning(
                f"Auth fid {auth_fid.fid} expired for user '{auth_fid.uname}' "
                f"(age={auth_fid.age:.1f}s, timeout={self._auth_timeout}s)"
            )
            return False

        token = auth_fid.get_token()

        if not token:
            auth_fid.error = "no token provided"
            auth_fid.authenticated = False
            return False

        if self.validate_token(token):
            auth_fid.authenticated = True
            auth_fid.error = ""
            logger.info(f"Auth succeeded for user '{auth_fid.uname}' (fid={auth_fid.fid})")
            return True
        else:
            auth_fid.authenticated = False
            auth_fid.error = "invalid token"
            logger.warning(f"Auth failed for user '{auth_fid.uname}' (fid={auth_fid.fid})")
            return False

    def add_secret(self, token: str):
        token = token.strip()
        if token:
            self._secrets.add(token)
            logger.info(f"Added auth token (total: {len(self._secrets)})")

    def remove_secret(self, token: str) -> bool:
        token = token.strip()
        if token in self._secrets:
            self._secrets.discard(token)
            logger.info(f"Removed auth token (total: {len(self._secrets)})")
            return True
        return False


class AuthContext:
    """
    Per-connection auth state.

    Each connection gets an AuthContext that tracks its auth fids and
    authenticated sessions. This is the interface the 9P server uses.
    """

    def __init__(self, manager: AuthManager):
        self._manager = manager
        self._auth_fids: Dict[int, AuthFid] = {}
        # Set of authenticated unames — once a user authenticates on
        # this connection, subsequent Tattach with the same uname is
        # allowed without re-auth.
        self._authenticated_sessions: Set[str] = set()

    @property
    def auth_required(self) -> bool:
        """Whether this connection requires auth (delegates to manager)."""
        return self._manager.enabled

    @property
    def manager(self) -> AuthManager:
        return self._manager

    def handle_tauth(self, fid: int, uname: str, aname: str) -> AuthFid:
        """
        Handle a Tauth request. Creates and returns an AuthFid.
        The caller should respond with Rauth containing the aqid.
        """
        auth_fid = self._manager.create_auth_fid(fid, uname, aname)
        self._auth_fids[fid] = auth_fid
        return auth_fid

    def handle_auth_write(self, fid: int, data: bytes) -> int:
        """
        Handle a Twrite to an auth fid. Routes data through the AuthFid
        state machine. Attempts authentication when a token/password has
        been fully received. Returns the number of bytes written.
        """
        auth_fid = self._auth_fids.get(fid)
        if auth_fid is None:
            raise ValueError(f"Unknown auth fid {fid}")

        count = auth_fid.write(data)

        # Attempt authentication when we have token data in a terminal phase.
        if auth_fid._phase in ("raw", "pass_wait") and auth_fid._token_buf:
            self._manager.authenticate(auth_fid)
            if auth_fid.authenticated:
                auth_fid._read_buf = b"ok\n"
            else:
                auth_fid._read_buf = f"err: {auth_fid.error}\n".encode()
            auth_fid._phase = "done"

        return count

    def handle_auth_read(self, fid: int, offset: int, count: int) -> bytes:
        """Handle a Tread on an auth fid. Returns the auth status / offer."""
        auth_fid = self._auth_fids.get(fid)
        if auth_fid is None:
            raise ValueError(f"Unknown auth fid {fid}")

        return auth_fid.read(offset, count)

    def handle_auth_clunk(self, fid: int):
        """Handle clunk of an auth fid."""
        auth_fid = self._auth_fids.pop(fid, None)
        if auth_fid and auth_fid.authenticated:
            # Remember this session as authenticated
            self._authenticated_sessions.add(auth_fid.uname)

    def is_auth_fid(self, fid: int) -> bool:
        """Check if a fid is an auth fid."""
        return fid in self._auth_fids

    def get_auth_fid(self, fid: int) -> Optional[AuthFid]:
        return self._auth_fids.get(fid)

    def check_attach(self, afid: int, uname: str) -> Optional[str]:
        """
        Check whether a Tattach should be allowed.

        Returns None if allowed, or an error string if denied.

        Rules:
            - If auth is disabled → always allowed
            - If user already authenticated on this connection → allowed
            - If afid references a valid, authenticated AuthFid with
              matching uname → allowed
            - Otherwise → denied
        """
        if not self._manager.enabled:
            return None  # Auth disabled, allow everything

        # Session reuse — already authenticated on this connection
        if uname in self._authenticated_sessions:
            return None

        if afid == NOFID:
            return "authentication required"

        auth_fid = self._auth_fids.get(afid)
        if auth_fid is None:
            return "invalid auth fid"

        if not auth_fid.authenticated:
            return "authentication failed"

        if auth_fid.uname != uname:
            return (
                f"auth fid user '{auth_fid.uname}' does not match "
                f"attach user '{uname}'"
            )

        # Remember session for subsequent attaches on this connection
        self._authenticated_sessions.add(uname)

        return None  # Allowed

    def cleanup(self):
        """Clean up all auth state for this connection."""
        self._auth_fids.clear()
        self._authenticated_sessions.clear()