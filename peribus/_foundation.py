"""
peribus._foundation — concatenation of: embeddings.py, identity.py, trust.py

This is a build artefact. The original module names live as section
banners below so `grep "^# ===="` jumps to each one.
"""

from __future__ import annotations


# ============================================================================
# embeddings.py
# ----------------------------------------------------------------------------
"""
peribus.embeddings — turn content into vectors locally

Three things live here:
  1. The local embedder (whatever model is in /n/llm/embed/, or a fallback).
  2. The identity vector — an EMA of recent activity vectors. This is your
     "current self" as the rhizome sees it. It drifts as you work.
  3. Cosine similarity helpers and a compact "vector sketch" used in gossip.

The embedder interface is async because real models block, and we want
peribusd to stay responsive while embedding a chunk of text.
"""


import asyncio
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol


# Default vector width. If a real model produces a different size, the daemon
# adapts to whatever the embedder returns — this is just a starting point.
DEFAULT_DIM = 384

# How many dimensions the gossip "sketch" uses. Small enough to fit many
# of these in a single UDP packet.
SKETCH_DIM = 32


class Embedder(Protocol):
    """Anything with an `embed(text)` async method that returns a vector."""

    async def embed(self, text: str) -> List[float]: ...

    @property
    def dim(self) -> int: ...


# ---------------------------------------------------------------------------
# Fallback embedder: no model, just a deterministic hash-projection.
# Useful for development, demos, and graceful degradation when /n/llm/embed
# isn't populated. The vectors it produces are stable and have decent
# locality for similar text — not great, but enough to wire up the system.
# ---------------------------------------------------------------------------

class HashEmbedder:
    """Deterministic embedder for development. Replace with a real model in prod."""

    def __init__(self, dim: int = DEFAULT_DIM):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, text: str) -> List[float]:
        """
        Hash each token to a sign and bucket. This is a poor man's
        SimHash — close strings produce close vectors, far strings drift.
        Fast enough to never block.
        """
        import hashlib

        vec = [0.0] * self._dim
        # Tokenize on whitespace and 3-grams so short strings still get signal.
        tokens = text.lower().split()
        for n in (1, 2, 3):
            for i in range(len(tokens) - n + 1):
                tok = " ".join(tokens[i:i + n])
                h = hashlib.blake2s(tok.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(h[:4], "little") % self._dim
                sign = 1.0 if (h[4] & 1) else -1.0
                vec[bucket] += sign

        # L2-normalize so cosine == dot product.
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# Identity vector — your current "self" in vector space.
#
# Updated by the daemon whenever you write to scene/, share something, or
# read something on the feed. EMA so old activity decays gracefully.
# ---------------------------------------------------------------------------

@dataclass
class IdentityVector:
    """EMA-tracked identity vector with thread-safe updates."""

    dim: int
    alpha: float = 0.1                     # EMA weight for new observations
    vector: List[float] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self):
        if not self.vector:
            self.vector = [0.0] * self.dim

    async def observe(self, new_vec: List[float], weight: float = 1.0) -> None:
        """
        Move identity toward `new_vec`. `weight` lets us boost certain events
        (e.g. publishing your own work counts more than glancing at a post).
        """
        if len(new_vec) != self.dim:
            # Either we got a wrong-sized vector (skip) or this is the first
            # vector and dim was a guess; adapt on first real input.
            if all(v == 0.0 for v in self.vector):
                async with self._lock:
                    self.dim = len(new_vec)
                    self.vector = list(new_vec)
                return
            return

        a = min(1.0, max(0.0, self.alpha * weight))
        async with self._lock:
            self.vector = [
                (1 - a) * old + a * new
                for old, new in zip(self.vector, new_vec)
            ]
            # Re-normalize so cosine stays well-behaved.
            norm = math.sqrt(sum(v * v for v in self.vector)) or 1.0
            self.vector = [v / norm for v in self.vector]

    def snapshot(self) -> List[float]:
        return list(self.vector)


# ---------------------------------------------------------------------------
# Cosine similarity + vector serialization
# ---------------------------------------------------------------------------

def cosine(a: List[float], b: List[float]) -> float:
    """
    Cosine similarity. Returns 0.0 if either vector is zero or sizes mismatch.
    Vectors are assumed to be L2-normalized; if not, this still works but
    becomes a dot product instead of true cosine.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # Clamp to handle floating-point drift past ±1.
    if dot > 1.0:
        return 1.0
    if dot < -1.0:
        return -1.0
    return dot


def pack_vector(vec: List[float]) -> bytes:
    """Pack a vector as little-endian float32. Used for binary file reads and gossip."""
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(data: bytes) -> List[float]:
    """Inverse of pack_vector."""
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data[:n * 4]))


def make_sketch(vec: List[float], sketch_dim: int = SKETCH_DIM) -> List[float]:
    """
    Produce a low-dimensional sketch of a vector for cheap gossip.

    Uses random projection seeded by a constant so all nodes produce
    compatible sketches. This is a JL-transform: distances in sketch space
    approximate distances in the original space.
    """
    import random

    if len(vec) <= sketch_dim:
        return list(vec)

    rng = random.Random(0xC0DECAFE)  # MUST be the same across all nodes
    out = [0.0] * sketch_dim
    for i, v in enumerate(vec):
        # Re-seed deterministically per source dim.
        rng.seed(0xC0DECAFE ^ i)
        for j in range(sketch_dim):
            out[j] += v * (rng.random() * 2 - 1)

    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / norm for v in out]


# ---------------------------------------------------------------------------
# Embedder loader — looks for a real model in /n/llm/embed, falls back.
# ---------------------------------------------------------------------------

def load_embedder(llm_mount: Optional[str] = "/n/llm") -> Embedder:
    """
    Try to use whatever embedder lives in the local llmfs. If nothing
    is there or the import fails, fall back to the hash embedder.

    The contract for a real llmfs embedder is: a Python module at
    /n/llm/embed/embedder.py exposing `class Embedder` with the
    Embedder protocol above. We don't import it eagerly — that's
    deferred until first use, so peribusd can start without llmfs.
    """
    if llm_mount:
        candidate = Path(llm_mount) / "embed" / "embedder.py"
        if candidate.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "peribus_embedder_local", candidate
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "Embedder"):
                    return mod.Embedder()
            except Exception as e:
                # Log via print so we don't depend on a logger at import time.
                print(f"[peribus] llmfs embedder load failed ({e}); using fallback")

    return HashEmbedder()

# ============================================================================
# identity.py
# ----------------------------------------------------------------------------
"""
peribus.identity — who you are on the rhizome

Your identity is an Ed25519 keypair. Your NodeID is the BLAKE3 hash of
your public key, base32-encoded (lowercase, no padding) — short, URL-safe,
and unforgeable. Other nodes verify your posts by checking signatures
against your published pubkey.

Keys live in ~/.peribus/identity/ — kept out of the synthetic filesystem
on purpose. The synthetic /n/peribus/identity/nodeid is read-only and
just exposes the public NodeID.
"""


import base64
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# We keep the crypto deps soft — if they're missing, peribusd refuses
# to start with a clear error, rather than crashing somewhere downstream.
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    _HAVE_CRYPTO = True
except ImportError:
    _HAVE_CRYPTO = False


def _b32(data: bytes) -> str:
    """RFC4648 base32 lowercase, no padding. Filesystem-safe."""
    return base64.b32encode(data).decode("ascii").lower().rstrip("=")


def _nodeid_hash(data: bytes) -> bytes:
    """
    Hash function for NodeID derivation.

    DELIBERATELY SHA-256 ONLY — never blake3. Two peers must compute the
    same NodeID from the same pubkey, and we cannot rely on every machine
    having the same optional libraries installed. If machine A has the
    `blake3` package and machine B doesn't, an "is this peer who they
    say they are" check would compare a blake3 digest against a sha256
    digest, which always disagrees, and every cross-machine connection
    gets dropped at hello with a "pubkey mismatch" warning.

    Don't switch this to anything else without coordinating across all
    nodes simultaneously.
    """
    return hashlib.sha256(data).digest()


@dataclass
class Identity:
    """A node's cryptographic identity."""

    private_key: "Ed25519PrivateKey"
    public_key: "Ed25519PublicKey"
    nodeid: str  # base32 of hash(pubkey_bytes), 26 chars

    @classmethod
    def load_or_create(cls, identity_dir: Optional[Path] = None) -> "Identity":
        """
        Load identity from disk, or generate a fresh one if none exists.

        Storage layout:
            ~/.peribus/identity/
                ed25519.key       — private key (PEM, mode 0600)
                ed25519.pub       — public key (PEM)
                nodeid            — cached NodeID string
        """
        if not _HAVE_CRYPTO:
            raise RuntimeError(
                "peribus requires `cryptography` — install with: "
                "pip install cryptography"
            )

        if identity_dir is None:
            identity_dir = Path.home() / ".peribus" / "identity"
        identity_dir.mkdir(parents=True, exist_ok=True)

        priv_path = identity_dir / "ed25519.key"

        if priv_path.exists():
            with priv_path.open("rb") as f:
                priv = serialization.load_pem_private_key(f.read(), password=None)
            if not isinstance(priv, Ed25519PrivateKey):
                raise RuntimeError(f"Bad key type at {priv_path}: {type(priv)}")
            pub = priv.public_key()
        else:
            priv = Ed25519PrivateKey.generate()
            pub = priv.public_key()

            # Write private key with restrictive perms.
            priv_pem = priv.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            priv_path.write_bytes(priv_pem)
            os.chmod(priv_path, 0o600)

            pub_pem = pub.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            (identity_dir / "ed25519.pub").write_bytes(pub_pem)

        # NodeID is hash(raw pubkey bytes). Stable across restarts.
        pub_raw = pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        nodeid = _b32(_nodeid_hash(pub_raw))[:26]

        # Cache for human inspection.
        (identity_dir / "nodeid").write_text(nodeid + "\n")

        return cls(private_key=priv, public_key=pub, nodeid=nodeid)

    def sign(self, data: bytes) -> bytes:
        """Sign data with the node's private key."""
        return self.private_key.sign(data)

    def public_key_bytes(self) -> bytes:
        """Raw 32-byte Ed25519 public key, for sharing on the wire."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )


def verify_signature(pubkey_bytes: bytes, data: bytes, signature: bytes) -> bool:
    """Verify a signature against raw 32-byte pubkey bytes. Returns False on any failure."""
    if not _HAVE_CRYPTO:
        return False
    try:
        pub = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
        pub.verify(signature, data)
        return True
    except Exception:
        return False


def nodeid_from_pubkey(pubkey_bytes: bytes) -> str:
    """Compute the NodeID for a given raw pubkey. Used to verify peer claims."""
    return _b32(_nodeid_hash(pubkey_bytes))[:26]

# ============================================================================
# trust.py
# ----------------------------------------------------------------------------
"""
peribus.trust — contacts and invitations

Since the feed is public ("everyone the same"), trust isn't about gating
content. It's about two narrower problems:

  1. Contacts. You want a stable list of NodeIDs you actually know
     (relatives, friends), separate from the firehose of strangers
     surfaced by rendezvous. The contacts list survives across restarts,
     gets shown specially in the UI, and is used to prefer direct dials
     over relay paths.

  2. Bootstrapping a connection with someone specific. mDNS only finds
     LAN peers, and rendezvous queries are vector-ranked — so if your
     uncle's vector doesn't resemble yours, he might not surface in
     queries even when both of you are online. Invitations solve this:
     a small signed token (a peribus:// URL) that says "this NodeID is
     someone I know, dial them directly when you see them". You send
     the URL out-of-band (text message, signal, email), the receiver's
     daemon imports it, and from then on the two of you find each
     other via rendezvous-by-NodeID even if your vectors never align.

The crypto is straightforward: an invite is a signed JSON payload.
The signer asserts "I (NodeID X, pubkey P) issued this invite for
(NodeID Y) at time T". The receiver verifies the signature and adds
NodeID Y to their contacts. From that point on they ask the rendezvous
server "do you know about Y?" on a faster cadence than vector-based
queries, and accept punch_requests from Y without resonance gating.

Storage:
    ~/.peribus/contacts.json     — contacts list, JSON
    Invites are not stored on disk — they're transient URLs.

The wire format:
    peribus://invite/v1?from=<nodeid>&to=<nodeid>&pk=<b64>&sig=<b64>&exp=<unix>

`from` is the issuer's NodeID, `pk` is the issuer's pubkey (so the
receiver can verify without already knowing them), `sig` is the
issuer's Ed25519 signature over the canonical body, `exp` is an
expiry timestamp. Receivers reject invites past expiry.
"""


import base64
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, quote, urlparse

logger = logging.getLogger(__name__)


INVITE_VERSION = "v1"
INVITE_DEFAULT_TTL_S = 7 * 24 * 3600   # one week
INVITE_MAX_TTL_S = 90 * 24 * 3600      # hard cap; stale invites are bad hygiene


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------

@dataclass
class Contact:
    """One known peer in the user's address book."""
    nodeid: str
    label: str = ""              # human alias ("Mom", "uncle bob")
    pubkey_b64: str = ""         # remembered so we can verify even before first meeting
    added_at: float = 0.0
    introduced_by: str = ""      # nodeid of whoever issued the invite, "" if self-added

    def to_json(self) -> dict:
        return {
            "nodeid": self.nodeid,
            "label": self.label,
            "pubkey_b64": self.pubkey_b64,
            "added_at": self.added_at,
            "introduced_by": self.introduced_by,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Contact":
        return cls(
            nodeid=d["nodeid"],
            label=d.get("label", ""),
            pubkey_b64=d.get("pubkey_b64", ""),
            added_at=float(d.get("added_at", 0.0)),
            introduced_by=d.get("introduced_by", ""),
        )


class ContactBook:
    """
    Persistent contacts list. Loaded from disk at daemon start, written
    on every mutation. Small enough that we don't bother with a real
    database.
    """

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            path = Path.home() / ".peribus" / "contacts.json"
        self.path = path
        self._contacts: Dict[str, Contact] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            for entry in data.get("contacts", []):
                c = Contact.from_json(entry)
                self._contacts[c.nodeid] = c
        except Exception as e:
            logger.warning(f"contacts: failed to load {self.path}: {e}")

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {"contacts": [c.to_json() for c in self._contacts.values()]}
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self.path)
        except Exception as e:
            logger.warning(f"contacts: failed to save {self.path}: {e}")

    def add(self, contact: Contact) -> None:
        if contact.added_at == 0.0:
            contact.added_at = time.time()
        self._contacts[contact.nodeid] = contact
        self._save()

    def remove(self, nodeid: str) -> bool:
        if nodeid in self._contacts:
            del self._contacts[nodeid]
            self._save()
            return True
        return False

    def get(self, nodeid: str) -> Optional[Contact]:
        return self._contacts.get(nodeid)

    def all(self) -> List[Contact]:
        return list(self._contacts.values())

    def is_contact(self, nodeid: str) -> bool:
        return nodeid in self._contacts

    def label_for(self, nodeid: str) -> str:
        c = self._contacts.get(nodeid)
        return c.label if c and c.label else nodeid


# ---------------------------------------------------------------------------
# Invitations
# ---------------------------------------------------------------------------

@dataclass
class Invite:
    """A signed introduction. Either generated by us or received from someone."""
    from_nodeid: str
    to_nodeid: str
    issuer_pubkey: bytes      # raw 32-byte Ed25519 pubkey of the issuer
    expires_at: float         # unix seconds
    signature: bytes = b""    # over the canonical body

    def canonical_body(self) -> bytes:
        """Bytes that get signed. Order matters; never change without bumping version."""
        return f"peribus-invite/{INVITE_VERSION}|{self.from_nodeid}|{self.to_nodeid}|{int(self.expires_at)}".encode("utf-8")

    def to_url(self) -> str:
        params = [
            ("from", self.from_nodeid),
            ("to", self.to_nodeid),
            ("pk", base64.urlsafe_b64encode(self.issuer_pubkey).decode("ascii").rstrip("=")),
            ("sig", base64.urlsafe_b64encode(self.signature).decode("ascii").rstrip("=")),
            ("exp", str(int(self.expires_at))),
        ]
        qs = "&".join(f"{k}={quote(v, safe='')}" for k, v in params)
        return f"peribus://invite/{INVITE_VERSION}?{qs}"

    @classmethod
    def from_url(cls, url: str) -> "Invite":
        """Parse a peribus://invite/... URL. Raises ValueError on malformed input."""
        parsed = urlparse(url)
        if parsed.scheme != "peribus":
            raise ValueError(f"not a peribus URL: scheme={parsed.scheme!r}")
        # urllib doesn't parse "peribus://invite/v1" cleanly because of the
        # custom scheme — netloc gets "invite" and path gets "/v1". Handle both shapes.
        kind = parsed.netloc or parsed.path.lstrip("/").split("/", 1)[0]
        if kind != "invite":
            raise ValueError(f"not an invite URL: kind={kind!r}")
        # Version is the last path segment.
        path_parts = [p for p in parsed.path.split("/") if p]
        version = path_parts[-1] if path_parts else ""
        if version != INVITE_VERSION:
            raise ValueError(f"unsupported invite version: {version!r}")

        q = parse_qs(parsed.query)
        try:
            from_nodeid = q["from"][0]
            to_nodeid = q["to"][0]
            pk_b64 = q["pk"][0]
            sig_b64 = q["sig"][0]
            expires_at = float(q["exp"][0])
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"missing/invalid invite field: {e}")

        # urlsafe_b64 may be missing padding; restore it.
        def _b64(s: str) -> bytes:
            pad = "=" * (-len(s) % 4)
            return base64.urlsafe_b64decode(s + pad)

        return cls(
            from_nodeid=from_nodeid,
            to_nodeid=to_nodeid,
            issuer_pubkey=_b64(pk_b64),
            expires_at=expires_at,
            signature=_b64(sig_b64),
        )

    def is_expired(self, now: Optional[float] = None) -> bool:
        return (now or time.time()) >= self.expires_at


def make_invite(
    issuer_identity,         # Identity from peribus.identity
    to_nodeid: str,
    ttl_s: float = INVITE_DEFAULT_TTL_S,
) -> Invite:
    """
    Create a signed invite from `issuer_identity` for `to_nodeid`.

    The issuer asserts "I know this person, please add them to your
    contacts". It does not give the receiver any extra access — it
    just provides a verified introduction.

    Cap ttl at INVITE_MAX_TTL_S to avoid forever-tokens that age into
    being a security problem if the issuer's machine gets compromised.
    """
    ttl_s = min(max(60.0, ttl_s), INVITE_MAX_TTL_S)
    invite = Invite(
        from_nodeid=issuer_identity.nodeid,
        to_nodeid=to_nodeid,
        issuer_pubkey=issuer_identity.public_key_bytes(),
        expires_at=time.time() + ttl_s,
    )
    invite.signature = issuer_identity.sign(invite.canonical_body())
    return invite


def verify_invite(invite: Invite) -> Optional[str]:
    """
    Verify an invite. Returns None on success, or a human-readable
    rejection reason. Checks performed:

      1. Issuer pubkey matches issuer NodeID (so you can't claim to be
         someone you're not).
      2. Signature verifies against issuer pubkey.
      3. Not expired.
    """
    from peribus._foundation import nodeid_from_pubkey, verify_signature

    if not invite.from_nodeid or not invite.to_nodeid:
        return "missing nodeid"
    if not invite.issuer_pubkey or not invite.signature:
        return "missing pubkey or signature"
    if nodeid_from_pubkey(invite.issuer_pubkey) != invite.from_nodeid:
        return "issuer pubkey does not match issuer nodeid"
    if invite.is_expired():
        return "expired"
    if not verify_signature(invite.issuer_pubkey, invite.canonical_body(), invite.signature):
        return "signature does not verify"
    return None