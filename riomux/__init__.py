"""
riomux — 9P2000 Multiplexer

Presents multiple backend 9P servers under a single mount point.
Operates at the wire level to preserve all semantics (blocking reads,
clunk triggers, streaming, flush).

    /n/mux/
    ├── rio/      → backend rio server
    └── llm/      → backend llm server

Usage:
    python -m riomux --port 5642 --backend rio=127.0.0.1:5641 --backend llm=127.0.0.1:5640
    9pfuse 127.0.0.1:5642 /n/mux

Then:
    cat /n/mux/llm/agents/claude/output          # blocks until output
    echo 'hello' > /n/mux/llm/agents/claude/input # triggers generation
    while true; do cat /n/mux/llm/agents/claude/output > /n/mux/rio/scene/parse; done
"""

from .server import MuxServer
from .mux import MuxConnection
from .backend import BackendConnection

__all__ = ['MuxServer', 'MuxConnection', 'BackendConnection']