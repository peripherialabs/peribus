"""
riomux — 9P2000 Multiplexer CLI

Usage:
    python -m riomux --port 5642 \
        --backend rio=127.0.0.1:5641 \
        --backend llm=127.0.0.1:5640

Then mount:
    9pfuse 127.0.0.1:5642 /n/mux

Filesystem:
    /n/mux/
    ├── rio/      → all of rio's filesystem
    └── llm/      → all of llm's filesystem

Streaming / blocking / clunk semantics are preserved exactly:
    while true; do cat /n/mux/llm/agents/claude/output > /n/mux/rio/scene/parse; done
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from typing import Dict, Optional, Tuple

from .server import MuxServer
from .auth import AuthManager


def _load_token_from_file(path: str) -> str:
    """Read the first non-empty, non-comment line of a file as a token."""
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                return line
    raise ValueError(f"no token found in '{path}'")


def parse_backend(spec: str) -> Tuple[str, str, int, Optional[str]]:
    """
    Parse a backend spec. Accepted forms (same as /n/ctl):
    
        name=host:port
        name=host:port token=<token>
        name=host:port token=@<path>
        name=host:port:<token>            (colon form)
    
    Returns (name, host, port, token_or_None). The token is held only
    in memory and never echoed by riomux.
    """
    if '=' not in spec:
        raise ValueError(
            f"Invalid backend spec '{spec}'. Expected name=host:port"
        )
    
    name, rest = spec.split('=', 1)
    name = name.strip()
    rest = rest.strip()
    
    # Optional trailing 'token=<x>' / 'token=@<path>' (space-separated).
    # Only the first such kwarg is recognized; anything else errors.
    token: Optional[str] = None
    parts = rest.split()
    if len(parts) >= 2:
        addr = parts[0]
        for extra in parts[1:]:
            if extra.startswith("token="):
                raw = extra[len("token="):]
                if raw.startswith("@"):
                    token = _load_token_from_file(raw[1:])
                else:
                    token = raw
            else:
                raise ValueError(
                    f"Unknown option '{extra}' in backend spec '{spec}'. "
                    f"Supported: token=<value> | token=@<path>"
                )
    else:
        addr = rest
    
    if ':' not in addr:
        raise ValueError(f"Invalid address '{addr}'. Expected host:port")
    
    # Disambiguate host:port vs host:port:token by checking whether the
    # rsplit tail is numeric.
    host, last = addr.rsplit(':', 1)
    try:
        port = int(last)
    except ValueError:
        # Colon form: addr was host:port:token; the trailing piece is a token.
        if ':' not in host:
            raise ValueError(f"Invalid address '{addr}'. Expected host:port")
        colon_token = last
        host, port_str = host.rsplit(':', 1)
        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port '{port_str}' in '{addr}'")
        # Explicit token= kwarg wins over colon form if both given.
        if token is None:
            token = colon_token
    
    host = host.strip()
    return name, host, port, (token if token else None)


def _load_backend_tokens_env() -> Dict[str, str]:
    """
    Parse RIOMUX_BACKEND_TOKENS=name1:tok1,name2:tok2 → {name1: tok1, ...}.
    Useful for keeping tokens out of argv (which is visible in ps).
    """
    raw = os.environ.get("RIOMUX_BACKEND_TOKENS", "").strip()
    if not raw:
        return {}
    out: Dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or ':' not in entry:
            continue
        name, token = entry.split(':', 1)
        name = name.strip()
        token = token.strip()
        if name and token:
            out[name] = token
    return out


def main():
    parser = argparse.ArgumentParser(
        description="9P2000 Multiplexer — mounts multiple 9P servers under one namespace"
    )
    parser.add_argument(
        '--port', type=int, default=5642,
        help='Port to listen on (default: 5642)'
    )
    parser.add_argument(
        '--host', default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--backend', action='append', required=True,
        help='Backend spec: name=host:port (can be repeated)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--auth-token', action='append',
        help='Auth token (can be repeated). Enables auth.'
    )
    parser.add_argument(
        '--auth-file',
        help='Path to file with auth tokens (one per line)'
    )
    parser.add_argument(
        '--auth-timeout', type=float, default=60.0,
        help='Auth fid timeout in seconds (default: 60)'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.WARNING
    if args.verbose:
        level = logging.INFO
    if args.debug:
        level = logging.DEBUG
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Parse backends. Final shape: name → (host, port, token_or_None).
    backends: Dict[str, Tuple[str, int, Optional[str]]] = {}
    for spec in args.backend:
        try:
            name, host, port, token = parse_backend(spec)
            backends[name] = (host, port, token)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    if not backends:
        print("Error: at least one --backend is required", file=sys.stderr)
        sys.exit(1)
    
    # Overlay tokens from env var RIOMUX_BACKEND_TOKENS so they can
    # be supplied out-of-band (avoiding ps visibility). An entry here
    # only sets a token for an already-declared backend.
    env_tokens = _load_backend_tokens_env()
    for name, env_tok in env_tokens.items():
        if name in backends:
            host, port, existing = backends[name]
            # CLI/spec token wins; env fills in when CLI didn't supply one.
            backends[name] = (host, port, existing or env_tok)
        else:
            print(
                f"Warning: RIOMUX_BACKEND_TOKENS has token for unknown "
                f"backend '{name}' — ignoring",
                file=sys.stderr,
            )
    
    # Create auth manager
    auth_manager = AuthManager(
        secrets=args.auth_token,
        secrets_file=args.auth_file,
        auth_timeout=args.auth_timeout,
    )
    
    # Print startup info — never print tokens, only an auth marker.
    print(f"riomux — 9P Multiplexer")
    print(f"  Listening: {args.host}:{args.port}")
    if auth_manager.enabled:
        print(f"  Auth: enabled ({len(auth_manager._secrets)} token(s))")
    else:
        print(f"  Auth: disabled (no tokens configured)")
    print(f"  Backends:")
    for name, (host, port, token) in backends.items():
        marker = " (auth)" if token else ""
        print(f"    {name} → {host}:{port}{marker}")
    print(f"  Mount: 9pfuse 127.0.0.1:{args.port} /n/mux")
    print()
    
    # Create server
    server = MuxServer(backends=backends, auth_manager=auth_manager)

    # Run.
    #
    # The previous version installed a signal handler that did
    # `loop.create_task(server.stop())` — which races
    # `loop.run_until_complete(server.serve(...))`. By the time the task is
    # scheduled, the run_until_complete coroutine is already raising
    # CancelledError, so server.stop() never gets to close active
    # connections. That left the FUSE mount and downstream backends stuck
    # on dead reads, which is why Ctrl-C took ~2 minutes to clear.
    #
    # Now: run everything inside a single coroutine that races the serve
    # task against a stop event set by SIGINT/SIGTERM, then explicitly
    # awaits server.stop().
    async def run():
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

        serve_task = asyncio.create_task(
            server.serve(host=args.host, port=args.port)
        )
        stop_task = asyncio.create_task(stop_event.wait())

        done, pending = await asyncio.wait(
            {serve_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        print("\nShutting down (signal received)...")
        await server.stop()

        if not serve_task.done():
            serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.exception(f"Server error during shutdown: {e}")

        if not stop_task.done():
            stop_task.cancel()
            try:
                await stop_task
            except asyncio.CancelledError:
                pass

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        # asyncio.run() converts SIGINT into KeyboardInterrupt only if our
        # signal handler hasn't been installed yet (race during startup).
        # Treat it as a clean exit.
        pass


if __name__ == '__main__':
    main()