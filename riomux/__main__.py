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
import signal
import sys
from typing import Dict, Tuple

from .server import MuxServer


def parse_backend(spec: str) -> Tuple[str, str, int]:
    """
    Parse a backend spec: name=host:port
    Returns (name, host, port).
    """
    if '=' not in spec:
        raise ValueError(f"Invalid backend spec '{spec}'. Expected name=host:port")
    
    name, addr = spec.split('=', 1)
    name = name.strip()
    
    if ':' not in addr:
        raise ValueError(f"Invalid address '{addr}'. Expected host:port")
    
    host, port_str = addr.rsplit(':', 1)
    host = host.strip()
    port = int(port_str.strip())
    
    return name, host, port


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
    
    # Parse backends
    backends: Dict[str, Tuple[str, int]] = {}
    for spec in args.backend:
        try:
            name, host, port = parse_backend(spec)
            backends[name] = (host, port)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    if not backends:
        print("Error: at least one --backend is required", file=sys.stderr)
        sys.exit(1)
    
    # Print startup info
    print(f"riomux — 9P Multiplexer")
    print(f"  Listening: {args.host}:{args.port}")
    print(f"  Backends:")
    for name, (host, port) in backends.items():
        print(f"    {name} → {host}:{port}")
    print(f"  Mount: 9pfuse 127.0.0.1:{args.port} /n/mux")
    print()
    
    # Create server
    server = MuxServer(backends=backends)
    
    # Run
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    def shutdown(sig):
        print(f"\nShutting down (signal {sig})...")
        loop.create_task(server.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown, sig)
    
    try:
        loop.run_until_complete(server.serve(host=args.host, port=args.port))
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(server.stop())
        loop.close()


if __name__ == '__main__':
    main()