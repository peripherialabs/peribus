#!/usr/bin/env python3
"""
LLMFS Server - LLM capabilities as a filesystem

Usage:
    python -m llmfs.main [options]

Options:
    --provider NAME    Default provider (claude, openai, gemini, groq, openrouter)
    --port PORT        9P server port (default: 5640)
    --host HOST        9P server host (default: 0.0.0.0)
    --unix PATH        Unix socket path (instead of TCP)
    --debug            Enable debug logging
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ninep.server import Server9P
from ninep.auth import AuthManager
from llmfs.filesystem import LLMFSRoot
from llmfs.providers import get_provider, list_providers


class LLMFSAuthManager(AuthManager):
    """
    AuthManager with LLMFS-specific env var names.
    
    Lets rio and llmfs have their own tokens (RIO_AUTH_TOKENS vs
    LLMFS_AUTH_TOKENS) without colliding, while sharing the same
    auth machinery.
    """
    ENV_TOKENS  = "LLMFS_AUTH_TOKENS"
    ENV_FILE    = "LLMFS_AUTH_FILE"
    ENV_TIMEOUT = "LLMFS_AUTH_TIMEOUT"


class LLMFSServer:
    """LLMFS as a standalone server"""
    
    def __init__(self, provider_name: str = None, auth_manager=None):
        # Initialize provider
        provider = None
        if provider_name:
            provider = get_provider(provider_name)
        
        # Create filesystem
        self.filesystem = LLMFSRoot(provider=provider)
        self.filesystem.register_function("write_code_to_scene", self.write_scene_parse)
        
        # Create 9P server (with optional auth)
        self.auth_manager = auth_manager
        self.server = Server9P(self.filesystem, auth_manager=auth_manager)
        
        self._running = False

    def write_scene_parse(self, content: str):
        """Writes content to the specific scene parse path."""
        try:
            import os
            # Ensure the directory exists
            os.makedirs("/n/rioa/scene", exist_ok=True)
            with open("/n/rioa/scene/parse", "w") as f:
                f.write(content)
            return {"status": "success", "path": "/n/rioa/scene/parse"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    
    async def start_tcp(self, host: str = '0.0.0.0', port: int = 5640):
        """Start TCP server"""
        self._running = True
        
        print(f"LLMFS server starting...")
        print(f"  Provider: {self.filesystem.provider.name}")
        print(f"  Default model: {self.filesystem.provider.default_model}")
        print(f"  Listening on: {host}:{port}")
        if self.auth_manager and self.auth_manager.enabled:
            print(f"  Auth: enabled ({len(self.auth_manager._secrets)} token(s))")
        else:
            print(f"  Auth: disabled")
        print()
        print(f"Mount with: 9pfuse tcp!localhost!{port} /mnt/llm")
        print()
        print("Example usage:")
        print("  echo 'new claude' > /mnt/llm/ctl")
        print("  echo 'model any-model-string' > /mnt/llm/claude/ctl")
        print("  echo 'Hello!' > /mnt/llm/claude/input")
        print("  cat /mnt/llm/claude/output")
        print()
        
        await self.server.serve_tcp(host, port)
    
    async def start_unix(self, path: str):
        """Start Unix socket server"""
        self._running = True
        
        print(f"LLMFS server starting...")
        print(f"  Provider: {self.filesystem.provider.name}")
        print(f"  Socket: {path}")
        print()
        
        await self.server.serve_unix(path)
    
    async def stop(self):
        """Stop the server"""
        self._running = False
        await self.server.stop()


async def main():
    parser = argparse.ArgumentParser(
        description="LLMFS - LLM capabilities as a filesystem"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=list_providers(),
        help="Default LLM provider"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5640,
        help="TCP port (default: 5640)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="TCP host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--unix", "-u",
        metavar="PATH",
        help="Unix socket path (instead of TCP)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--auth-token",
        action="append",
        metavar="TOKEN",
        help="Auth token (repeatable). Enables 9P token auth. "
             "Also reads LLMFS_AUTH_TOKENS (comma-separated)."
    )
    parser.add_argument(
        "--auth-file",
        metavar="PATH",
        help="Path to a file with auth tokens (one per line, # comments). "
             "Also reads LLMFS_AUTH_FILE."
    )
    parser.add_argument(
        "--auth-timeout",
        type=float,
        default=60.0,
        help="Auth fid timeout in seconds (default: 60)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    
    # Build auth manager. CLI args take precedence; LLMFS_AUTH_* env
    # vars are consulted by LLMFSAuthManager. Zero secrets → disabled,
    # backward-compatible with unauthed clients.
    auth_file = args.auth_file or os.environ.get(LLMFSAuthManager.ENV_FILE)
    auth_manager = LLMFSAuthManager(
        secrets=args.auth_token,
        secrets_file=auth_file,
        auth_timeout=args.auth_timeout,
    )
    
    # Create server
    try:
        server = LLMFSServer(
            provider_name=args.provider,
            auth_manager=auth_manager,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure you have API keys set:", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY=...", file=sys.stderr)
        print("  export OPENAI_API_KEY=...", file=sys.stderr)
        print("  export GEMINI_API_KEY=...", file=sys.stderr)
        sys.exit(1)
    
    # Handle signals.
    #
    # The previous version did `loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))`
    # which is a fire-and-forget on a loop that's already about to die: by
    # the time the create_task runs, run_until_complete has raised
    # CancelledError into start_tcp(). The result is a noisy traceback and
    # a serve_forever() that never gets a clean stop() call, so active
    # connections aren't shut down — that contributed to the multi-minute
    # Ctrl-C hang.
    #
    # Instead: race the server task against a stop event set by SIGINT/SIGTERM.
    # Whichever finishes first wins, then we explicitly call server.stop()
    # to close active connections and wait for the serve task to wind down.
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    if sys.platform != 'win32':
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

    # Start server
    try:
        if args.unix:
            serve_task = asyncio.create_task(server.start_unix(args.unix))
        else:
            serve_task = asyncio.create_task(server.start_tcp(args.host, args.port))

        stop_task = asyncio.create_task(stop_event.wait())

        done, pending = await asyncio.wait(
            {serve_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Either the server died on its own or we got a signal. Either way:
        # tell the server to actually stop (close connections), then await
        # the serve task so its finally-clauses can run. Suppress
        # CancelledError because that's expected on signal shutdown.
        print("\nShutting down...")
        await server.stop()

        if not serve_task.done():
            serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.exception(f"Server error during shutdown: {e}")

        # Cancel the stop_task too if it's still pending (signal never came)
        if not stop_task.done():
            stop_task.cancel()
            try:
                await stop_task
            except asyncio.CancelledError:
                pass

    except Exception as e:
        logging.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())