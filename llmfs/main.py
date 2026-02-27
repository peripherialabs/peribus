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
from llmfs.filesystem import LLMFSRoot
from llmfs.providers import get_provider, list_providers


class LLMFSServer:
    """LLMFS as a standalone server"""
    
    def __init__(self, provider_name: str = None):
        # Initialize provider
        provider = None
        if provider_name:
            provider = get_provider(provider_name)
        
        # Create filesystem
        self.filesystem = LLMFSRoot(provider=provider)
        self.filesystem.register_function("write_code_to_scene", self.write_scene_parse)
        
        # Create 9P server
        self.server = Server9P(self.filesystem)
        
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
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    
    # Create server
    try:
        server = LLMFSServer(provider_name=args.provider)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nMake sure you have API keys set:", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY=...", file=sys.stderr)
        print("  export OPENAI_API_KEY=...", file=sys.stderr)
        print("  export GEMINI_API_KEY=...", file=sys.stderr)
        sys.exit(1)
    
    # Handle signals
    loop = asyncio.get_event_loop()
    
    if sys.platform != 'win32':
        def signal_handler():
            print("\nShutting down...")
            asyncio.create_task(server.stop())
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)
    
    # Start server
    try:
        if args.unix:
            await server.start_unix(args.unix)
        else:
            await server.start_tcp(args.host, args.port)
    except Exception as e:
        logging.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())