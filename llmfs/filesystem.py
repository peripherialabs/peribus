"""
LLMFS Filesystem Root

This module defines the root filesystem structure for LLMFS.
It exposes LLM capabilities as a synthetic filesystem.

Directory structure:
    /n/llm/
    ├── ctl           # Global control
    ├── providers     # Single file: cat to see all providers + models
    ├── claude/       # Agent named 'claude'
    │   ├── ctl
    │   ├── input
    │   ├── output
    │   ├── history
    │   ├── config
    │   ├── system
    │   ├── rules
    │   ├── state
    │   └── errors
    ├── av/           # Gemini AudioVisual agent
    │   ├── ctl
    │   ├── ...
    ├── grok_av/      # Grok AudioVisual agent
    │   ├── ctl
    │   ├── ...
    ├── openai_av/    # OpenAI Realtime AudioVisual agent
    │   ├── ctl
    │   ├── ...
    └── ...
"""

import asyncio
import json
from typing import Dict, List, Optional

from core.files import SyntheticDir, SyntheticFile, CtlFile, CtlHandler
from core.types import FidState

from .providers import LLMProvider, get_provider, list_providers
from .agent import Agent
from .av_agent import AVAgent, AVConfig
from .av_grok_agent import GrokAVAgent, GrokAVConfig
from .av_openai_agent import OpenAIAVAgent, OpenAIAVConfig
from .ts_agent import TSAgent



class LLMFSCtlHandler(CtlHandler):
    """Control handler for LLMFS root"""
    
    def __init__(self, fs: 'LLMFSRoot'):
        self.fs = fs
    
    async def execute(self, command: str) -> Optional[str]:
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd == "provider":
            if arg:
                await self.fs.set_provider(arg)
                return f"Provider set to {arg}"
            return self.fs.provider.name
        
        elif cmd == "new":
            # Create new agent: new <name> [provider] [model]
            if not arg:
                raise ValueError("Usage: new <name> [provider] [model]")
            parts = arg.split()
            name = parts[0]
            provider = parts[1] if len(parts) > 1 else None
            model = parts[2] if len(parts) > 2 else None
            self.fs.create_agent(name, provider, model)
            return f"Agent '{name}' created"
        
        elif cmd == "av":
            # Create new AV agent: av <name> [voice] [video_mode]
            if not arg:
                raise ValueError("Usage: av <name> [voice] [video_mode]")
            parts = arg.split()
            name = parts[0]
            voice = parts[1] if len(parts) > 1 else "Aoede"
            video_mode = parts[2] if len(parts) > 2 else "none"
            self.fs.create_av_agent(name, voice=voice, video_mode=video_mode)
            return f"AV Agent '{name}' created"
        
        elif cmd == "grok":
            # Create new Grok AV agent: grok <name> [voice]
            if not arg:
                raise ValueError("Usage: grok <name> [voice]")
            parts = arg.split()
            name = parts[0]
            voice = parts[1] if len(parts) > 1 else "Ara"
            self.fs.create_grok_av_agent(name, voice=voice)
            return f"Grok AV Agent '{name}' created"
        
        elif cmd == "openai":
            # Create new OpenAI AV agent: openai <name> [voice] [model]
            if not arg:
                raise ValueError("Usage: openai <name> [voice] [model]")
            parts = arg.split()
            name = parts[0]
            voice = parts[1] if len(parts) > 1 else "marin"
            model = parts[2] if len(parts) > 2 else None
            self.fs.create_openai_av_agent(name, voice=voice, model=model)
            return f"OpenAI AV Agent '{name}' created"
        
        elif cmd == "ts":
            # Create new TS agent: ts <name> [voice]
            if not arg:
                raise ValueError("Usage: ts <name> [voice]")
            parts = arg.split()
            name = parts[0]
            voice = parts[1] if len(parts) > 1 else None
            self.fs.create_ts_agent(name, voice=voice)
            return f"TS Agent '{name}' created"
        
        elif cmd == "delete":
            if not arg:
                raise ValueError("Usage: delete <name>")
            self.fs.delete_agent(arg)
            return f"Agent '{arg}' deleted"
        
        elif cmd == "machine":
            # Register/unregister a mounted machine
            # machine add <name>   — register a machine
            # machine remove <name> — unregister a machine
            # machine list         — list machines
            if not arg:
                machines = self.fs.get_machines()
                return " ".join(machines) if machines else "(none)"
            
            sub_parts = arg.split(None, 1)
            sub_cmd = sub_parts[0].lower()
            sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""
            
            if sub_cmd == "add" and sub_arg:
                self.fs.add_machine(sub_arg)
                return f"Machine '{sub_arg}' registered"
            elif sub_cmd == "remove" and sub_arg:
                self.fs.remove_machine(sub_arg)
                return f"Machine '{sub_arg}' unregistered"
            elif sub_cmd == "list":
                machines = self.fs.get_machines()
                return " ".join(machines) if machines else "(none)"
            else:
                raise ValueError("Usage: machine add|remove|list <name>")
        
        else:
            raise ValueError(f"Unknown command: {cmd}. Available: provider, new, av, grok, openai, ts, delete, machine")
    
    async def get_status(self) -> bytes:
        lines = [
            f"provider {self.fs.provider.name}",
            f"agents {len(self.fs.agents)}",
            f"av_agents {len(self.fs.av_agents)}",
            f"grok_av_agents {len(self.fs.grok_av_agents)}",
            f"openai_av_agents {len(self.fs.openai_av_agents)}",
            f"ts_agents {len(self.fs.ts_agents)}",
            f"machines {' '.join(self.fs.get_machines()) or '(none)'}",
        ]
        return ("\n".join(lines) + "\n").encode()


class ProvidersFile(SyntheticFile):
    """
    Single file listing all available providers and their models.
    
    Reading returns a formatted list of all providers with their models.
    """
    
    def __init__(self):
        super().__init__("providers")
    
    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        lines = []
        for name in list_providers():
            try:
                provider = get_provider(name)
                models = provider.get_models()
                lines.append(f"{name}:")
                for model in models:
                    lines.append(f"  {model}")
            except Exception:
                lines.append(f"{name}: (not available)")
            lines.append("")
        
        data = ("\n".join(lines)).encode()
        return data[offset:offset + count]
    
    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("Providers file is read-only")


class LLMFSRoot(SyntheticDir):
    """
    Root of the LLMFS filesystem.
    
    Agents are created directly under the root directory alongside
    the ctl and providers files.
    
    Filesystem structure:
        /n/llm/
        ├── ctl           # Global control
        ├── providers     # File: cat to see all providers + models
        ├── claude/       # Agent named 'claude'
        │   ├── ctl
        │   ├── input
        │   ├── output
        │   ├── history
        │   ├── config
        │   ├── system
        │   ├── rules
        │   ├── state
        │   └── errors
        ├── av/           # Gemini AudioVisual agent
        │   ├── ctl
        │   ├── input
        │   ├── output
        │   ├── history
        │   ├── config
        │   ├── system
        │   ├── status
        │   └── errors
        ├── grok_av/      # Grok AudioVisual agent
        │   ├── ctl
        │   ├── ...
        ├── openai_av/    # OpenAI Realtime AudioVisual agent
        │   ├── ctl
        │   ├── input
        │   ├── OUTPUT
        │   ├── history
        │   ├── config
        │   ├── system
        │   ├── status
        │   ├── CODE
        │   ├── AUDIO
        │   ├── mic
        │   └── errors
        └── ...
    """
    
    # Reserved names that cannot be used for agents
    RESERVED_NAMES = {"ctl", "providers"}
    
    def __init__(self, provider: LLMProvider = None):
        super().__init__("")  # Root has empty name
        
        # Initialize provider
        if provider is None:
            # Try to get default provider
            for name in ["claude", "openai", "gemini"]:
                try:
                    provider = get_provider(name)
                    break
                except Exception:
                    continue
        
        if provider is None:
            raise ValueError("No LLM provider available. Set API keys in environment.")
        
        self.provider = provider
        
        # Text agents
        self.agents: Dict[str, Agent] = {}
        
        # Gemini AV agents
        self.av_agents: Dict[str, AVAgent] = {}
        
        # Grok AV agents
        self.grok_av_agents: Dict[str, GrokAVAgent] = {}
        
        # OpenAI AV agents
        self.openai_av_agents: Dict[str, OpenAIAVAgent] = {}
        
        # TS agents
        self.ts_agents: Dict[str, TSAgent] = {}
        
        # Global function registry for AV agents (shared by Gemini, Grok, and OpenAI)
        self.function_registry = {}
        
        # Machine registry: tracks mounted 9P machines (from riomux ctl)
        # Excludes "llm" (self). Used for auto-creating agent rules.
        self._machines: List[str] = []
        
        # Build filesystem tree
        self.add(CtlFile("ctl", LLMFSCtlHandler(self)))
        self.add(ProvidersFile())
    
    def _check_name(self, name: str):
        """Validate that an agent name doesn't conflict with reserved files"""
        if name in self.RESERVED_NAMES:
            raise ValueError(f"Name '{name}' is reserved (conflicts with {name} file)")
        if (name in self.agents or name in self.av_agents 
                or name in self.grok_av_agents or name in self.openai_av_agents
                or name in self.ts_agents):
            raise ValueError(f"Agent '{name}' already exists")
    
    async def set_provider(self, name: str):
        """Switch to a different provider"""
        self.provider = get_provider(name)
    
    def create_agent(
        self, 
        name: str, 
        provider_name: str = None,
        model: str = None, 
        system: str = None
    ) -> Agent:
        """Create a new text agent"""
        self._check_name(name)
        
        # Get provider
        provider = self.provider
        if provider_name:
            provider = get_provider(provider_name)
        
        # Create agent
        agent = Agent(
            name=name,
            provider=provider,
            route_manager=None,
            default_model=model
        )
        
        if system:
            agent.config.system = system
        
        self.agents[name] = agent
        self.add(agent)
        
        # Backlink so agent ctl can access machine registry
        agent._fs_root = self
        
        return agent
    
    def create_av_agent(
        self,
        name: str,
        voice: str = "Aoede",
        video_mode: str = "none",
        system: str = None
    ) -> AVAgent:
        """Create a new Gemini AudioVisual agent"""
        self._check_name(name)
        
        # Create AV agent
        agent = AVAgent(
            name=name,
            route_manager=None,
            function_registry=self.function_registry
        )
        
        agent.config.voice = voice
        agent.config.video_mode = video_mode
        
        if system:
            agent.config.system = system
        
        self.av_agents[name] = agent
        self.add(agent)
        
        return agent
    
    def create_grok_av_agent(
        self,
        name: str,
        voice: str = "Ara",
        system: str = None,
    ) -> GrokAVAgent:
        """Create a new Grok AudioVisual agent"""
        self._check_name(name)
        
        # Create Grok AV agent
        agent = GrokAVAgent(
            name=name,
            route_manager=None,
            function_registry=self.function_registry,
        )
        
        agent.config.voice = voice
        
        if system:
            agent.config.system = system
        
        self.grok_av_agents[name] = agent
        self.add(agent)
        
        return agent
    
    def create_openai_av_agent(
        self,
        name: str,
        voice: str = "marin",
        model: str = None,
        system: str = None,
    ) -> OpenAIAVAgent:
        """Create a new OpenAI Realtime AudioVisual agent"""
        self._check_name(name)
        
        # Create OpenAI AV agent
        agent = OpenAIAVAgent(
            name=name,
            route_manager=None,
            function_registry=self.function_registry,
        )
        
        agent.config.voice = voice
        
        if model:
            agent.config.model = model
        
        if system:
            agent.config.system = system
        
        self.openai_av_agents[name] = agent
        self.add(agent)
        
        return agent
    
    def create_ts_agent(
        self,
        name: str,
        voice: str = None
    ) -> TSAgent:
        """Create a new Text-to-Speech agent"""
        self._check_name(name)
        
        # Create TS agent
        agent = TSAgent(
            name=name,
            route_manager=None
        )
        
        if voice:
            agent.config.voice = voice
        
        self.ts_agents[name] = agent
        self.add(agent)
        
        return agent
    
    def delete_agent(self, name: str):
        """Delete an agent (text, AV, Grok AV, OpenAI AV, or TS)"""
        if name in self.agents:
            agent = self.agents.pop(name)
            asyncio.create_task(agent.cancel())
            self.remove(name)
        elif name in self.av_agents:
            agent = self.av_agents.pop(name)
            asyncio.create_task(agent.stop())
            self.remove(name)
        elif name in self.grok_av_agents:
            agent = self.grok_av_agents.pop(name)
            asyncio.create_task(agent.stop())
            self.remove(name)
        elif name in self.openai_av_agents:
            agent = self.openai_av_agents.pop(name)
            asyncio.create_task(agent.stop())
            self.remove(name)
        elif name in self.ts_agents:
            agent = self.ts_agents.pop(name)
            asyncio.create_task(agent.stop())
            self.remove(name)
        else:
            raise ValueError(f"Agent '{name}' not found")
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get a text agent by name"""
        return self.agents.get(name)
    
    def get_av_agent(self, name: str) -> Optional[AVAgent]:
        """Get a Gemini AV agent by name"""
        return self.av_agents.get(name)
    
    def get_grok_av_agent(self, name: str) -> Optional[GrokAVAgent]:
        """Get a Grok AV agent by name"""
        return self.grok_av_agents.get(name)
    
    def get_openai_av_agent(self, name: str) -> Optional[OpenAIAVAgent]:
        """Get an OpenAI AV agent by name"""
        return self.openai_av_agents.get(name)
    
    def get_ts_agent(self, name: str) -> Optional[TSAgent]:
        """Get a TS agent by name"""
        return self.ts_agents.get(name)
    
    def register_function(self, name: str, func):
        """Register a function for AV agents (Gemini, Grok, and OpenAI) to call"""
        self.function_registry[name] = func
    
    # ── Machine registry ───────────────────────────────────────
    
    def get_machines(self) -> List[str]:
        """Return list of currently registered machine names."""
        return list(self._machines)
    
    def add_machine(self, name: str):
        """
        Register a mounted machine. Auto-creates rules for all agents
        that have register_machines enabled.
        
        The "llm" machine (self) is always ignored.
        """
        name_lower = name.lower()
        if name_lower == "llm" or name_lower in self._machines:
            return
        
        self._machines.append(name_lower)
        
        # Propagate to all agents with register enabled
        for agent in self._all_agents():
            if agent.register_machines:
                agent.add_machine_rule(name_lower)
    
    def remove_machine(self, name: str):
        """
        Unregister a machine. Removes auto-created rules from all agents.
        """
        name_lower = name.lower()
        if name_lower not in self._machines:
            return
        
        self._machines.remove(name_lower)
        
        # Remove from all agents
        for agent in self._all_agents():
            if agent.register_machines:
                agent.remove_machine_rule(name_lower)
    
    def _all_agents(self) -> List[Agent]:
        """Return all text agents (the ones that support register)."""
        return list(self.agents.values())