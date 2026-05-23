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
from .embedding import EmbedAgent
from .meta_agent import (
    MetaAgent, get_agents_dir,
    load_saved_module, list_saved_modules,
)



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

        elif cmd == "embed":
            # Create new embedding agent: embed [name]
            if not arg:
                arg = "embed"
            name = arg.split()[0]
            self.fs.create_embed_agent(name)
            return f"Embed Agent '{name}' created"

        elif cmd == "meta":
            # Create the meta-agent: meta [name] [provider] [model]
            #
            # By default it lands at /<root>/meta with whatever the root
            # provider is. The meta-agent's job is to write source code
            # for other agents (see meta_agent.py).
            parts = arg.split() if arg else []
            name = parts[0] if len(parts) > 0 else "meta"
            provider = parts[1] if len(parts) > 1 else None
            model = parts[2] if len(parts) > 2 else None
            self.fs.create_meta_agent(name, provider, model)
            return f"Meta Agent '{name}' created"

        elif cmd == "load":
            # Load a saved custom agent from $LLMFS_AGENTS_DIR.
            # Usage: load <name>
            if not arg:
                # No arg → list what's available.
                names = list_saved_modules(self.fs.agents_dir)
                return " ".join(names) if names else "(none)"
            name = arg.split()[0]
            self.fs.load_custom_agent(name)
            return f"Custom agent '{name}' loaded"
        
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
            raise ValueError(f"Unknown command: {cmd}. Available: provider, new, av, grok, openai, ts, embed, meta, load, delete, machine")
    
    async def get_status(self) -> bytes:
        lines = [
            f"provider {self.fs.provider.name}",
            f"agents {len(self.fs.agents)}",
            f"av_agents {len(self.fs.av_agents)}",
            f"grok_av_agents {len(self.fs.grok_av_agents)}",
            f"openai_av_agents {len(self.fs.openai_av_agents)}",
            f"ts_agents {len(self.fs.ts_agents)}",
            f"embed_agents {len(self.fs.embed_agents)}",
            f"meta_agents {len(self.fs.meta_agents)}",
            f"custom_agents {len(self.fs.custom_agents)}",
            f"agents_dir {self.fs.agents_dir}",
            f"saved {' '.join(list_saved_modules(self.fs.agents_dir)) or '(none)'}",
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

        self.embed_agents: Dict[str, EmbedAgent] = {}

        # Meta-agents (LLM agents that write other agents' code)
        self.meta_agents: Dict[str, MetaAgent] = {}

        # Custom agents loaded from $LLMFS_AGENTS_DIR. Each is a generic
        # SyntheticDir whose schema is defined by the generated module.
        # Stored separately because we don't own their lifecycle (no
        # uniform stop()/cancel() contract beyond what they expose).
        self.custom_agents: Dict[str, SyntheticDir] = {}

        # Resolve persistence dir once at construction; honour
        # $LLMFS_AGENTS_DIR with a sensible default.
        self.agents_dir = get_agents_dir()
        
        # Global function registry for AV agents (shared by Gemini, Grok, and OpenAI)
        self.function_registry = {}
        
        # Machine registry: tracks mounted 9P machines (from riomux ctl)
        # Excludes "llm" (self). Used for auto-creating agent rules.
        self._machines: List[str] = []
        
        # Build filesystem tree
        self.add(CtlFile("ctl", LLMFSCtlHandler(self)))
        self.add(ProvidersFile())

        # Re-load any custom agents persisted from previous sessions.
        # Failures are logged to stderr rather than crashing the server
        # — one broken file shouldn't take the whole filesystem down.
        self._autoload_saved_agents()
    
    def _check_name(self, name: str):
        """Validate that an agent name doesn't conflict with reserved files"""
        if name in self.RESERVED_NAMES:
            raise ValueError(f"Name '{name}' is reserved (conflicts with {name} file)")
        if (name in self.agents or name in self.av_agents 
                or name in self.grok_av_agents or name in self.openai_av_agents
                or name in self.ts_agents or name in self.embed_agents
                or name in self.meta_agents or name in self.custom_agents):
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

    def create_embed_agent(
        self,
        name: str = "embed",
    ) -> EmbedAgent:
        """Create a new embedding filesystem agent"""
        self._check_name(name)

        agent = EmbedAgent(name=name)

        # Try to load existing index from disk
        agent.load_from_disk()

        self.embed_agents[name] = agent
        self.add(agent)

        return agent

    def create_meta_agent(
        self,
        name: str = "meta",
        provider_name: str = None,
        model: str = None,
    ) -> MetaAgent:
        """
        Create a meta-agent. Talk to it to generate code for new agent types.
        See llmfs.meta_agent for the full contract.
        """
        self._check_name(name)

        provider = self.provider
        if provider_name:
            provider = get_provider(provider_name)

        agent = MetaAgent(
            name=name,
            provider=provider,
            fs_root=self,
            default_model=model,
        )

        self.meta_agents[name] = agent
        self.add(agent)

        # Same machine-rule back-channel the standard Agent uses.
        agent._fs_root = self

        return agent

    def mount_custom_agent(
        self,
        name: str,
        instance: SyntheticDir,
        source_path=None,
    ):
        """
        Mount a SyntheticDir produced by a generated module under the root.

        Called by MetaAgent.build_from_last_output and load_custom_agent.
        Idempotent-ish: if a custom agent of the same name is already
        mounted, it's replaced (the old one is dropped).
        """
        # Make sure the instance presents the requested name; the contract
        # says create(name) honours it, but defensive-rename if needed so
        # the directory entry matches the dict key.
        if instance.name != name:
            instance.name = name

        if name in self.custom_agents:
            # Replace: remove the old node so .add() doesn't double-add.
            self.remove(name)
            self.custom_agents.pop(name, None)
        else:
            # Could collide with a non-custom agent slot; _check_name was
            # the caller's job, but be safe.
            try:
                self._check_name(name)
            except ValueError:
                # Name is taken by something we don't own — refuse rather
                # than overwrite a built-in agent type.
                raise

        self.custom_agents[name] = instance
        self.add(instance)

        if source_path is not None:
            # Stash for diagnostics (e.g. so we can show where it came from).
            setattr(instance, "_source_path", str(source_path))

    def load_custom_agent(self, name: str) -> SyntheticDir:
        """
        Load a saved custom agent from $LLMFS_AGENTS_DIR by name.

        Used by `echo 'load <name>' > ctl` and by startup auto-load.
        """
        module, path = load_saved_module(name, self.agents_dir)
        instance = module.create(name)
        self.mount_custom_agent(name, instance, source_path=path)
        return instance

    def _autoload_saved_agents(self):
        """
        At startup, walk $LLMFS_AGENTS_DIR and load every .py file. One
        broken file logs and is skipped; we don't fail the whole boot.
        """
        import sys
        for name in list_saved_modules(self.agents_dir):
            try:
                self.load_custom_agent(name)
            except Exception as e:
                # No logger configured yet at this point in some setups;
                # fall through to stderr.
                print(
                    f"[llmfs] WARNING: failed to load saved agent "
                    f"'{name}': {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
    
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
        """Delete an agent (text, AV, Grok AV, OpenAI AV, TS, embed, meta, or custom)"""
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
        elif name in self.embed_agents:
            agent = self.embed_agents.pop(name)
            asyncio.create_task(agent.stop())
            self.remove(name)
        elif name in self.meta_agents:
            agent = self.meta_agents.pop(name)
            asyncio.create_task(agent.cancel())
            self.remove(name)
        elif name in self.custom_agents:
            # Custom agents are generic SyntheticDirs; we don't assume any
            # particular shutdown method. Try common ones, ignore failures.
            agent = self.custom_agents.pop(name)
            for shutdown_name in ("stop", "cancel", "close"):
                method = getattr(agent, shutdown_name, None)
                if callable(method):
                    try:
                        result = method()
                        if asyncio.iscoroutine(result):
                            asyncio.create_task(result)
                    except Exception:
                        pass
                    break
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

    def get_embed_agent(self, name: str) -> Optional[EmbedAgent]:
        """Get an embedding agent by name"""
        return self.embed_agents.get(name)
    
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
        if name_lower in ["llm", "peribus"] or name_lower in self._machines:
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