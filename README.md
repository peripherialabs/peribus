# Peribus

A Plan 9-inspired operating environment where LLM agents, a display server, and a multiplexer are exposed as a 9P filesystem. Everything is a file. One `cat` pipes an AI response into a scene parser. One `echo` rewrites reality.

## ⚠️ DANGER — Read This First

**This software is experimental and has NO security model.** It gives LLM agents direct filesystem and shell access. A single malformed request can wipe your machine — or every machine mounted on your network. **Run only on an isolated, private network with nothing you can't afford to lose.**

## Architecture

```
/n/                        ← mount root
├── llm/                   ← LLMFS: LLM agent filesystem
│   ├── claude/            ← agent instance
│   │   ├── input          ← write a prompt
│   │   ├── output         ← blocking read for response stream
│   │   ├── system         ← system prompt
│   │   ├── config         ← model, provider, temperature
│   │   └── ...
│   └── ctl                ← create/remove agents
├── rio/                   ← Rio: display server + scene graph
│   ├── scene/
│   │   ├── parse          ← write code/commands to execute
│   │   ├── CONTEXT        ← blocking read of executed code
│   │   └── screen         ← screenshot (PNG)
│   ├── version            ← undo/redo
│   └── ctl
└── ctl                    ← mux control (add/remove backends)
```

The **multiplexer** (riomux) stitches backends together at the 9P wire level. Mount remote machines and a single LLM request fans out across the hive.

## Quick Start

**Tested on:** Ubuntu (22.04/24.04)

**Python:** 3.11 recommended (via deadsnakes PPA if needed)

### 1. System prerequisites

```bash
sudo apt install $(cat pre.txt)
```

Contents of `pre.txt`:
```
libminizip-dev
libxcb-cursor0
portaudio19-dev
```

> **Note:** `portaudio19-dev` is required before `pip install pyaudio` will succeed.

### 2. Install 9pfuse (required for mounting)

Peribus uses `9pfuse` to mount 9P filesystems via FUSE. The easiest route is the [standalone build](https://github.com/aperezdc/9pfuse):

```bash
# Install FUSE and build dependencies
sudo apt install libfuse-dev fuse meson

# Clone, build, and install
git clone https://github.com/aperezdc/9pfuse.git
cd 9pfuse
meson setup build
meson compile -Cbuild
sudo cp build/9pfuse /usr/local/bin/
```

Verify it works:
```bash
9pfuse
```

> `start.py` checks for `9pfuse` on PATH at launch and will warn you if it's missing. Without it, you can still run the servers but will need to mount manually.
>
> **Why not the Linux kernel's v9fs?** You can `mount -t 9p` and it will connect, but v9fs does not support streaming reads. Peribus relies on blocking reads that stream data as it arrives (e.g. tailing an LLM response). With v9fs you'll get buffered chunks or EOF instead of a live stream, which breaks the core interaction model. Use 9pfuse.
>
> **Plan 9:** If you're running an actual Plan 9 machine, you can mount and operate the entire network natively — no FUSE needed. It's 9P all the way down.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment

Copy or create a `.env` file at the project root with your API keys:

```
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
GROQ_API_KEY=gsk_...
CEREBRAS_API_KEY=...
CARTESIA_API_KEY=...
```

### 5. Mount point

The system uses `/n` as the mount root. `start.py` should handle creation, but if needed:

```bash
sudo mkdir -p /n
sudo chown $USER /n
```

### 6. Launch

```bash
python start.py
```

The start script walks you through mode selection: create a new mux, connect to an existing one, or run standalone.

## First Steps

1. **Onboarding** — On first launch, right-click to open the onboarding menu and follow the walkthrough.

2. **Open a terminal** and run the `/coder` macro — this sets up a workspace-aware coding agent with everything wired up automatically. See `agent.py` for the agent model and `terminal_widget.py` for how macros configure it.

3. **Voice agents** — Right-click and scroll the widget to select between voice providers (Grok, Gemini, OpenAI). TTS/STT included.

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+T` | Terminal mode (manual 9P commands) |
| `Ctrl+O` | Operator panel |
| `Ctrl+P` | Version panel |
| `Ctrl+E` | AI Editor / Explorer |

## Terminal Commands

```
/coder [provider] [model]    Setup coding agent (recommended first command)
/new name [system]            Create + connect an agent
/connect <name>               Connect to existing agent
/master [provider] [model]    Spawn coordinating master agent
/av [voice]                   Grok voice agent
/av_gemini [voice]            Gemini voice agent
/attach <src> <dst>           Route one file to another (blocking, no polling)
/use <alias>                  Quick model switch (sonnet, opus, kimi, flash, ...)
```

See the full command reference in `terminal_widget.py`.

## Multiplexer — Hive Mode

Mount remote Peribus machines and distribute work across them:

```bash
# Add remote machines
echo 'add workstation2 192.168.1.50:5642' > /n/ctl
echo 'add workstation3 192.168.1.51:5642' > /n/ctl

# Then create your /coder — the macro auto-connects everything
# and compacts the system context across the hive
```

## The 9P Filesystem

The 9P filesystem is the core of everything. All state is files. All control is reads and writes. Example:

```bash
# Send a prompt and stream the response into the scene parser
echo "draw a red cube" > /n/llm/claude/input
cat /n/llm/claude/output > /n/rio/scene/parse
```

Blocking read semantics are preserved through the mux — `cat` blocks until content is ready, just like Plan 9.

## License

Apache License 2.0 — Copyright 2025–2026 Peripheria. See [LICENSE](LICENSE) for details.