"""
Embedding Filesystem Agent for LLMFS

Semantic search over Python mini-app source files (.py only).
Uses sentence-transformers (all-MiniLM-L6-v2) for fast, CPU-friendly
text embeddings and FAISS for nearest-neighbour retrieval.
No GPU or VL model required.

Directory structure:
    /n/llm/embed/
    ├── ctl           # Control: scan, rebuild, status, config
    ├── input         # Write a search query (triggers search on clunk)
    ├── OUTPUT        # Blocking read: search results stream
    ├── PATH          # Blocking read: first (top) result path only
    ├── history       # JSON log of past queries + results
    ├── config        # JSON configuration (folders, top_k, etc.)
    ├── system        # System prompt / description of the agent
    ├── rules         # Plumbing rules for result extraction
    ├── state         # Snapshot/restore full agent state
    ├── errors        # Error stream
    ├── descriptions  # Read: current descriptions.txt content
    │                 # Write: manually add/edit descriptions
    ├── index_status  # Read-only: FAISS index stats
    └── {SUP_OUTPUTS} # Supplementary outputs from plumbing rules

Control commands (echo into ctl):
    scan <folder> [-r]     Scan folder, index .py files
    rebuild                Rebuild FAISS index from descriptions
    add <path>             Index a single .py file
    remove <path>          Remove a file from the index
    folders                List scanned folders
    stats                  Index statistics
    open <N>               Auto-open top N results
    top_k <N>              Set number of search results
    clear                  Clear index and descriptions

Supported file types:
    Python  : .py
"""

import asyncio
import json
import os
import re
import time
import gc
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import numpy as np

from core.files import (
    SyntheticDir, SyntheticFile, StreamFile, QueueFile,
    CtlFile, CtlHandler
)
from core.types import FidState


# ---------------------------------------------------------------------------
# File-type classification  (Python source files only)
# ---------------------------------------------------------------------------

PYTHON_EXTS = {'.py'}

TEXT_FILE_MAX_CHARS = 8000  # enough for most mini-app sources


def is_python(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in PYTHON_EXTS


def is_supported(path: str) -> bool:
    return is_python(path)


def file_type_label(path: str) -> str:
    return "python"


# ---------------------------------------------------------------------------
# Lazy-loaded backends
# ---------------------------------------------------------------------------

class BackendMutex:
    """No-op mutex kept for API compatibility (single backend now)."""

    def __init__(self):
        pass

    def register(self, describe, embed):
        pass

    def request_describe(self):
        pass

    def request_embed(self):
        pass


class DescribeBackend:
    """
    Reads Python source files and returns their content as the
    "description" used for indexing.  No model, no GPU required.
    """

    def __init__(self, mutex: 'BackendMutex'):
        self._loaded = True  # always ready

    def _ensure_loaded(self):
        pass

    def describe(self, path: str, **kwargs) -> str:
        """Return the file content (up to TEXT_FILE_MAX_CHARS characters)."""
        if not is_python(path):
            raise ValueError(f"Only .py files are supported: {path}")
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(TEXT_FILE_MAX_CHARS)
            return content.strip()
        except Exception as e:
            raise RuntimeError(f"Could not read {path}: {e}") from e

    def unload(self):
        pass


class EmbedBackend:
    """
    sentence-transformers (all-MiniLM-L6-v2) + FAISS for fast semantic
    search over Python source code.

    - CPU-friendly; loads in < 1 s after first download.
    - Dimension: 384.
    - Loaded lazily on first embed/search call.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIMENSION = 384
    BATCH_SIZE = 64

    def __init__(self, mutex: 'BackendMutex'):
        self._model = None
        self._loaded = False
        self.index = None
        self.metadata: List[Tuple[str, str]] = []  # (path, content)

    def _ensure_loaded(self):
        if self._loaded:
            return
        import faiss
        from sentence_transformers import SentenceTransformer

        # First load downloads ~90 MB; subsequent loads are instant from cache.
        self._model = SentenceTransformer(self.MODEL_NAME)
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.DIMENSION)
        self._loaded = True

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised float32 embeddings for a list of texts."""
        import faiss
        vecs = self._model.encode(
            texts,
            batch_size=self.BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
        return vecs

    def build_from_descriptions(self, descriptions: List[Tuple[str, str]]) -> int:
        """
        Build FAISS index from a list of (path, content) tuples.
        Only includes entries where the path still exists on disk.
        """
        import faiss

        self._ensure_loaded()
        self.index = faiss.IndexFlatIP(self.DIMENSION)
        self.metadata.clear()

        valid = [(p, c) for p, c in descriptions if os.path.exists(p)]
        if not valid:
            return 0

        paths, contents = zip(*valid)
        vecs = self._embed_texts(list(contents))
        self.index.add(vecs)
        self.metadata = list(zip(paths, contents))
        return len(self.metadata)

    def add_single(self, path: str, content: str):
        """Add a single file to the index."""
        self._ensure_loaded()
        vec = self._embed_texts([content])
        self.index.add(vec)
        self.metadata.append((path, content))

    def remove_by_path(self, path: str) -> bool:
        """
        Remove a file from the index by path.
        FAISS IndexFlatIP does not support removal, so we rebuild.
        """
        import faiss

        abs_path = os.path.abspath(path)
        indices_to_keep = [
            i for i, (p, _) in enumerate(self.metadata) if p != abs_path
        ]

        if len(indices_to_keep) == len(self.metadata):
            return False

        if not indices_to_keep:
            self.index = faiss.IndexFlatIP(self.DIMENSION)
            self.metadata.clear()
            return True

        old_vectors = np.array(
            [self.index.reconstruct(i) for i in indices_to_keep]
        ).astype("float32")
        new_metadata = [self.metadata[i] for i in indices_to_keep]

        self.index = faiss.IndexFlatIP(self.DIMENSION)
        self.index.add(old_vectors)
        self.metadata = new_metadata
        return True

    def search(self, query: str, k: int = 5) -> List[Tuple[float, str, str]]:
        """
        Search the index.  Returns list of (score, path, content) tuples.
        """
        self._ensure_loaded()

        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = self._embed_texts([query])
        actual_k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, actual_k)

        results = []
        for i in range(actual_k):
            idx = indices[0][i]
            if idx == -1:
                continue
            path, content = self.metadata[idx]
            results.append((float(distances[0][i]), path, content))
        return results

    def save(self, index_file: str, meta_file: str):
        """Persist index and metadata to disk."""
        import faiss
        import pickle

        if self.index is not None:
            faiss.write_index(self.index, index_file)
        with open(meta_file, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_file: str, meta_file: str) -> bool:
        """Load index and metadata from disk without instantiating the model."""
        import faiss
        import pickle

        if os.path.exists(index_file) and os.path.exists(meta_file):
            self.index = faiss.read_index(index_file)
            with open(meta_file, "rb") as f:
                self.metadata = pickle.load(f)
            self._loaded = False
            return True
        return False

    @property
    def total(self) -> int:
        return self.index.ntotal if self.index else 0

    def unload(self):
        """Free memory."""
        del self._model
        self._model = None
        self._loaded = False
        gc.collect()


# ---------------------------------------------------------------------------
# File collection utility
# ---------------------------------------------------------------------------

def collect_files(root: str, recursive: bool = False) -> List[str]:
    """
    Return a sorted list of absolute paths for all supported files
    under root. Skips __pycache__ and hidden files/dirs.
    """
    supported = []

    if recursive:
        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            dirnames[:] = sorted(
                d for d in dirnames
                if d != "__pycache__" and not d.startswith(".")
            )
            for filename in sorted(filenames):
                if filename.startswith("."):
                    continue
                if is_supported(filename):
                    supported.append(os.path.join(dirpath, filename))
    else:
        for filename in sorted(os.listdir(root)):
            if filename.startswith("."):
                continue
            path = os.path.join(root, filename)
            if os.path.isfile(path) and is_supported(filename):
                supported.append(path)

    return supported


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class EmbedAgentState(Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    INDEXING = "indexing"
    SEARCHING = "searching"
    ERROR = "error"


@dataclass
class EmbedAgentConfig:
    """Configuration for the embedding agent."""
    # Search settings
    top_k: int = 5

    # Persistence
    descriptions_file: str = "descriptions.txt"
    index_file: str = "minilm_memory.index"
    metadata_file: str = "minilm_metadata.pkl"

    # Scanned folders
    folders: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    score: float
    path: str
    description: str
    file_type: str


@dataclass
class QueryRecord:
    query: str
    timestamp: float
    results: List[SearchResult]


# ---------------------------------------------------------------------------
# Synthetic files for the embedding agent
# ---------------------------------------------------------------------------

class EmbedCtlHandler(CtlHandler):
    """Control handler for the embedding filesystem agent."""

    def __init__(self, agent: 'EmbedAgent'):
        self.agent = agent

    async def execute(self, command: str) -> Optional[str]:
        parts = command.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "scan":
            if not arg:
                raise ValueError("Usage: scan <folder> [-r]")
            recursive = False
            folder = arg
            if arg.endswith(" -r") or arg.endswith(" --recursive"):
                recursive = True
                folder = arg.rsplit(None, 1)[0]
            asyncio.create_task(self.agent.scan_folder(folder, recursive))
            mode = "recursive" if recursive else "flat"
            return f"Scanning {folder} ({mode})..."

        elif cmd == "rebuild":
            asyncio.create_task(self.agent.rebuild_index())
            return "Rebuilding index..."

        elif cmd == "add":
            if not arg:
                raise ValueError("Usage: add <path>")
            asyncio.create_task(self.agent.add_file(arg))
            return f"Adding {arg}..."

        elif cmd == "remove":
            if not arg:
                raise ValueError("Usage: remove <path>")
            removed = self.agent.remove_file(arg)
            return f"Removed: {arg}" if removed else f"Not found: {arg}"

        elif cmd == "folders":
            folders = self.agent.embed_config.folders
            return "\n".join(folders) if folders else "(none)"

        elif cmd == "stats":
            return self.agent.get_stats()

        elif cmd == "top_k":
            if arg:
                self.agent.embed_config.top_k = int(arg)
                return f"top_k set to {arg}"
            return str(self.agent.embed_config.top_k)

        elif cmd == "clear":
            await self.agent.clear_all()
            return "Index and descriptions cleared"

        elif cmd == "save":
            self.agent.save_to_disk()
            return "Saved index and descriptions to disk"

        elif cmd == "load":
            target = arg.lower() if arg else ""
            if target == "describe":
                asyncio.create_task(self.agent.load_describe())
                return "Loading describe model..."
            elif target in ("search", "embed"):
                asyncio.create_task(self.agent.load_search())
                return "Loading search model..."
            else:
                loaded = self.agent.load_from_disk()
                return "Loaded from disk" if loaded else "No saved index found"

        elif cmd == "unload":
            target = arg.lower() if arg else "all"
            if target in ("describe", "all"):
                self.agent.describer.unload()
            if target in ("embed", "search", "all"):
                self.agent.embedder.unload()
            return f"Unloaded: {target}"

        else:
            raise ValueError(
                f"Unknown command: {cmd}. "
                f"Available: scan, rebuild, add, remove, folders, stats, "
                f"top_k, clear, save, "
                f"load [describe|search], unload"
            )

    async def get_status(self) -> bytes:
        a = self.agent
        lines = [
            f"state {a.state.value}",
            f"indexed {a.embedder.total}",
            f"descriptions {len(a.descriptions)}",
            f"queries {len(a.query_history)}",
            f"top_k {a.embed_config.top_k}",

            f"folders {len(a.embed_config.folders)}",
        ]
        if a.state == EmbedAgentState.ERROR and a.last_error:
            lines.append(f"error {a.last_error}")
        return ("\n".join(lines) + "\n").encode()


class EmbedInputFile(SyntheticFile):
    """
    Write a search query to trigger semantic search.

    Generation is triggered on clunk (fid close) so multi-chunk
    writes assemble into a single query.
    """

    def __init__(self, agent: 'EmbedAgent'):
        super().__init__("input")
        self.agent = agent
        self._last_input = ""
        self._write_buffers: Dict[int, bytearray] = {}

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        return self._last_input.encode()[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        fid_key = id(fid)
        if fid_key not in self._write_buffers:
            self._write_buffers[fid_key] = bytearray()

        buf = self._write_buffers[fid_key]
        if offset == 0 and len(buf) > 0:
            buf.clear()

        if offset + len(data) > len(buf):
            buf.extend(b'\x00' * (offset + len(data) - len(buf)))
        buf[offset:offset + len(data)] = data

        return len(data)

    async def clunk(self, fid: FidState):
        fid_key = id(fid)
        buf = self._write_buffers.pop(fid_key, None)

        if not buf:
            return

        query = bytes(buf).decode("utf-8", errors="replace").strip()
        if not query:
            return

        self._last_input = query
        asyncio.create_task(self.agent.search(query))


class EmbedHistoryFile(SyntheticFile):
    """Read query history as JSON. Write 'clear' to reset."""

    def __init__(self, agent: 'EmbedAgent'):
        super().__init__("history")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        records = []
        for qr in self.agent.query_history:
            records.append({
                "query": qr.query,
                "timestamp": qr.timestamp,
                "results": [
                    {
                        "score": r.score,
                        "path": r.path,
                        "description": r.description,
                        "type": r.file_type,
                    }
                    for r in qr.results
                ],
            })
        data = json.dumps(records, indent=2, ensure_ascii=False).encode()
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        text = data.decode("utf-8", errors="replace").strip().lower()
        if text in ("clear", "delete", ""):
            self.agent.query_history.clear()
        return len(data)


class EmbedConfigFile(SyntheticFile):
    """Read/write agent configuration as JSON."""

    def __init__(self, agent: 'EmbedAgent'):
        super().__init__("config")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        cfg = self.agent.embed_config
        config = {
            "top_k": cfg.top_k,
            "descriptions_file": cfg.descriptions_file,
            "index_file": cfg.index_file,
            "metadata_file": cfg.metadata_file,
            "folders": cfg.folders,
        }
        data = json.dumps(config, indent=2).encode()
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        try:
            config = json.loads(data.decode())
            cfg = self.agent.embed_config
            if "top_k" in config:
                cfg.top_k = int(config["top_k"])
            if "descriptions_file" in config:
                cfg.descriptions_file = config["descriptions_file"]
            if "index_file" in config:
                cfg.index_file = config["index_file"]
            if "metadata_file" in config:
                cfg.metadata_file = config["metadata_file"]
            return len(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")


class DescriptionsFile(SyntheticFile):
    """
    Read the current descriptions database.
    Write to manually add entries (format: /path | description).
    """

    def __init__(self, agent: 'EmbedAgent'):
        super().__init__("descriptions")
        self.agent = agent
        self._write_buffers: Dict[int, bytearray] = {}

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        lines = [
            f"{path} | {desc[:80].replace(chr(10), ' ')}{'...' if len(desc) > 80 else ''}"
            for path, desc in self.agent.descriptions
        ]
        data = ("\n".join(lines) + "\n").encode() if lines else b"(empty)\n"
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        fid_key = id(fid)
        if fid_key not in self._write_buffers:
            self._write_buffers[fid_key] = bytearray()

        buf = self._write_buffers[fid_key]
        if offset == 0 and len(buf) > 0:
            buf.clear()
        if offset + len(data) > len(buf):
            buf.extend(b'\x00' * (offset + len(data) - len(buf)))
        buf[offset:offset + len(data)] = data

        return len(data)

    async def clunk(self, fid: FidState):
        fid_key = id(fid)
        buf = self._write_buffers.pop(fid_key, None)
        if not buf:
            return

        text = bytes(buf).decode("utf-8", errors="replace").strip()
        if not text:
            return

        for line in text.split("\n"):
            line = line.strip()
            if "|" in line:
                path, desc = [x.strip() for x in line.split("|", 1)]
                self.agent.descriptions.append((path, desc))


class IndexStatusFile(SyntheticFile):
    """Read-only file showing FAISS index statistics."""

    def __init__(self, agent: 'EmbedAgent'):
        super().__init__("index_status")
        self.agent = agent

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        data = self.agent.get_stats().encode()
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError("index_status is read-only")


# ---------------------------------------------------------------------------
# Supplementary output file
# ---------------------------------------------------------------------------

class EmbedSupplementaryOutputFile(SyntheticFile):
    """
    Blocking output file for extracted content from search results.
    Follows the same WAITING → READY → CONSUMED lifecycle as the
    main agent's supplementary outputs.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.blocks: List[str] = []
        self._content_ready = asyncio.Event()
        self._content_consumed = False
        self._lock = asyncio.Lock()

    def add_block(self, content: str):
        self.blocks.append(content)

    def mark_ready(self):
        self._content_ready.set()

    def clear(self):
        self.blocks.clear()
        self._content_ready.clear()
        self._content_consumed = False

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        if offset == 0 and self._content_consumed:
            async with self._lock:
                if self._content_consumed:
                    self._content_consumed = False
                    self._content_ready.clear()

        await self._content_ready.wait()

        async with self._lock:
            content = "\n\n".join(self.blocks)
            if content:
                content += "\n"
            data = content.encode()
            chunk = data[offset:offset + count]
            if offset + len(chunk) >= len(data):
                self._content_consumed = True
            return chunk

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError(f"{self.name} is read-only")


# ---------------------------------------------------------------------------
# PATH file: blocking read returning the first (top) result path
# ---------------------------------------------------------------------------

class EmbedPathFile(SyntheticFile):
    """
    Blocking read-only file that returns the single top result path from
    the most recent search.

    Where OUTPUT returns the full formatted list of results, PATH returns
    just the first (highest-scoring) path, with a trailing newline. A read
    blocks until a search has produced a result; it follows the same
    WAITING -> READY -> CONSUMED lifecycle as the supplementary outputs so
    each search makes the value re-readable.
    """

    def __init__(self, name: str = "PATH"):
        super().__init__(name)
        self._path: str = ""
        self._ready = asyncio.Event()
        self._consumed = False
        self._lock = asyncio.Lock()

    def set_path(self, path: str):
        """Record the top result path and mark it ready for reading."""
        self._path = path or ""
        self._consumed = False
        self._ready.set()

    def clear(self):
        self._path = ""
        self._ready.clear()
        self._consumed = False

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        # Reset the gate at the start of a fresh read once consumed, so the
        # next read blocks until a new search populates a path.
        if offset == 0 and self._consumed:
            async with self._lock:
                if self._consumed:
                    self._consumed = False
                    self._ready.clear()

        await self._ready.wait()

        async with self._lock:
            content = (self._path + "\n") if self._path else "\n"
            data = content.encode()
            chunk = data[offset:offset + count]
            if offset + len(chunk) >= len(data):
                self._consumed = True
            return chunk

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        raise PermissionError(f"{self.name} is read-only")


# ---------------------------------------------------------------------------
# The embedding agent
# ---------------------------------------------------------------------------

EMBED_HELP_TEXT = """\
NAME
    {name} — semantic code-search agent (embeddings + FAISS)

SYNOPSIS
    echo "scan /path/to/src -r" > /n/llm/{name}/ctl
    echo "where is the retry logic?" > /n/llm/{name}/input
    cat /n/llm/{name}/OUTPUT      # ranked results
    cat /n/llm/{name}/PATH        # top result path only

DESCRIPTION
    Indexes Python source files and searches them semantically using
    sentence-transformers embeddings and a FAISS index. Point it at one
    or more folders with `scan`, then write a natural-language query to
    `input`. OUTPUT returns the ranked, formatted matches; PATH returns
    just the single best-matching file path (handy for plumbing into an
    editor). The index and generated descriptions can be saved to and
    loaded from disk.

FILES
    ctl           Control commands (see COMMANDS). Read it for status.
    input         Write a natural-language query to trigger a search.
    OUTPUT        Blocking read; formatted ranked results.
    PATH          Blocking read; the top result's path only.
    descriptions  Read/write the description database.
    index_status  Read-only index statistics.
    history       Query history as JSON.
    config        Read/write configuration as JSON.
    errors        Error log.
    help          This file.

COMMANDS
    scan <folder> [-r]   Index a folder (add -r/--recursive for subdirs).
    add <path>           Index a single file.
    remove <path>        Remove a file from the index.
    rebuild              Rebuild the index from scratch.
    folders              List indexed folders.
    stats                Print index statistics.
    top_k [n]            Get/set the number of results returned.
    save                 Save index + descriptions to disk.
    load                 Load index + descriptions from disk.
    load describe|search Load just the describe or search model.
    unload [describe|search|all]   Unload model(s) to free memory.
    clear                Clear the index and descriptions.

EXAMPLES
    # Index a project, then search
    echo "scan ~/proj/src -r" > /n/llm/{name}/ctl
    echo "function that parses 9P messages" > /n/llm/{name}/input
    cat /n/llm/{name}/OUTPUT

    # Open the best match in your editor
    $EDITOR "$(cat /n/llm/{name}/PATH)"

NOTES
    Requires numpy, sentence-transformers, and FAISS (plus a describe
    model). The first scan/search loads models, which can take a while
    and use significant memory; use `unload` to free it. Reads on OUTPUT
    and PATH block until a search produces a result.
"""


class EmbedAgent(SyntheticDir):
    """
    Embedding filesystem agent: index Python source files and search
    them semantically via sentence-transformers + FAISS.

    Filesystem structure:
        embed/
        ├── ctl              # Control commands
        ├── input            # Write search queries
        ├── OUTPUT           # Blocking read: formatted search results
        ├── PATH             # Blocking read: first (top) result path only
        ├── history          # JSON query history
        ├── config           # JSON configuration
        ├── descriptions     # Read/write description database
        ├── index_status     # Read-only index stats
        ├── errors           # Error stream
        └── {RULES_OUTPUTS}  # Dynamic supplementary outputs
    """

    def __init__(self, name: str = "embed"):
        super().__init__(name)

        self.state = EmbedAgentState.IDLE
        self.last_error: Optional[str] = None

        self.embed_config = EmbedAgentConfig()

        _mutex = BackendMutex()
        self.describer = DescribeBackend(_mutex)
        self.embedder = EmbedBackend(_mutex)
        _mutex.register(self.describer, self.embedder)

        self.descriptions: List[Tuple[str, str]] = []
        self.query_history: List[QueryRecord] = []

        self.plumbing_rules: List[Dict[str, str]] = []
        self.supplementary_outputs: Dict[str, EmbedSupplementaryOutputFile] = {}

        self.output = StreamFile("OUTPUT")
        self.path_file = EmbedPathFile("PATH")
        self.errors = QueueFile("errors")

        self.add(CtlFile("ctl", EmbedCtlHandler(self)))
        self.add(EmbedInputFile(self))
        self.add(self.output)
        self.add(self.path_file)
        self.add(EmbedHistoryFile(self))
        self.add(EmbedConfigFile(self))
        self.add(DescriptionsFile(self))
        self.add(IndexStatusFile(self))
        self.add(self.errors)

        from .meta_agent import HelpFile
        self.add(HelpFile(EMBED_HELP_TEXT.format(name=self.name), name="help"))

        self.create_supplementary_output("CONTENT")

    # --- Core operations ---

    async def scan_folder(self, folder: str, recursive: bool = False):
        """Scan a folder and generate descriptions for all supported files."""
        folder = os.path.abspath(folder)

        if not os.path.exists(folder):
            await self.errors.post(f"Folder not found: {folder}\n".encode())
            return

        self.state = EmbedAgentState.SCANNING
        await self.output.reset()

        try:
            files = collect_files(folder, recursive)
            mode = "recursive" if recursive else "flat"
            header = (
                f"Scanning {folder} ({mode}): "
                f"{len(files)} supported files\n\n"
            )
            await self.output.append(header.encode())

            if folder not in self.embed_config.folders:
                self.embed_config.folders.append(folder)

            new_descriptions = []

            for i, path in enumerate(files):
                rel_path = os.path.relpath(path, folder)
                progress = f"[{i + 1}/{len(files)}] {rel_path}... "
                await self.output.append(progress.encode())

                try:
                    desc = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.describer.describe,
                        path,
                    )

                    if desc:
                        abs_path = os.path.abspath(path)
                        new_descriptions.append((abs_path, desc))
                        self.descriptions.append((abs_path, desc))
                        result_line = f"DONE\n  > {desc}\n"
                        await self.output.append(result_line.encode())
                    else:
                        await self.output.append(b"EMPTY\n")

                except Exception as e:
                    error_msg = f"FAILED: {str(e)[:200]}\n"
                    await self.output.append(error_msg.encode())
                    await self.errors.post(
                        f"Error describing {path}: {e}\n".encode()
                    )


            self._save_descriptions()

            summary = (
                f"\nScan complete: {len(new_descriptions)} new descriptions. "
                f"Total: {len(self.descriptions)}\n"
            )
            await self.output.append(summary.encode())

            model_note = (
                " (downloading model on first run, please wait)\n"
                if not self.embedder._loaded else "\n"
            )
            await self.output.append(b"Rebuilding search index..." + model_note.encode())
            count = await asyncio.get_event_loop().run_in_executor(
                None, self._rebuild_index_sync
            )
            await self.output.append(
                f"Index rebuilt: {count} items indexed.\n".encode()
            )

            self.state = EmbedAgentState.IDLE

        except Exception as e:
            self.state = EmbedAgentState.ERROR
            self.last_error = str(e)
            await self.errors.post(f"Scan error: {e}\n".encode())

        finally:
            await self.output.finish()

    async def rebuild_index(self):
        """Rebuild FAISS index from current descriptions."""
        self.state = EmbedAgentState.INDEXING
        await self.output.reset()

        try:
            await self.output.append(
                f"Building index from {len(self.descriptions)} descriptions...\n".encode()
            )

            count = await asyncio.get_event_loop().run_in_executor(
                None, self.embedder.build_from_descriptions, self.descriptions
            )

            self.embedder.save(
                self.embed_config.index_file,
                self.embed_config.metadata_file,
            )

            result = f"Index built: {count} items.\n"
            await self.output.append(result.encode())
            self.state = EmbedAgentState.IDLE

        except Exception as e:
            self.state = EmbedAgentState.ERROR
            self.last_error = str(e)
            await self.errors.post(f"Index error: {e}\n".encode())

        finally:
            await self.output.finish()

    def _rebuild_index_sync(self) -> int:
        """Synchronous index rebuild (for run_in_executor)."""
        count = self.embedder.build_from_descriptions(self.descriptions)
        self.embedder.save(
            self.embed_config.index_file,
            self.embed_config.metadata_file,
        )
        return count

    async def add_file(self, path: str):
        """Describe and index a single file."""
        path = os.path.abspath(path)
        if not os.path.exists(path):
            await self.errors.post(f"File not found: {path}\n".encode())
            return

        if not is_supported(path):
            await self.errors.post(f"Unsupported file type: {path}\n".encode())
            return

        self.state = EmbedAgentState.SCANNING
        await self.output.reset()

        try:
            await self.output.append(
                f"Reading {path}...\n".encode()
            )

            desc = await asyncio.get_event_loop().run_in_executor(
                None,
                self.describer.describe,
                path,
            )

            if desc:
                self.descriptions.append((path, desc))
                self._save_descriptions()

                await asyncio.get_event_loop().run_in_executor(
                    None, self.embedder.add_single, path, desc
                )
                self.embedder.save(
                    self.embed_config.index_file,
                    self.embed_config.metadata_file,
                )

                result = f"Added: {path}\n  > {desc}\n"
                await self.output.append(result.encode())
            else:
                await self.output.append(b"Description was empty.\n")

            self.state = EmbedAgentState.IDLE

        except Exception as e:
            self.state = EmbedAgentState.ERROR
            self.last_error = str(e)
            await self.errors.post(f"Add error: {e}\n".encode())

        finally:
            await self.output.finish()

    def remove_file(self, path: str) -> bool:
        """Remove a file from descriptions and index."""
        abs_path = os.path.abspath(path)

        original_len = len(self.descriptions)
        self.descriptions = [
            (p, d) for p, d in self.descriptions if p != abs_path
        ]
        removed_desc = len(self.descriptions) < original_len

        removed_idx = self.embedder.remove_by_path(abs_path)

        if removed_desc:
            self._save_descriptions()
        if removed_idx:
            self.embedder.save(
                self.embed_config.index_file,
                self.embed_config.metadata_file,
            )

        return removed_desc or removed_idx

    async def search(self, query: str):
        """Execute a semantic search and stream results."""
        if self.embedder.total == 0:
            await self.errors.post(
                b"Index is empty. Run 'scan' first.\n"
            )
            return

        self.state = EmbedAgentState.SEARCHING
        await self.output.reset()

        for sup in self.supplementary_outputs.values():
            sup.clear()

        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                self.embedder.search,
                query,
                self.embed_config.top_k,
            )

            lines = [f"Query: {query}\n"]
            search_results = []

            for rank, (score, path, content) in enumerate(results, 1):
                lines.append(f"  [{score:.4f}] {path}")

                search_results.append(SearchResult(
                    score=score,
                    path=path,
                    description=content,
                    file_type="python",
                ))

            output_text = "\n".join(lines) + "\n"
            await self.output.append(output_text.encode())

            # Publish the first (top) result path to the blocking PATH file.
            if results:
                self.path_file.set_path(results[0][1])

            self.query_history.append(QueryRecord(
                query=query,
                timestamp=time.time(),
                results=search_results,
            ))

            if results and "CONTENT" in self.supplementary_outputs:
                top_path = results[0][1]
                try:
                    with open(top_path, "r", encoding="utf-8", errors="replace") as _f:
                        top_content = _f.read()
                except Exception as _e:
                    top_content = results[0][2]  # fall back to indexed (truncated) copy
                    await self.errors.post(f"CONTENT read error: {_e}\n".encode())
                content_file = self.supplementary_outputs["CONTENT"]
                content_file.add_block(top_content)
                content_file.mark_ready()

            await self._apply_plumbing(output_text)

            self.state = EmbedAgentState.IDLE

        except Exception as e:
            self.state = EmbedAgentState.ERROR
            self.last_error = str(e)
            await self.errors.post(f"Search error: {e}\n".encode())

        finally:
            await self.output.finish()

    # --- Plumbing ---

    def create_supplementary_output(
        self, name: str
    ) -> EmbedSupplementaryOutputFile:
        if name in self.supplementary_outputs:
            return self.supplementary_outputs[name]

        output_file = EmbedSupplementaryOutputFile(name.upper())
        self.supplementary_outputs[name] = output_file
        self.add(output_file)
        return output_file

    async def _apply_plumbing(self, content: str):
        """Apply plumbing rules to search output."""
        consumed_ranges = []

        def is_consumed(start: int, end: int) -> bool:
            for cs, ce in consumed_ranges:
                if start < ce and end > cs:
                    return True
            return False

        for rule in self.plumbing_rules:
            output_name = rule["output_name"]
            pattern = rule["pattern"]

            try:
                for m in re.finditer(pattern, content, re.DOTALL):
                    if is_consumed(m.start(), m.end()):
                        continue

                    groups = m.groupdict()
                    if (
                        output_name in groups
                        and groups[output_name] == output_name
                    ):
                        payload = (
                            groups.get("code")
                            or groups.get("content")
                            or groups.get(output_name)
                        )
                        if payload:
                            consumed_ranges.append((m.start(), m.end()))
                            if output_name in self.supplementary_outputs:
                                self.supplementary_outputs[
                                    output_name
                                ].add_block(payload)

            except Exception as e:
                await self.errors.post(
                    f"Plumbing rule failed for {output_name}: {e}\n".encode()
                )

        for sup in self.supplementary_outputs.values():
            sup.mark_ready()

    # --- Persistence ---

    def _save_descriptions(self):
        """Write descriptions to disk as JSON Lines (safe for multiline content)."""
        with open(
            self.embed_config.descriptions_file, "w", encoding="utf-8"
        ) as f:
            for path, desc in self.descriptions:
                f.write(json.dumps({"path": path, "content": desc}, ensure_ascii=False) + "\n")

    def _load_descriptions(self):
        """Load descriptions from disk (JSON Lines format)."""
        desc_file = self.embed_config.descriptions_file
        if not os.path.exists(desc_file):
            return

        self.descriptions.clear()
        with open(desc_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self.descriptions.append((obj["path"], obj["content"]))
                except (json.JSONDecodeError, KeyError):
                    # Legacy pipe-separated format fallback
                    if "|" in line:
                        path, desc = [x.strip() for x in line.split("|", 1)]
                        self.descriptions.append((path, desc))

    def save_to_disk(self):
        """Save descriptions and index to disk."""
        self._save_descriptions()
        if self.embedder.total > 0:
            self.embedder.save(
                self.embed_config.index_file,
                self.embed_config.metadata_file,
            )

    def load_from_disk(self) -> bool:
        """Load descriptions and index from disk."""
        self._load_descriptions()
        loaded = self.embedder.load(
            self.embed_config.index_file,
            self.embed_config.metadata_file,
        )
        if loaded:
            if not self.descriptions and self.embedder.metadata:
                self.descriptions = list(self.embedder.metadata)
        return loaded

    async def clear_all(self):
        """Clear all data."""
        import faiss

        self.descriptions.clear()
        self.query_history.clear()
        self.embedder.index = faiss.IndexFlatIP(EmbedBackend.DIMENSION)
        self.embedder.metadata.clear()

        for f in [
            self.embed_config.descriptions_file,
            self.embed_config.index_file,
            self.embed_config.metadata_file,
        ]:
            if os.path.exists(f):
                os.remove(f)

        for sup in self.supplementary_outputs.values():
            sup.clear()

        self.path_file.clear()

        await self.output.reset()
        self.state = EmbedAgentState.IDLE
        self.last_error = None

    def get_stats(self) -> str:
        lines = [
            f"state: {self.state.value}",
            f"descriptions: {len(self.descriptions)}",
            f"indexed: {self.embedder.total}",
            f"queries: {len(self.query_history)}",
            f"folders: {', '.join(self.embed_config.folders) or '(none)'}",
            f"model: {EmbedBackend.MODEL_NAME}",
            f"index_file: {self.embed_config.index_file}",
            f"metadata_file: {self.embed_config.metadata_file}",
            f"descriptions_file: {self.embed_config.descriptions_file}",
        ]
        return "\n".join(lines) + "\n"

    async def load_describe(self):
        """Unload search model then eagerly load the describe model."""
        self.state = EmbedAgentState.IDLE
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.describer._ensure_loaded
            )
            await self.output.reset()
            await self.output.append(b"Describe model ready.")
        except Exception as e:
            self.state = EmbedAgentState.ERROR
            self.last_error = str(e)
            await self.errors.post(f"load describe error: {e}".encode())
        finally:
            await self.output.finish()

    async def load_search(self):
        """Unload describe model then eagerly load the search model."""
        self.state = EmbedAgentState.IDLE
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.embedder._ensure_loaded
            )
            await self.output.reset()
            await self.output.append(b"Search model ready.")
        except Exception as e:
            self.state = EmbedAgentState.ERROR
            self.last_error = str(e)
            await self.errors.post(f"load search error: {e}".encode())
        finally:
            await self.output.finish()

    async def stop(self):
        """Clean shutdown."""
        self.describer.unload()
        self.embedder.unload()