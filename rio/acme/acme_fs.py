"""
Acme Filesystem Interface

Each acme window is a directory with control files:

  /n/rioa/acme/           - directory: lists open window IDs
  /n/rioa/acme/0/         - window 0 directory
  /n/rioa/acme/0/ctl      - control file: write commands (Get, Put, Del, Code, Main, Clear, show text, ai ...)
  /n/rioa/acme/0/text     - text content: read/write the text pane
  /n/rioa/acme/0/code     - accumulated code since window creation (read-only, appended via exec)
  /n/rioa/acme/0/exec     - write code here to execute it
  /n/rioa/acme/0/path     - read/write the current file path
  /n/rioa/acme/0/error    - last error message (read-only)

BUFFERING:
  All writable files buffer data across 9P chunks and only dispatch
  the action on clunk (fid close). This prevents fragmented writes from
  triggering partial executions. Same pattern as AgentInputFile/AgentHistoryFile.

Interaction patterns:
  # Execute code in window 0:
  echo 'label = QLabel("hello"); layout.addWidget(label)' > /n/rioa/acme/0/exec

  # Read current file into text pane:
  echo 'Get' > /n/rioa/acme/0/ctl

  # Show the code pane:
  echo 'show code' > /n/rioa/acme/0/ctl

  # AI interaction:
  echo 'ai make a calculator' > /n/rioa/acme/0/ctl

  # Set path:
  echo '/home/user/project' > /n/rioa/acme/0/path

  # Read accumulated code:
  cat /n/rioa/acme/0/code

  # Read text pane content:
  cat /n/rioa/acme/0/text

All toolbar words are commands: mid-click plumbs them to ctl.
"""

from PySide6.QtCore import QTimer

from core.files import SyntheticDir, SyntheticFile
from core.types import FidState


# ---------------------------------------------------------------------------
# Buffered write mixin
# ---------------------------------------------------------------------------

class _BufferedWriteMixin:
    """
    Mixin providing 9P-safe buffered writes.
    
    9P sends large writes as multiple chunks. If we dispatch on every
    write() call, code arrives fragmented. Instead we buffer chunks
    and only dispatch the complete payload on clunk (fid close).
    
    Subclasses implement _on_complete_write(text: str) to handle the
    assembled payload.
    """

    def _init_buffers(self):
        self._write_buffers = {}  # fid_key -> bytearray

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        """Buffer incoming 9P write chunks."""
        fid_key = id(fid)

        if fid_key not in self._write_buffers:
            self._write_buffers[fid_key] = bytearray()

        buf = self._write_buffers[fid_key]

        # Offset 0 with existing data = new write sequence
        if offset == 0 and len(buf) > 0:
            buf.clear()

        # Extend buffer to fit
        if offset + len(data) > len(buf):
            buf.extend(b'\x00' * (offset + len(data) - len(buf)))
        buf[offset:offset + len(data)] = data

        return len(data)

    async def clunk(self, fid: FidState):
        """Dispatch the complete buffered payload on fid close."""
        fid_key = id(fid)
        buf = self._write_buffers.pop(fid_key, None)

        if not buf:
            return

        text = bytes(buf).decode('utf-8', errors='replace').strip()
        if text:
            self._on_complete_write(text)

    def _on_complete_write(self, text: str):
        """Override in subclass to handle the complete write payload."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Per-window files
# ---------------------------------------------------------------------------

class AcmeCtlFile(_BufferedWriteMixin, SyntheticFile):
    """
    Control file for an acme window.
    
    Writing a command string executes it in the window.
    Reading returns the current window state summary.
    Writes are buffered until clunk so multi-chunk commands arrive intact.
    
    Commands:
      Get          - read file at path into text pane
      Put          - write text pane content to file at path
      Del          - close window
      Code         - raise text pane showing accumulated code
      Main         - raise the graphical/UI pane
      Clear        - erase accumulated code
      show text    - show text pane
      show code    - show code in text pane
      show err     - show last error in text pane
      ai <prompt>  - send prompt to AI agent, execute result
    """

    def __init__(self, acme_window):
        super().__init__("ctl")
        self._init_buffers()
        self.acme_window = acme_window

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        """Read returns a summary of window state"""
        w = self.acme_window
        lines = []
        lines.append(f"id {w.window_id}")
        lines.append(f"path {w.path or '(none)'}")
        lines.append(f"dirty {getattr(w, '_dirty', False)}")
        data = ("\n".join(lines) + "\n").encode("utf-8")
        return data[offset:offset + count]

    def _on_complete_write(self, text: str):
        """Dispatch complete command on Qt thread"""
        QTimer.singleShot(0, lambda cmd=text: self.acme_window.execute_ctl_command(cmd))


class AcmeTextFile(_BufferedWriteMixin, SyntheticFile):
    """
    The text pane content.
    
    Reading returns current text pane content.
    Writing sets the text pane content and raises the text pane.
    Writes are buffered until clunk so large text arrives intact.
    """

    def __init__(self, acme_window):
        super().__init__("text")
        self._init_buffers()
        self.acme_window = acme_window

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        w = self.acme_window
        text = w.get_text_content()
        data = text.encode("utf-8")
        return data[offset:offset + count]

    def _on_complete_write(self, text: str):
        """Set text pane content on Qt thread"""
        QTimer.singleShot(0, lambda t=text: self.acme_window.set_text_content_and_raise(t))


class AcmeCodeFile(SyntheticFile):
    """
    Accumulated code since the window was created.
    
    Reading returns the full concatenated code.
    Writing is ignored (code is appended via successful exec only).
    """

    def __init__(self, acme_window):
        super().__init__("code")
        self.acme_window = acme_window

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        code = self.acme_window.accumulated_code
        data = code.encode("utf-8")
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        # Writing to code is a no-op — code is appended via exec
        return len(data)


class AcmeExecFile(_BufferedWriteMixin, SyntheticFile):
    """
    Execute code in this window.
    
    Writing code here:
      - Buffered across 9P chunks, dispatched on clunk (fid close)
      - On success: appends to accumulated code, raises the graphical pane
      - On error: writes error to error file, shows error in text pane
    
    Reading returns empty (exec is write-only).
    """

    def __init__(self, acme_window):
        super().__init__("exec")
        self._init_buffers()
        self.acme_window = acme_window

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        return b""

    def _on_complete_write(self, text: str):
        """Execute complete code payload on Qt thread"""
        QTimer.singleShot(0, lambda c=text: self.acme_window.execute_code_from_fs(c))


class AcmePathFile(_BufferedWriteMixin, SyntheticFile):
    """
    Current file path for this window.
    
    Reading returns the path.
    Writing sets the path (updates toolbar).
    Writes are buffered until clunk so long paths arrive intact.
    """

    def __init__(self, acme_window):
        super().__init__("path")
        self._init_buffers()
        self.acme_window = acme_window

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        path = self.acme_window.path or ""
        data = (path + "\n").encode("utf-8")
        return data[offset:offset + count]

    def _on_complete_write(self, text: str):
        """Set path on Qt thread"""
        QTimer.singleShot(0, lambda p=text: self.acme_window.set_path(p))


class AcmeErrorFile(SyntheticFile):
    """
    Last error from code execution.
    
    Reading returns the last error.
    Writing is ignored (errors are set by exec failures).
    """

    def __init__(self, acme_window):
        super().__init__("error")
        self.acme_window = acme_window

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        error = self.acme_window.last_error or ""
        data = error.encode("utf-8")
        return data[offset:offset + count]

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        # Errors are set by the system, not by external writes
        return len(data)


# ---------------------------------------------------------------------------
# Per-window directory
# ---------------------------------------------------------------------------

class AcmeWindowDir(SyntheticDir):
    """
    Directory for a single acme window: /n/rioa/acme/{id}/
    
    Contains: ctl, text, code, exec, path, error
    """

    def __init__(self, window_id, acme_window):
        super().__init__(str(window_id))
        self.window_id = window_id
        self.acme_window = acme_window

        # Create all the files
        self.ctl_file = AcmeCtlFile(acme_window)
        self.text_file = AcmeTextFile(acme_window)
        self.code_file = AcmeCodeFile(acme_window)
        self.exec_file = AcmeExecFile(acme_window)
        self.path_file = AcmePathFile(acme_window)
        self.error_file = AcmeErrorFile(acme_window)

        self.add(self.ctl_file)
        self.add(self.text_file)
        self.add(self.code_file)
        self.add(self.exec_file)
        self.add(self.path_file)
        self.add(self.error_file)


# ---------------------------------------------------------------------------
# Top-level /n/rioa/acme/ directory
# ---------------------------------------------------------------------------

class AcmeDir(SyntheticDir):
    """
    Directory node for /n/rioa/acme/
    
    Lists all open acme windows by their integer ID.
    Each window is a subdirectory with control files.
    
    /n/rioa/acme/
    ├── 0/          # Window 0 directory
    │   ├── ctl
    │   ├── text
    │   ├── code
    │   ├── exec
    │   ├── path
    │   └── error
    ├── 1/          # Window 1 directory
    │   └── ...
    └── ...
    """

    def __init__(self):
        super().__init__("acme")
        self._acme = None
        self._windows = {}  # window_id -> AcmeWindowDir

    def set_acme(self, acme):
        """Set reference to the Acme core widget"""
        self._acme = acme

    def register_window(self, window_id, acme_window):
        """Register an acme window — creates the directory with all files"""
        wdir = AcmeWindowDir(window_id, acme_window)
        self._windows[window_id] = wdir
        self.add(wdir)
        print(f"[AcmeFS] Registered window {window_id} at acme/{window_id}/")
        return wdir

    def unregister_window(self, window_id):
        """Remove a window directory from the filesystem"""
        if window_id in self._windows:
            name = str(window_id)
            try:
                self.remove(name)
            except KeyError:
                pass
            del self._windows[window_id]
            print(f"[AcmeFS] Unregistered window {window_id}")


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_acme_dir = None

def get_acme_dir():
    """Get or create the global AcmeDir instance"""
    global _acme_dir
    if _acme_dir is None:
        _acme_dir = AcmeDir()
    return _acme_dir