"""
QuickFile — /n/{workspace}/scene/quick

Write a file path to display it on the scene using the appropriate
viewer for the file extension.

Usage:
    echo /path/to/image.png  > /n/rioa/scene/quick
    echo /path/to/report.pdf > /n/rioa/scene/quick
    echo /path/to/video.mp4  > /n/rioa/scene/quick
    echo /path/to/model.stl  > /n/rioa/scene/quick
    echo /path/to/notes.txt  > /n/rioa/scene/quick
    echo /path/to/music.mp3  > /n/rioa/scene/quick
    echo /some/directory/    > /n/rioa/scene/quick

    # Peribus apps — shows a confirmation popup before running:
    echo /some/peribus/apps/myapp.py > /n/rioa/scene/quick

Read:
    cat /n/rioa/scene/quick   # returns last displayed path
"""

import asyncio
import os
from core.types import FidState
from core.files import SyntheticFile

# Extension sets
_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp',
               '.tiff', '.tif', '.svg', '.ico'}
_VIDEO_EXTS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv',
               '.webm', '.m4v', '.mpeg', '.mpg'}
_AUDIO_EXTS = {'.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma', '.opus'}
_3D_EXTS    = {'.stl', '.obj', '.ply', '.glb', '.gltf', '.fbx', '.dae', '.3ds', '.off'}
_PDF_EXTS   = {'.pdf'}
_TEXT_EXTS  = {
    '.txt', '.md', '.rst', '.csv', '.tsv', '.log', '.json', '.yaml', '.yml',
    '.toml', '.ini', '.cfg', '.conf', '.xml', '.html', '.htm', '.css',
    '.js', '.ts', '.py', '.sh', '.bash', '.zsh', '.fish',
    '.c', '.cc', '.cpp', '.h', '.hpp', '.rs', '.go', '.rb', '.java',
    '.kt', '.swift', '.r', '.m', '.lua', '.pl', '.php', '.sql',
    '.diff', '.patch', '.env', '.makefile', '.mk',
}


def _is_peribus_app(path: str) -> bool:
    """
    Return True if path matches .../peribus/apps/<filename>.py
    Handles both absolute and relative paths.
    """
    parts = path.replace("\\", "/").split("/")
    # Need at least: [..., "peribus", "apps", "<name>.py"]
    if len(parts) < 3:
        return False
    return (
        parts[-3] == "peribus"
        and parts[-2] == "apps"
        and parts[-1].endswith(".py")
    )


def _pick_generator(path: str):
    """
    Return (generator_fn, resolved_path).
    Falls back to generate_quick_file_content for unknown extensions.
    """
    from .quick_generators import (
        generate_quick_image_viewer,
        generate_quick_video_player,
        generate_quick_audio_player,
        generate_quick_3d_viewer,
        generate_quick_pdf_viewer,
        generate_quick_file_content,
        generate_quick_directory_listing,
    )

    path = path.strip()
    if not path:
        return None, path

    if not os.path.isabs(path):
        path = os.path.abspath(path)

    # Directory?
    try:
        if os.path.isdir(path):
            return generate_quick_directory_listing, path
    except OSError:
        pass

    ext = os.path.splitext(path)[1].lower()

    if ext in _IMAGE_EXTS:
        return generate_quick_image_viewer, path
    if ext in _VIDEO_EXTS:
        return generate_quick_video_player, path
    if ext in _AUDIO_EXTS:
        return generate_quick_audio_player, path
    if ext in _3D_EXTS:
        return generate_quick_3d_viewer, path
    if ext in _PDF_EXTS:
        return generate_quick_pdf_viewer, path

    # Text / source / data / unknown extension → plain text viewer
    return generate_quick_file_content, path


def _show_run_confirmation(filename: str) -> bool:
    """
    Show a modal QMessageBox asking the user to confirm running a peribus app.
    Returns True if the user clicked Run, False otherwise.
    Must be called from the Qt main thread.
    """
    from PySide6.QtWidgets import QMessageBox, QPushButton

    msg = QMessageBox()
    msg.setWindowTitle("Run App")
    msg.setText(f"Run <b>{filename}</b>?")
    msg.setInformativeText(
        f"This will execute <code>{filename}</code> via the scene parser."
    )
    msg.setIcon(QMessageBox.Icon.Question)

    run_btn   = msg.addButton("Run",    QMessageBox.ButtonRole.AcceptRole)
    cancel_btn = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)  # noqa: F841
    msg.setDefaultButton(run_btn)

    msg.exec()
    return msg.clickedButton() is run_btn


class QuickFile(SyntheticFile):
    """
    /scene/quick — display any file on the scene by writing its path.

    All viewers embed their widget in the QGraphicsScene via
    graphics_scene.addWidget() and register with scene_manager so
    undo/redo/clear track them correctly.

    Special case — peribus apps:
        Writing .../peribus/apps/<name>.py shows a confirmation popup
        and, if confirmed, pipes the file's source code directly into
        /n/{workspace}/scene/parse (i.e. the executor), bypassing the
        normal viewer pipeline.
    """

    def __init__(self, scene_manager, executor, stdout_file=None, stderr_file=None,
                 parse_file=None):
        super().__init__("quick")
        self.scene_manager = scene_manager
        self.executor = executor
        self.stdout_file = stdout_file
        self.stderr_file = stderr_file
        self.parse_file   = parse_file   # SyntheticFile for /scene/parse
        self._last_path: str = ""
        self._write_bufs: dict[int, bytearray] = {}
        self._last_video: object = None   # direct ref to _VideoInstance; bypasses scene traversal

    # ── Read ─────────────────────────────────────────────────────────────────

    async def read(self, fid: FidState, offset: int, count: int) -> bytes:
        data = (self._last_path + "\n").encode() if self._last_path else b""
        return data[offset:offset + count]

    # ── Write (accumulate) ───────────────────────────────────────────────────

    async def write(self, fid: FidState, offset: int, data: bytes) -> int:
        if fid.fid not in self._write_bufs:
            self._write_bufs[fid.fid] = bytearray()
        self._write_bufs[fid.fid].extend(data)
        return len(data)

    # ── Clunk (act on close) ─────────────────────────────────────────────────

    # In quick_file.py -> QuickFile class
    def clunk(self, fid: FidState):
        buf = self._write_bufs.pop(fid.fid, None)
        if not buf:
            return
        
        raw_input = buf.decode("utf-8", errors="replace").strip()
        if not raw_input:
            return

        # Split by whitespace to handle: echo path1 path2 > quick
        # Or split by \n to handle multi-line writes
        paths = raw_input.split() 
        
        for path in paths:
            print(f"[QuickFile] opening: {path!r}")
            asyncio.create_task(self._display(path))

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _display(self, path: str):
        # ── Peribus app path ─────────────────────────────────────────────────
        if not os.path.isabs(path):
            resolved_check = os.path.abspath(path.strip())
        else:
            resolved_check = path.strip()

        if _is_peribus_app(resolved_check):
            await self._run_peribus_app(resolved_check)
            return

        # ── Normal viewer path ───────────────────────────────────────────────
        generator, resolved = _pick_generator(path)

        if generator is None:
            await self._err("[quick] empty path\n")
            return

        try:
            exists = os.path.exists(resolved)
        except OSError:
            exists = resolved.startswith("/n/")

        if not exists:
            await self._err(f"[quick] not found: {resolved}\n")
            return

        # ── HARDWARE RELEASE ─────────────────────────────────────────────────
        # Stop the previous video player directly — no scene traversal needed.
        # scene_manager.items may not expose proxy attributes at all, so the
        # old sweep was silently finding nothing and skipping stop() entirely.
        # Note: _last_video is only populated in the executor fallback path
        # (when parse_file is not wired up).
        if self._last_video is not None:
            try:
                self._last_video.stop()
            except Exception:
                pass
            self._last_video = None

        # Clear visual elements
        #await self.scene_manager.clear()

        # Yield to the Qt event loop so internal pipeline teardown and any
        # deferred deletions are processed before the new player initialises.
        import gc
        gc.collect()
        await asyncio.sleep(0)

        # ── EXECUTE NEW VIEWER ───────────────────────────────────────────────
        try:
            code = generator(resolved)
        except Exception as e:
            await self._err(f"[quick] generator error for {resolved!r}: {e}\n")
            return

        if self.parse_file is not None:
            self._last_path = resolved
            self.parse_file.dispatch(code)
            print(f"[QuickFile] dispatched to parse: {resolved!r}")
        else:
            # Fallback: execute directly when parse_file is not wired up.
            # Use builtins as a side-channel registry — always reachable from
            # any exec() namespace without an import. The generated video code
            # writes _widget to builtins._quick_video_instance; we read it back.
            import builtins
            builtins._quick_video_instance = None  # clear before execute

            result = await self.executor.execute(code)

            if result.success:
                self._last_path = resolved
                if generator.__name__ == "generate_quick_video_player":
                    self._last_video = getattr(builtins, "_quick_video_instance", None)
                    builtins._quick_video_instance = None
                if self.stdout_file:
                    await self.stdout_file.post(f"✓ quick: {resolved}\n".encode())
                    self.stdout_file.mark_ready()
            else:
                await self._err(result.error)

    # ── Peribus app runner ───────────────────────────────────────────────────

    async def _run_peribus_app(self, resolved: str):
        """
        Show a confirmation dialog, then — if the user confirms — read the
        .py file and write its source code to /n/{workspace}/scene/parse
        (exposed here as self.parse_file).
        """
        filename = os.path.basename(resolved)

        # Verify the file actually exists
        if not os.path.isfile(resolved):
            await self._err(f"[quick] peribus app not found: {resolved}\n")
            return

        # The confirmation dialog must run on the Qt main thread.
        # asyncio.get_event_loop().run_in_executor with None (thread pool)
        # is safe for Qt modal dialogs when called from a Qt-integrated loop.
        loop = asyncio.get_event_loop()
        confirmed = await loop.run_in_executor(None, _show_run_confirmation, filename)

        if not confirmed:
            print(f"[QuickFile] run cancelled by user: {filename!r}")
            if self.stdout_file:
                await self.stdout_file.post(
                    f"[quick] cancelled: {filename}\n".encode()
                )
                self.stdout_file.mark_ready()
            return

        # Read the app source
        try:
            with open(resolved, "r", encoding="utf-8") as fh:
                source = fh.read()
        except OSError as exc:
            await self._err(f"[quick] cannot read {resolved!r}: {exc}\n")
            return

        # Route through parse_file if wired up, otherwise fall back to executor
        if self.parse_file is not None:
            self._last_path = resolved
            self.parse_file.dispatch(source)
            print(f"[QuickFile] peribus app dispatched to parse: {filename!r}")
        else:
            # Fallback: execute directly through the executor
            print(
                f"[QuickFile] parse_file not wired — executing {filename!r} directly"
            )
            result = await self.executor.execute(source)
            if result.success:
                self._last_path = resolved
                if self.stdout_file:
                    await self.stdout_file.post(
                        f"✓ quick (peribus/direct): {resolved}\n".encode()
                    )
                    self.stdout_file.mark_ready()
            else:
                await self._err(result.error)

    async def _err(self, msg: str):
        print(msg.rstrip())
        if self.stderr_file:
            await self.stderr_file.post(msg.encode())
            self.stderr_file.mark_ready()