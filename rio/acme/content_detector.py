"""
Content type detection for determining how to display files
"""

import os
import mimetypes


def _safe_probe_9p(func, path, timeout=0.4):
    """Run a filesystem probe with a timeout for 9P paths.
    
    Any file under /n/ might be a blocking synthetic file (StreamFile,
    SupplementaryOutputFile, etc.).  We can't maintain a static list
    because supplementary outputs have arbitrary user-defined names.
    
    Returns the probe result, or None on timeout/error.
    """
    import concurrent.futures
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(func, path)
            return future.result(timeout=timeout)
    except (concurrent.futures.TimeoutError, Exception):
        return None


def _is_9p_path(path):
    """Check if path is under the 9P mount at /n."""
    return path == '/n' or path.startswith('/n/')


def detect_content_type(path):
    """
    Detect content type of a file
    Returns: 'directory', 'image', 'video', 'audio', '3d', 'pdf', 'text', or None
    """
    if not path:
        return None
    
    # For 9P paths, avoid any probe that might block on synthetic files.
    # Use extension-based detection first, and timeout-protect stat calls.
    if _is_9p_path(path):
        # Extension-based detection first (never blocks)
        ext = os.path.splitext(path)[1].lower()
        if ext:
            if ext in ['.obj', '.glb', '.gltf', '.stl', '.ply']:
                return '3d'
            if ext == '.svg':
                return 'image'
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type:
                if mime_type.startswith('image/'):
                    return 'image'
                elif mime_type.startswith('video/'):
                    return 'video'
                elif mime_type.startswith('audio/'):
                    return 'audio'
                elif mime_type == 'application/pdf':
                    return 'pdf'
                elif mime_type.startswith('text/'):
                    return 'text'

        # No extension — try stat with timeout protection.
        # NEVER call bare os.path.isdir/exists on 9P paths; the stat
        # syscall can block indefinitely on synthetic files.
        is_dir = _safe_probe_9p(lambda p: os.path.isdir(p), path)
        if is_dir:
            return 'directory'

        # If the probe timed out or the path isn't a directory,
        # return None and let the caller decide.
        return None
    
    if not os.path.exists(path):
        return None
    
    if os.path.isdir(path):
        return 'directory'
    
    # Check file extension
    ext = os.path.splitext(path)[1].lower()
    
    # 3D model files
    if ext in ['.obj', '.glb', '.gltf', '.stl', '.ply']:
        return '3d'
    
    # SVG images (handle specially before mimetypes)
    if ext == '.svg':
        return 'image'
    
    # Use mimetypes for standard files
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type:
        if mime_type.startswith('image/'):
            return 'image'
        elif mime_type.startswith('video/'):
            return 'video'
        elif mime_type.startswith('audio/'):
            return 'audio'
        elif mime_type == 'application/pdf':
            return 'pdf'
        elif mime_type.startswith('text/'):
            return 'text'
    
    # Default to text for unknown types
    return 'text'


def is_executable_code(code):
    """
    Check if code looks like it should be executed
    Returns True if it contains Qt widget creation
    """
    code = code.strip()
    
    if not code:
        return False
    
    # Check for UI widget creation keywords that indicate executable code
    ui_keywords = [
        'QPushButton', 'QLabel', 'QSlider', 'QLineEdit', 
        'layout.addWidget', 'QVBoxLayout', 'QHBoxLayout',
        'QCheckBox', 'QRadioButton', 'QComboBox',
        'QMediaPlayer', 'QVideoWidget', 'QWebEngineView',
        'ImageViewWidget', 'Mesh3DWidget', 'QTextEdit'
    ]
    
    for keyword in ui_keywords:
        if keyword in code:
            return True
    
    return False