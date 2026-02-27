#!/usr/bin/env python3
"""
Acme - A generative text editor
Everything is a program that writes to /n/rioa/acme/{id} files

Acme windows are exposed as files under /n/rioa/acme/{id}.
AI interaction uses filesystem read/write operations.
"""

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow

from .acme_core import Acme

# Global window ID counter
_next_window_id = 0

def get_next_window_id():
    """Get next unique window ID"""
    global _next_window_id
    wid = _next_window_id
    _next_window_id += 1
    return wid


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = QMainWindow()
    window.setWindowTitle("Acme - Generative")
    window.setGeometry(100, 100, 1200, 800)
    
    # Create Acme widget (registers with /n/rioa/acme/ filesystem)
    acme = Acme()
    window.setCentralWidget(acme)
    
    window.show()
    sys.exit(app.exec())