"""
ACME - A generative text editor

Each acme window is a directory under /n/rioa/acme/:

  /n/rioa/acme/           - lists open windows
  /n/rioa/acme/0/         - window 0 directory
  /n/rioa/acme/0/ctl      - control: write commands (Get, Put, Del, Code, Main, Clear, ai ...)
  /n/rioa/acme/0/text     - text pane content (read/write)
  /n/rioa/acme/0/code     - accumulated code since window start (read-only)
  /n/rioa/acme/0/exec     - write code to execute
  /n/rioa/acme/0/path     - current file path (read/write)
  /n/rioa/acme/0/error    - last error (read-only)

Every toolbar word is a ctl command. Mid-click plumbs the word to ctl.

AI interaction:
  echo 'ai make a calculator' > /n/rioa/acme/0/ctl
  # equivalent to:
  echo 'make a calculator' > $acme_agent/input && cat $acme_agent/output > /n/rioa/acme/0/exec
"""

__version__ = "3.0.0"
__author__ = "ACME Development Team"

from .acme_core import Acme
from .acme_fs import AcmeDir, AcmeWindowDir

__all__ = ['Acme', 'AcmeDir', 'AcmeWindowDir']