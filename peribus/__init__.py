"""
peribus/0.1 — the mycelium layer

A peer-to-peer social filesystem mounted at /n/peribus.
Discovery is semantic (embedding-based), connections grow with attention.

The whole thing is a SyntheticDir tree served over 9P, like everything
else in the rio stack. The daemon (peribusd) is the only process that
needs to run; mounting it makes the network appear as a filesystem.
"""

PROTOCOL_VERSION = "peribus/0.1"