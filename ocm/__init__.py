"""Ontology-Constrained Memory (OCM).

A reusable, pluggable, write-time-governed memory module for long-horizon LLM
agents. OCM converts unstructured input into typed graph assertions, validates
them against an ontology and constraint rules, runs every candidate through a
contradiction gate before commit, and attaches provenance to everything it
accepts.

See the design document for the full architecture (Write Pipeline W1-W8,
Retrieval Pipeline R0-R4, Ontology layer, storage, API, agent, and evaluation).
"""

__version__ = "0.1.0"
