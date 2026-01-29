"""Affinetes integration for Templar Crusades.

URL-Based Architecture:
- Validator owns the evaluation image (templar-eval)
- Miner's code is downloaded from their committed URL
- Supports docker (local) and basilica (remote) modes
"""

from .runner import AffinetesRunner, EvaluationResult

__all__ = ["AffinetesRunner", "EvaluationResult"]
