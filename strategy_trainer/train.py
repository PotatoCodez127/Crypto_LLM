"""
train.py - The Strategy Sandbox Evaluator
Executes the Walk-Forward Judge to evaluate AI strategy performance.
"""

import io
import sys

from prepare import run_walk_forward_optimization

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if __name__ == "__main__":
    # run_walk_forward_optimization() already prints FINAL_RESULT internally
    run_walk_forward_optimization()
