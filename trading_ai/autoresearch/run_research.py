"""
Runner script for AutoResearch experiments.
"""

import subprocess
import sys
import os
from datetime import datetime


def run_autoresearch_experiment():
    """
    Run a single AutoResearch experiment.
    """
    print(f"Starting AutoResearch experiment at {datetime.now()}")

    try:
        # Run the researcher script and capture output
        result = subprocess.run(
            [sys.executable, "autoresearch/researcher.py"],
            capture_output=True,
            text=True,
            cwd=".",
        )

        # Print stdout
        print(result.stdout)

        # Print stderr if there were errors
        if result.stderr:
            print("Errors:", result.stderr)

        # Check return code
        if result.returncode == 0:
            print("Experiment completed successfully")
        else:
            print(f"Experiment failed with return code {result.returncode}")

    except Exception as e:
        print(f"Error running experiment: {e}")


if __name__ == "__main__":
    run_autoresearch_experiment()
