#!/bin/bash
set -e

PERIOD=${1:-1y}
DESCRIPTION=${2:-"no description"}

echo "Running experiment with period=$PERIOD, description: $DESCRIPTION"

# Run strategy evaluation
python prepare.py --period $PERIOD > run.log 2>&1

# Extract metrics
SCORE=$(grep "FINAL_RESULT:" run.log | tail -1 | sed 's/FINAL_RESULT://')
DRAWDOWN=$(grep "Max Drawdown" run.log | tail -1 | sed 's/.*Max Drawdown  : //' | sed 's/%//')
TRADES=$(grep "Total Trades" run.log | tail -1 | sed 's/.*Total Trades  : //')
COMMIT=$(git rev-parse --short HEAD)

# Determine status (keep by default)
STATUS="keep"

# Append to results.tsv
echo -e "$COMMIT\t$SCORE\t$DRAWDOWN\t$TRADES\t$STATUS\t$DESCRIPTION" >> results.tsv

echo "Results logged:"
tail -1 results.tsv
