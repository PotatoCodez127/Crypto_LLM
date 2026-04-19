# autoresearch (Quant Trading Edition)

This is an experiment to have the LLM do its own research and optimize a quantitative trading strategy.

## Setup
To set up a new experiment, work with the user to:
1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed constants, data fetching (ccxt), and risk evaluation. Do not modify.
   - `train.py` — the file you modify. It contains the strategy logic, indicator math, and the execution loop.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
5. **Confirm and go**: Confirm setup looks good and start.

## Experimentation
The training script runs for a limited time budget. You launch it as: `uv run --active train.py`.

**What you CAN do:**
- Modify `train.py` — this is the ONLY file you edit. Change the indicators, entry/exit logic, stop losses, position sizing, or risk parameters. 

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed risk constraints (like maximum drawdown limits) and data loading.
- Add new external dependencies not already in the environment.

**The goal is simple: get the HIGHEST FINAL_RESULT.** If your strategy fails the hardcoded risk constraints in `prepare.py`, it will output `FINAL_RESULT: 0.0`. You must debug the strategy logic in `train.py` to survive the risk checks and achieve a positive score. 

## Output format
Once the script finishes, you extract the key metric from the log file:
```bash
grep "^FINAL_RESULT:" run.log
```

## Logging results
When an experiment is done, log it to `results.tsv` (tab-separated):
```
commit	final_result	status	description
```
1. git commit hash (short, 7 chars)
2. final_result achieved (e.g. 15.42) — use 0.0 for crashes or risk rejections
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

## The experiment loop
LOOP FOREVER:
1. Tune `train.py` with an experimental strategy idea.
2. git commit
3. Run the experiment: `uv run --active train.py > run.log 2>&1`
4. Read out the results: `grep "^FINAL_RESULT:" run.log`
5. Record the results in the tsv.
6. If `FINAL_RESULT` **improved (higher)**, you "advance" the branch, keeping the git commit.
7. If `FINAL_RESULT` is equal, worse, or 0.0, git reset back to where you started.

**NEVER STOP**: Run this loop indefinitely until the human stops you.