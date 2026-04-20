# Autonomous Quantitative Trading Research

This is an experiment to have an autonomous AI agent conduct quantitative trading research and strategy optimization.

## Setup

To set up a new experiment, work with the user to:
1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). 
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `prepare.py` — The testing framework, data loader, and judge. Do not modify.
   - `strategy.py` — **The ONLY file you modify.** Contains the `get_signals` function.
   - `train.py` — The execution script. You run this to test `strategy.py`.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row.
5. **Confirm and go**: Confirm setup looks good and kick off the loop.

## Experimentation

Your goal is to optimize the trading strategy to achieve the highest possible `FINAL_RESULT`. 

**What you CAN do:**
- Modify `strategy.py`. You can change indicators, stop-loss logic, entry/exit conditions, parameters, and risk management logic.

**What you CANNOT do:**
- Modify `prepare.py` or `train.py`.
- Modify `results.tsv` to log your experiment results.
- Install new packages. You can only use what is already available (pandas, numpy, etc.).
- Break the `get_signals(df)` function signature. It must always accept a dataframe and return a dataframe with a `signal` column (1 for long, -1 for short, 0 for flat).

**The Metric:**
When you execute the test, the script will output a comprehensive performance report and end with:
`FINAL_RESULT:<score>`
Your goal is to maximize this score.

**Simplicity criterion**: All else being equal, simpler is better. Do not overfit the strategy with 20 convoluted rules just to squeeze out a tiny fraction of a point. 

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).
The TSV has a header row and 4 columns:

`commit    final_result    status    description`

1. git commit hash (short, 7 chars)
2. FINAL_RESULT achieved (e.g. 1.2345) — use 0.0000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

## The Experiment Loop

Because you are running inside Aider, you must use Aider's `/run` command to execute tests. 

LOOP FOREVER:
1. Tune `strategy.py` with an experimental idea by directly hacking the code.
2. Commit the changes to git.
3. Run the experiment using Aider's run command: `/run python offline_training/train.py`
4. Read the output provided by Aider. Look for `FINAL_RESULT:`.
5. If the run crashed (Python traceback), attempt a quick fix. If it's fundamentally broken, log "crash" and move on.
6. Record the results in the `results.tsv` file (leave it untracked by git).
7. If `FINAL_RESULT` improved (HIGHER is better), you "advance" the branch, keeping the git commit.
8. If `FINAL_RESULT` is equal or worse, use `git reset --hard HEAD~1` to rewind back to where you started.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. You are autonomous. If you run out of ideas, think harder—try new indicator combinations (Bollinger Bands, VWAP, ADX), tweak the adaptive ATR math, or change trend filters. The loop runs until the human interrupts you.