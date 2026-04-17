# AutoResearch for Cryptocurrency Trading Strategies

This is an experiment to have the AI autonomously improve cryptocurrency trading strategies.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr17`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `autoresearch/strategy_evaluator.py` — fixed evaluation framework for strategies. Do not modify.
   - `autoresearch/researcher.py` — the file you modify. Strategy generation, optimization, and research loop.
4. **Verify data exists**: Check that historical data is available in `./data/`. If not, tell the human to run data collection.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs backtests on historical data. The evaluation script runs for a **variable time budget** but should complete in reasonable time. You launch it simply as: `python autoresearch/researcher.py`.

**What you CAN do:**
- Modify `autoresearch/researcher.py` — this is the only file you edit. Everything is fair game: strategy generation, optimization algorithms, hyperparameters, etc.

**What you CANNOT do:**
- Modify `autoresearch/strategy_evaluator.py`. It is read-only. It contains the fixed evaluation framework.
- Install new packages or add dependencies. You can only use what's already in `requirements.txt`.
- Modify the evaluation harness. The `evaluate_strategy` function in `strategy_evaluator.py` is the ground truth metric.

**The goal is simple: get the highest risk-adjusted return.** We measure this with a combination of:
- Total return
- Sharpe ratio 
- Maximum drawdown
- Consistency of returns

**Capital preservation** is a hard constraint. Any strategy that has a max drawdown > 50% is automatically rejected.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the evaluation script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
total_return:          1.25
sharpe_ratio:          1.8
max_drawdown:          0.15
consistency:           0.75
risk_adjusted_score:   1.45
experiment_time:       45.2
---
```

## Logging results

When an experiment is done, log it to `autoresearch/results.tsv` (tab-separated).

The TSV has a header row and 6 columns:

```
commit	risk_adjusted_score	total_return	sharpe_ratio	max_drawdown	status	description
```

1. git commit hash (short, 7 chars)
2. risk_adjusted_score achieved (e.g. 1.45) — use 0.000000 for crashes
3. total_return (e.g. 1.25 for 25% total return)
4. sharpe_ratio (e.g. 1.8)
5. max_drawdown (e.g. 0.15 for 15% drawdown) — use 1.0 for disqualifying violations (>50%)
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	risk_adjusted_score	total_return	sharpe_ratio	max_drawdown	status	description
a1b2c3d	1.45	1.25	1.8	0.15	keep	baseline strategy
b2c3d4e	1.52	1.32	1.9	0.14	keep	increased position sizing
c3d4e5f	0.85	0.95	1.1	0.25	discard	reduced risk tolerance
d4e5f6g	0.000000	0.0	0.0	1.0	crash	increased leverage (blowup)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr17`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `autoresearch/researcher.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python autoresearch/researcher.py > autoresearch/run.log 2>&1`
5. Read out the results: parse the summary output
6. If the output parsing fails, the run crashed. Run `tail -n 50 autoresearch/run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If risk_adjusted_score improved (higher), you "advance" the branch, keeping the git commit
9. If risk_adjusted_score is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take reasonable time. If a run exceeds 30 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (error, or a bug), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read academic papers on trading strategies, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.