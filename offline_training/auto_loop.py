import os
import subprocess
import re
import time
import sys

# Configuration
AIDER_CMD = "aider"
STRATEGY_FILE = "strategy.py"
TRAIN_CMD = [sys.executable, "train.py"]
RESULTS_FILE = "results.tsv"

def get_history_and_best():
    """Reads the TSV to find the ALL-TIME best score and the last 5 runs for AI context."""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("commit\tfinal_result\tstatus\tdescription\n")
        return -999.0, []
    
    best_score = -999.0
    history = []
    with open(RESULTS_FILE, "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                try:
                    score = float(parts[1])
                    status = parts[2]
                    # Find the highest "keep" score ever achieved
                    if status == "keep" and score > best_score:
                        best_score = score
                    # Format history to feed to the AI
                    history.append(f"Score: {score} | Status: {status}")
                except ValueError:
                    pass
    
    return best_score, history[-5:] # Return only the last 5 runs to save tokens

def get_current_commit():
    return subprocess.getoutput("git rev-parse --short HEAD").strip()

def log_result(commit, score, status, desc):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{score}\t{status}\t{desc}\n")
        f.flush()
        os.fsync(f.fileno())

def run_experiment():
    best_score, recent_history = get_history_and_best()
    print(f"\n" + "="*50)
    print(f"🚀 STARTING NEW ITERATION | Target to beat: {best_score}")
    print("="*50)

    commit_before = get_current_commit()
    history_text = "\n".join(recent_history)

    # The AI now gets to see its past mistakes!
    prompt = (
        f"The ALL-TIME BEST FINAL_RESULT is {best_score}.\n"
        f"Here are the results of your last few attempts (learn from them):\n{history_text}\n\n"
        f"You MUST modify the code in {STRATEGY_FILE} right now to try and beat {best_score}. "
        "Do not repeat failed ideas. Analyze what worked and what didn't. "
        "Output the actual code edits using the correct SEARCH/REPLACE format. "
        "Concrete ideas: Change the thresholds for the Volume Delta (cvd_20), adjust the Z-score (z_score_50) extremes, "
        "modify the cooldown period, or tweak the ATR adaptive math. DO NOT introduce RSI or MACD. "
        "Focus on the microstructure (volume vs price deviations). "
        "Do not apologize, do not explain. Just write the code edits."
    )
    
    print("🤖 Aider is thinking and coding...")
    
    aider_process = subprocess.run([
        AIDER_CMD,
        "--message", prompt,
        "--yes", 
        STRATEGY_FILE
    ], capture_output=True, text=True)

    if aider_process.returncode != 0:
        print(f"⚠️ Aider encountered an API error. Waiting 30 seconds before retrying...")
        time.sleep(30)
        return

    commit_after = get_current_commit()

    if commit_before == commit_after:
        print("⚠️ Aider did not make any changes. Skipping test.")
        log_result(commit_before, 0.0, "no_change", "Aider failed to produce new code")
        time.sleep(3)
        return

    print("\n📈 Running the backtest...")
    result = subprocess.run(TRAIN_CMD, capture_output=True, text=True)
    full_output = result.stdout + "\n" + result.stderr
    
    match = re.search(r"FINAL_RESULT:([-\d.]+)", full_output)
    
    if match:
        score = float(match.group(1))
        status = "keep" if score > best_score else "discard"
        print(f"\n📊 Result: {score}")
    else:
        score = 0.0
        status = "crash"
        print(f"\n⚠️ Script crashed or returned invalid output. Log:")
        print(full_output[-1000:]) 

    # Forward-Moving Git History Logic
    if status == "keep":
        print(f"✅ SUCCESS! New high score. Keeping changes.")
        log_result(commit_after, score, status, "Auto-experiment success")
    else:
        print(f"❌ FAILED (Score {score} <= {best_score}). Preserving history and restoring base...")
        
        # 1. Log the failed attempt so we have a record of what the AI tried
        log_result(commit_after, score, status, "Failed attempt logged")
        
        # 2. Safely restore strategy.py to the winning state WITHOUT deleting the failed commit
        subprocess.run(["git", "restore", "--source", commit_before, "--staged", "--worktree", STRATEGY_FILE], capture_output=True)
        
        # 3. Create a new commit to lock in the restoration
        subprocess.run(["git", "commit", "-m", f"Auto-revert to best state {commit_before} (Failed Score: {score})"], capture_output=True)
        
        reverted_commit = get_current_commit()
        log_result(reverted_commit, 0.0, "revert", f"Restored {commit_before}")

    print("Waiting 3 seconds before the next loop...")
    time.sleep(3)

if __name__ == "__main__":
    print("🔥 STARTING AUTORESEARCH INFINITE LOOP 🔥")
    print("Press CTRL+C to stop at any time.")
    while True:
        try:
            run_experiment()
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user. Exiting gracefully.")
            break
        except Exception as e:
            print(f"⚠️ Unexpected Manager Error: {e}")
            time.sleep(10)