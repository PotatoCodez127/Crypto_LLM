import os
import subprocess
import re
import time
import sys
import traceback
from litellm import completion

# 1. Import our new RAG Memory Bank
from rag_memory import StrategyMemoryBank

# Configuration
AIDER_CMD = os.path.join(os.path.dirname(sys.executable), "aider.exe")
STRATEGY_FILE = "strategy.py"
TRAIN_CMD = [sys.executable, "train.py"]
RESULTS_FILE = "results.tsv"

def get_history_and_best():
    """Reads the TSV to find the ALL-TIME best score and the last 5 runs."""
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
                    if status == "keep" and score > best_score:
                        best_score = score
                    history.append(f"Score: {score} | Status: {status}")
                except ValueError:
                    pass
    
    return best_score, history[-5:] 

def get_current_commit():
    return subprocess.getoutput("git rev-parse --short HEAD").strip()

def log_result(commit, score, status, desc):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit}\t{score}\t{status}\t{desc}\n")
        f.flush()
        os.fsync(f.fileno())

def generate_hypothesis(best_score):
    """The 'Lead Quant' generates a 1-sentence idea to search the DB against."""
    print("🤔 Generating new research hypothesis...")
    try:
        # Route through your local LiteLLM proxy
        response = completion(
            model="openai/deepseek-v3.2",
            api_base="http://localhost:4000", 
            api_key="sk-dummy-key-1234",  # <-- ADD THIS DUMMY KEY
            messages=[
                {"role": "system", "content": "You are a quantitative researcher. Respond with EXACTLY ONE SENTENCE proposing a specific technical or statistical improvement to a Python trading strategy. Do not explain. Just the hypothesis."},
                {"role": "user", "content": f"Our current best Out-Of-Sample score is {best_score}. Give me ONE new hypothesis to try."}
            ],
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Hypothesis generation failed: {e}")
        return "Adjust moving average periods and tighten risk management."

def run_experiment(memory_bank):
    best_score, recent_history = get_history_and_best()
    print(f"\n" + "="*50)
    print(f"🚀 STARTING NEW ITERATION | Target to beat: {best_score}")
    print("="*50)

    commit_before = get_current_commit()

    # --- THE RAG PIPELINE ---
    # 1. Generate the Idea
    hypothesis = generate_hypothesis(best_score)
    print(f"\n💡 AI Hypothesis: {hypothesis}")

    # 2. Query the Memory Bank
    memories = memory_bank.query_similar_trials(hypothesis, n_results=3)
    memory_text = ""
    
    if memories:
        print("🔍 Found similar past attempts in Memory Bank. Injecting context...")
        memory_text = "--- HISTORICAL MEMORY BANK (SIMILAR PAST TRIALS) ---\n"
        for m in memories:
            memory_text += f"PAST IDEA: '{m['summary']}' | RESULT SCORE: {m['score']} | STATUS: {m['status']}\n"
        memory_text += "\nCRITICAL: Analyze why the 'discard' ideas above failed. DO NOT repeat the exact same parameters or logic. Pivot your approach.\n"
    else:
        print("🔍 No highly similar past attempts found. Entering uncharted territory.")

    # 3. Build the Context-Aware Prompt
    prompt = (
        f"The ALL-TIME BEST FINAL_RESULT is {best_score}.\n"
        f"Your specific mission for this iteration is: {hypothesis}\n\n"
        f"{memory_text}\n"
        f"Modify the code in {STRATEGY_FILE} right now to execute this mission and try to beat {best_score}. "
        "Output the actual code edits using the correct SEARCH/REPLACE format. "
        "Do not apologize, do not explain. Just write the code edits."
    )
    
    print(f"\n🤖 Aider is coding the hypothesis...") 
    
    aider_process = subprocess.run([
        AIDER_CMD,
        "--message", prompt,
        "--no-auto-commits",
        "--no-show-release-notes",
        "--no-check-update",
        "--no-show-model-warnings",
        "--yes", 
        STRATEGY_FILE
    ], capture_output=True, text=True, encoding="utf-8") # <-- ADD ENCODING HERE

    if aider_process.returncode != 0:
        print(f"⚠️ Aider encountered an error (Exit code {aider_process.returncode}). Waiting 30 seconds...")
        time.sleep(30)
        return

    # Check if Aider actually modified the file
    git_status = subprocess.getoutput(f"git status --porcelain {STRATEGY_FILE}").strip()

    if not git_status:
        print("⚠️ Aider did not make any changes. Skipping test.")
        time.sleep(3)
        return

    # Manually commit the changes instantly
    subprocess.run(["git", "add", STRATEGY_FILE], capture_output=True)
    subprocess.run(["git", "commit", "-m", f"Auto-experiment: {hypothesis[:50]}..."], capture_output=True)
    
    commit_after = get_current_commit()

    print(f"\n📈 Running the Walk-Forward Judge...") 
    result = subprocess.run(TRAIN_CMD, capture_output=True, text=True, encoding="utf-8") # <-- ADD ENCODING HERE
    full_output = result.stdout + "\n" + result.stderr
    
    match = re.search(r"FINAL_RESULT:([-\d.]+)", full_output)
    
    if match:
        score = float(match.group(1))
        status = "keep" if score > best_score else "discard"
        print(f"\n📊 OOS Result: {score}")
    else:
        score = 0.0
        status = "crash"
        print(f"\n⚠️ Judge crashed or returned invalid output.")

    # --- SAVE TO RAG MEMORY ---
    if status == "keep":
        print(f"✅ SUCCESS! New high score.")
        log_result(commit_after, score, status, "Auto-experiment success")
        memory_bank.log_trial(commit_after, hypothesis, score, status) # <-- Log to ChromaDB
    else:
        print(f"❌ FAILED (Score {score} <= {best_score}). Preserving history and restoring base...")
        log_result(commit_after, score, status, "Failed attempt logged")
        memory_bank.log_trial(commit_after, hypothesis, score, status) # <-- Log failure to ChromaDB so it learns!
        
        # Revert logic
        subprocess.run(["git", "restore", "--source", commit_before, "--staged", "--worktree", STRATEGY_FILE], capture_output=True)
        subprocess.run(["git", "commit", "-m", f"Auto-revert to {commit_before}"], capture_output=True)

    print("Waiting 3 seconds before the next loop...")
    time.sleep(3)

if __name__ == "__main__":
    print("🔥 STARTING RAG-AUGMENTED AUTORESEARCH LOOP 🔥")
    print("Press CTRL+C to stop at any time.")
    
    # Initialize DB once when the script starts
    db = StrategyMemoryBank() 
    
    while True:
        try:
            run_experiment(db)
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user. Exiting gracefully.")
            break
        except Exception as e:
            print(f"\n⚠️ Unexpected Error: {e}")
            traceback.print_exc()
            time.sleep(10)