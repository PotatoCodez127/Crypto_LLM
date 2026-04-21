import os
import subprocess
import re
import time
import sys
import traceback
from litellm import completion

from rag_memory import StrategyMemoryBank

AIDER_CMD = os.path.join(os.path.dirname(sys.executable), "aider.exe")
STRATEGY_FILE = "strategy.py"
TRAIN_CMD = [sys.executable, "train.py"]
RESULTS_FILE = "results.tsv"

def get_history_and_best():
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
    """The 'Lead Quant' generates reasoning and a hypothesis to search the DB against."""
    print("🤔 Lead Quant is analyzing the current state...")
    try:
        response = completion(
            model="openai/deepseek-v3.2",
            api_base="http://localhost:4000", 
            api_key="sk-dummy-key-1234", 
            temperature=0.7, # Added to prevent model freeze
            messages=[
                {
                    "role": "system", 
                    "content": "You are an elite quantitative researcher improving a Python trading strategy."
                },
                {
                    "role": "user", 
                    "content": (
                        f"Our current best Out-Of-Sample score is {best_score}. Formulate your next move.\n\n"
                        "You MUST format your response EXACTLY like this:\n"
                        "THINKING: [Explain your logic in 2 sentences]\n"
                        "HYPOTHESIS: [Write a 1-sentence strict coding instruction]"
                    )
                }
            ],
            max_tokens=400
        )
        
        # Safely grab the content
        content = response.choices[0].message.content
        
        # --- NEW DIAGNOSTIC CATCH ---
        if content is None or content.strip() == "":
            print("\n❌ API ERROR: The model returned an absolutely empty string.")
            print("--- FULL API RESPONSE OBJECT ---")
            print(response) # This will print finish_reasons and token usage!
            print("--------------------------------\n")
            return "API generated empty response.", ""

        content = content.strip()
        
        # 1. Strip out any internal <think> reasoning tokens
        import re
        content_clean = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # 2. Aggressive Regex hunting
        thinking_match = re.search(r'THINKING\s*[:\-]?\s*(.*?)(?=HYPOTHESIS\s*[:\-]?|$)', content_clean, re.IGNORECASE | re.DOTALL)
        hypothesis_match = re.search(r'HYPOTHESIS\s*[:\-]?\s*(.*)', content_clean, re.IGNORECASE | re.DOTALL)
        
        thinking = thinking_match.group(1).strip() if thinking_match else "Failed to parse thinking."
        hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else ""
        
        # 3. Clean up stray markdown artifacts
        thinking = thinking.replace("**", "").replace("*", "").replace("`", "")
        hypothesis = hypothesis.replace("**", "").replace("*", "").replace("`", "")
        
        # --- THE RAW ERROR LOGGER ---
        if not hypothesis or len(hypothesis) < 5:
            print("\n❌ PARSING ERROR: The AI completely ignored the formatting instructions.")
            print("--- RAW AI OUTPUT ---")
            print(content)
            print("---------------------\n")
            
            with open("llm_error_log.txt", "a", encoding="utf-8") as f:
                import time
                f.write(f"--- FAILED PARSE AT {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                f.write(content + "\n\n")
                
        return thinking, hypothesis
        
    except Exception as e:
        print(f"⚠️ Hypothesis generation failed: {e}")
        return "System error.", ""

def run_experiment(memory_bank):
    best_score, recent_history = get_history_and_best()
    print(f"\n" + "="*50)
    print(f"🚀 STARTING NEW ITERATION | Target to beat: {best_score}")
    print("="*50)

    commit_before = get_current_commit()

    # --- 1. Generate the Idea ---
    thinking, hypothesis = generate_hypothesis(best_score)
    
    # GUARDRAIL 1: Reject empty or garbage hypotheses
    if not hypothesis or len(hypothesis) < 5:
        print("\n⚠️ Lead Quant produced an empty or invalid hypothesis. Skipping to next iteration...")
        time.sleep(3)
        return
    
    print(f"\n🧠 AI Reasoning:\n   > {thinking}")
    print(f"\n💡 AI Hypothesis:\n   > {hypothesis}")

    # --- 2. Query the Memory Bank ---
    memories = memory_bank.query_similar_trials(hypothesis, n_results=3)
    memory_text = ""
    
    if memories:
        print("\n📚 RAG Memory Triggered! Recalling past trials:")
        memory_text = "--- HISTORICAL MEMORY BANK (SIMILAR PAST TRIALS) ---\n"
        for i, m in enumerate(memories):
            summary_preview = m['summary'][:75] + "..." if len(m['summary']) > 75 else m['summary']
            print(f"   [{i+1}] {m['status'].upper()} (Score: {m['score']}) -> {summary_preview}")
            memory_text += f"PAST IDEA: '{m['summary']}' | RESULT SCORE: {m['score']} | STATUS: {m['status']}\n"
            
        memory_text += "\nCRITICAL: Analyze why the 'discard' ideas above failed. DO NOT repeat the exact same parameters or logic. Pivot your approach.\n"
    else:
        print("\n📚 RAG Memory: No highly similar past attempts found. Entering uncharted territory.")

    # --- 3. Build the Prompt & Code ---
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
    ], capture_output=True, text=True, encoding="utf-8")

    if aider_process.returncode != 0:
        print(f"⚠️ Aider encountered an error. Waiting 10 seconds...")
        time.sleep(10)
        return

    git_status = subprocess.getoutput(f"git status --porcelain {STRATEGY_FILE}").strip()

    if not git_status:
        print("⚠️ Aider did not make any changes. Skipping test.")
        time.sleep(3)
        return

    subprocess.run(["git", "add", STRATEGY_FILE], capture_output=True)
    subprocess.run(["git", "commit", "-m", f"Auto-experiment: {hypothesis[:50]}..."], capture_output=True)
    
    commit_after = get_current_commit()

    # --- 4. The WFO Judge ---
    print(f"\n📈 Running the Walk-Forward Judge...") 
    result = subprocess.run(TRAIN_CMD, capture_output=True, text=True, encoding="utf-8")
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

    # GUARDRAIL 2: Reject 0.0 Scores
    if score == 0.0:
        print(f"\n🗑️ GUARDRAIL TRIGGERED: Score is exactly 0.0. Restoring code and abandoning memory log...")
        subprocess.run(["git", "restore", "--source", commit_before, "--staged", "--worktree", STRATEGY_FILE], capture_output=True)
        subprocess.run(["git", "commit", "-m", f"Auto-revert (0.0 score) to {commit_before}"], capture_output=True)
        time.sleep(3)
        return

    # --- 5. Save Valid Results ---
    if status == "keep":
        print(f"✅ SUCCESS! New high score.")
        log_result(commit_after, score, status, "Auto-experiment success")
        memory_bank.log_trial(commit_after, hypothesis, score, status) 
    else:
        print(f"❌ FAILED (Score {score} <= {best_score}). Preserving history and restoring base...")
        log_result(commit_after, score, status, "Failed attempt logged")
        memory_bank.log_trial(commit_after, hypothesis, score, status) 
        
        subprocess.run(["git", "restore", "--source", commit_before, "--staged", "--worktree", STRATEGY_FILE], capture_output=True)
        subprocess.run(["git", "commit", "-m", f"Auto-revert to {commit_before}"], capture_output=True)

    print("Waiting 3 seconds before the next loop...")
    time.sleep(3)

if __name__ == "__main__":
    from fetch_data import fetch_historical_data
    fetch_historical_data() # Will silently skip if the file already exists

    print("🔥 STARTING RAG-AUGMENTED AUTORESEARCH LOOP 🔥")
    print("Press CTRL+C to stop at any time.")
    
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