import os
import subprocess
import re
import time
import sys
import traceback
from litellm import completion

from rag_memory import StrategyMemoryBank

AIDER_CMD = os.path.join(os.path.dirname(sys.executable), "aider.exe")
# POINT AIDER TO THE CONTROL PANEL
STRATEGY_FILE = "ai_config.py" 
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
    print("🤔 Lead Quant is analyzing the current state...")
    try:
        response = completion(
            model="openai/deepseek-v3.2",
            api_base="http://localhost:4000", 
            api_key="sk-dummy-key-1234", 
            temperature=0.8, 
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are an elite AI Data Scientist tuning an XGBoost trading strategy.\n"
                        "Your job is NO LONGER to write if/else heuristic rules.\n"
                        "Instead, you must tune the XGBoost hyperparameters (max_depth, learning_rate, n_estimators) "
                        "and select/combine the V2 features ('cvd_trend', 'atr_14', 'close_zscore_50', 'volume_zscore_24') to maximize the Out-Of-Sample score."
                    )
                },
                {
                    "role": "user", 
                    "content": (
                        f"Our current best Out-Of-Sample score is {best_score}.\n"
                        "CRITICAL WARNING: If the score is -999.0, your model is acting cowardly. It is predicting '0' (Hold) for every candle and taking 0 trades. "
                        "You MUST force the model to take risks. Try drastically increasing max_depth, changing learning_rate, or using a subset of features.\n\n"
                        "You MUST format your response EXACTLY like this:\n"
                        "THINKING: [Explain your logic in 2 sentences]\n"
                        "HYPOTHESIS: [Write a 1-sentence strict coding instruction]"
                    )
                }
            ]
        )
        
        content = response.choices[0].message.content
        
        if content is None or content.strip() == "":
            print("\n❌ API ERROR: The model returned an empty string.")
            return "API error.", ""

        content = content.strip()
        
        content_clean = re.sub(r'<think>.*?(</think>|$)', '', content, flags=re.DOTALL).strip()
        
        thinking_match = re.search(r'THINKING\s*[:\-]?\s*(.*?)(?=(?:HYPOTHESIS|HYPATHESIS|HYPERTUNING|ACTION)\s*[:\-]?|$)', content_clean, re.IGNORECASE | re.DOTALL)
        hypothesis_match = re.search(r'(?:HYPOTHESIS|HYPATHESIS|HYPERTUNING|ACTION)\s*[:\-]?\s*(.*)', content_clean, re.IGNORECASE | re.DOTALL)
        
        thinking = thinking_match.group(1).strip() if thinking_match else "Failed to parse thinking."
        hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else ""
        
        thinking = thinking.replace("**", "").replace("*", "").replace("`", "")
        hypothesis = hypothesis.replace("**", "").replace("*", "").replace("`", "")
        
        if not hypothesis or len(hypothesis) < 5:
            print("\n❌ PARSING ERROR: The AI completely ignored formatting.")
            with open("llm_error_log.txt", "a", encoding="utf-8") as f:
                f.write(f"--- FAILED PARSE AT {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n{content}\n\n")
                
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

    max_retries = 3
    hypothesis = ""
    memory_text = ""
    
    for attempt in range(max_retries):
        thinking, temp_hypothesis = generate_hypothesis(best_score)
        
        if not temp_hypothesis or len(temp_hypothesis) < 5:
            print("\n⚠️ Invalid hypothesis generated. Retrying...")
            time.sleep(2)
            continue
            
        print(f"\n🧠 AI Reasoning (Attempt {attempt + 1}):\n   > {thinking}")
        print(f"💡 AI Hypothesis:\n   > {temp_hypothesis}")
        
        raw_memories = memory_bank.query_similar_trials(temp_hypothesis, n_results=10)
        
        top_5 = raw_memories[:5]
        discard_count = sum(1 for m in top_5 if m['status'] in ['discard', 'crash'])
        
        if discard_count >= 2:
            print(f"🛑 REJECTION LOOP: ChromaDB found {discard_count} past failures for this exact concept!")
            print("Sending the Lead Quant back to the drawing board to save Aider tokens...")
            time.sleep(2)
            continue 
        else:
            hypothesis = temp_hypothesis
            if raw_memories:
                print("\n📚 RAG Memory Triggered! Filtering optimal context...")
                successes = [m for m in raw_memories if m['status'] == 'keep']
                failures = [m for m in raw_memories if m['status'] in ['discard', 'crash']]
                
                best_success = sorted(successes, key=lambda x: x['score'], reverse=True)[:1]
                worst_failures = failures[:2]
                final_memories = best_success + worst_failures
                
                if final_memories:
                    memory_text = "--- HISTORICAL MEMORY BANK (CRITICAL CONTEXT) ---\n"
                    for i, m in enumerate(final_memories):
                        tag = "🏆 WINNER" if m['status'] == 'keep' else "☠️ LANDMINE"
                        summary_preview = m['summary'][:75] + "..." if len(m['summary']) > 75 else m['summary']
                        print(f"   [{i+1}] {tag} (Score: {m['score']}) -> {summary_preview}")
                        memory_text += f"PAST IDEA: '{m['summary']}' | RESULT SCORE: {m['score']} | STATUS: {m['status']}\n"
                    
                    memory_text += "\nCRITICAL: Analyze the ☠️ LANDMINE ideas above and DO NOT repeat them. Analyze the 🏆 WINNER ideas for inspiration to pivot your approach.\n"
                else:
                    print("   > No highly relevant extremes found after filtering.")
            else:
                print("\n📚 RAG Memory: No past attempts found. Entering uncharted territory.")
            break 

    if not hypothesis:
        print("\n⚠️ Lead Quant couldn't find a novel idea after 3 tries. Skipping iteration to avoid loop trap.")
        time.sleep(3)
        return

    # NEW AIDER PROMPT TARGETING ONLY CONFIG
    prompt = (
        f"The ALL-TIME BEST FINAL_RESULT is {best_score}.\n"
        f"Your specific mission for this iteration is: {hypothesis}\n\n"
        f"{memory_text}\n"
        f"CRITICAL RULES:\n"
        f"Modify the variables in {STRATEGY_FILE} right now to execute this mission and try to beat {best_score}. "
        f"ONLY modify the FEATURES list and the MODEL_PARAMS dictionary. "
        f"Available features are: ['cvd_trend', 'atr_14', 'close_zscore_50', 'volume_zscore_24'].\n"
        "Output the actual code edits using the correct SEARCH/REPLACE format."
    )
    
    print(f"\n🤖 Aider is coding the hypothesis...") 
    
    aider_process = subprocess.run([
        AIDER_CMD,
        "--message", prompt,
        "--no-auto-commits",
        "--yes", 
        STRATEGY_FILE
    ], capture_output=True, text=True, encoding="utf-8")

    git_status = subprocess.getoutput(f"git status --porcelain {STRATEGY_FILE}").strip()

    if not git_status:
        print("⚠️ Aider did not make any changes. Skipping test.")
        time.sleep(3)
        return

    subprocess.run(["git", "add", STRATEGY_FILE], capture_output=True)
    subprocess.run(["git", "commit", "-m", f"Auto-experiment: {hypothesis[:50]}..."], capture_output=True)
    
    commit_after = get_current_commit()

    print(f"\n📈 Running the Walk-Forward Judge...") 
    result = subprocess.run(TRAIN_CMD, capture_output=True, text=True, encoding="utf-8")
    full_output = result.stdout + "\n" + result.stderr
    
    match = re.search(r"FINAL_RESULT:([-\d.]+)", full_output)
    
    if match:
        score = float(match.group(1))
        status = "keep" if score > best_score else "discard"
        print(f"\n📊 OOS Result: {score}")
        
        # EXPOSE THE SILENT LOGS
        if score == -999.0:
            print("\n--- 🚨 JUDGE SILENT VETO LOGS 🚨 ---")
            print(full_output.strip())
            print("------------------------------------\n")
            
    else:
        score = 0.0
        status = "crash"
        print(f"\n⚠️ Judge crashed or returned invalid output. Error logs below:")
        print("--- 🚨 CRASH LOGS 🚨 ---")
        print(full_output.strip())
        print("------------------------\n")

    if score == 0.0:
        subprocess.run(["git", "restore", "--source", commit_before, "--staged", "--worktree", STRATEGY_FILE], capture_output=True)
        time.sleep(3)
        return

    if status == "keep":
        print(f"✅ SUCCESS! New high score.")
        log_result(commit_after, score, status, "Auto-experiment success")
        memory_bank.log_trial(commit_after, hypothesis, score, status) 
    else:
        print(f"❌ FAILED (Score {score} <= {best_score}). Preserving history and restoring base...")
        log_result(commit_after, score, status, "Failed attempt logged")
        memory_bank.log_trial(commit_after, hypothesis, score, status) 
        subprocess.run(["git", "restore", "--source", commit_before, "--staged", "--worktree", STRATEGY_FILE], capture_output=True)

    print("Waiting 3 seconds before the next loop...")
    time.sleep(3)

if __name__ == "__main__":
    from fetch_data import fetch_historical_data
    fetch_historical_data() 
    
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