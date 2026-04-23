import os
import subprocess
import re
import time
import sys
import traceback
import hashlib
import shutil
from litellm import completion

from rag_memory import StrategyMemoryBank

STRATEGY_FILE = "ai_config.py" 
BEST_CONFIG_FILE = "best_ai_config.py" # Replaces Git restores
TRAIN_CMD = [sys.executable, "train.py"]
RESULTS_FILE = "results.tsv"

def get_code_hash(code_string):
    """Generates a unique 7-character hash to replace Git Commits for the database."""
    return hashlib.md5(code_string.encode('utf-8')).hexdigest()[:7]

def get_history_and_best():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("trial_id\tfinal_result\tstatus\tdescription\n")
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

def log_result(trial_id, score, status, desc):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{trial_id}\t{score}\t{status}\t{desc}\n")
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
                        "Instead, you must tune the XGBoost hyperparameters, select from the 8 available features, "
                        "AND tune the TARGET_LOOKAHEAD (1 to 5) and THRESHOLD_PERCENTILE (70 to 95) variables to maximize the Out-Of-Sample score."
                    )
                },
                {
                    "role": "user", 
                    "content": (
                        f"Our current best Out-Of-Sample score is {best_score}.\n"
                        "CRITICAL WARNING: If the score is -999.0, your model is acting cowardly. You MUST force it to take risks.\n\n"
                        "CRITICAL RULE: You may ONLY use these exact feature names: ['cvd_trend', 'atr_14', 'close_zscore_50', 'volume_zscore_24', 'rsi_14', 'macd_line', 'bb_lower', 'bb_upper']. Do NOT invent features like 'bb_width' or 'volume_ma'.\n\n"
                        "You MUST format your response EXACTLY like this (valid multi-line Python code):\n"
                        "THINKING: [Explain your logic]\n"
                        "HYPOTHESIS:\n"
                        "FEATURES=['cvd_trend', 'rsi_14', 'macd_line']\n"
                        "TARGET_LOOKAHEAD=2\n"
                        "THRESHOLD_PERCENTILE=85\n"
                        "MODEL_PARAMS={'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100}"
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
        return thinking, hypothesis
        
    except Exception as e:
        print(f"⚠️ Hypothesis generation failed: {e}")
        return "System error.", ""

def run_experiment(memory_bank):
    local_best, recent_history = get_history_and_best()
    global_best = memory_bank.get_global_best_score()
    best_score = max(local_best, global_best)
    
    print(f"\n" + "="*50)
    print(f"🚀 STARTING NEW ITERATION | Target to beat: {best_score}")
    print("="*50)

    max_retries = 3
    hypothesis = ""
    
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
            print("Sending the Lead Quant back to the drawing board to save LLM tokens...")
            time.sleep(2)
            continue 
        else:
            hypothesis = temp_hypothesis
            if raw_memories:
                print("\n📚 RAG Memory Triggered! Filtering optimal context...")
                successes = [m for m in raw_memories if m['status'] == 'keep']
                best_success = sorted(successes, key=lambda x: x['score'], reverse=True)[:1]
                if best_success:
                    print(f"   [1] 🏆 WINNER (Score: {best_success[0]['score']}) -> {best_success[0]['summary'][:75]}...")
            else:
                print("\n📚 RAG Memory: No past attempts found. Entering uncharted territory.")
            break 

    if not hypothesis:
        print("\n⚠️ Lead Quant couldn't find a novel idea after 3 tries. Skipping iteration to avoid loop trap.")
        time.sleep(3)
        return

    print(f"\n⚡ Direct Code Injection: Extracting variables and writing to {STRATEGY_FILE}...")
    
    try:
        features_match = re.search(r"FEATURES\s*=\s*\[.*?\]", hypothesis, re.DOTALL)
        lookahead_match = re.search(r"TARGET_LOOKAHEAD\s*=\s*\d+", hypothesis)
        threshold_match = re.search(r"THRESHOLD_PERCENTILE\s*=\s*\d+", hypothesis)
        params_match = re.search(r"MODEL_PARAMS\s*=\s*\{.*?\}", hypothesis, re.DOTALL)

        if not (features_match and lookahead_match and threshold_match and params_match):
            print("⚠️ AI provided incomplete Python variables. Skipping injection.")
            time.sleep(3)
            return

        clean_code = (
            f"{features_match.group(0)}\n"
            f"{lookahead_match.group(0)}\n"
            f"{threshold_match.group(0)}\n"
            f"{params_match.group(0)}\n"
        )
        
        # Write to the execution file
        with open(STRATEGY_FILE, "w", encoding="utf-8") as f:
            f.write(clean_code)
            
        trial_id = get_code_hash(clean_code)
            
    except Exception as e:
        print(f"⚠️ Failed to parse or write to {STRATEGY_FILE}: {e}")
        time.sleep(3)
        return

    print(f"\n📈 Running the Walk-Forward Judge...") 
    result = subprocess.run(TRAIN_CMD, capture_output=True, text=True, encoding="utf-8")
    full_output = result.stdout + "\n" + result.stderr
    
    match = re.search(r"FINAL_RESULT:([-\d.]+)", full_output)
    
    if match:
        score = float(match.group(1))
        status = "keep" if score > best_score else "discard"
        print(f"\n📊 OOS Result: {score}")
        
        if score == -999.0:
            print("\n--- 🚨 JUDGE SILENT VETO LOGS 🚨 ---")
            print(full_output.strip())
            print("------------------------------------\n")
    else:
        score = 0.0
        status = "crash"
        print(f"\n⚠️ Judge crashed or returned invalid output.")

    # ===============================================
    # THE REVERT LOGIC (No Git Required)
    # ===============================================
    if score == 0.0 or score == -999.0:
        print(f"\n🗑️ GUARDRAIL TRIGGERED: Score is {score}. Restoring {BEST_CONFIG_FILE} to avoid database poisoning...")
        if os.path.exists(BEST_CONFIG_FILE):
            shutil.copy(BEST_CONFIG_FILE, STRATEGY_FILE)
        time.sleep(3)
        return

    if status == "keep":
        print(f"✅ SUCCESS! New high score.")
        # Backup this winning config to be our new baseline!
        shutil.copy(STRATEGY_FILE, BEST_CONFIG_FILE)
        log_result(trial_id, score, status, "Auto-experiment success")
        memory_bank.log_trial(trial_id, hypothesis, score, status) 
    else:
        print(f"❌ FAILED (Score {score} <= {best_score}). Preserving history and restoring base...")
        # Revert back to the last known winner
        if os.path.exists(BEST_CONFIG_FILE):
            shutil.copy(BEST_CONFIG_FILE, STRATEGY_FILE)
        log_result(trial_id, score, status, "Failed attempt logged")
        memory_bank.log_trial(trial_id, hypothesis, score, status) 

    print("Waiting 3 seconds before the next loop...")
    time.sleep(3)

if __name__ == "__main__":
    from fetch_data import fetch_historical_data
    fetch_historical_data() 
    
    # Initialize the backup file on the very first run
    if not os.path.exists(BEST_CONFIG_FILE) and os.path.exists(STRATEGY_FILE):
        shutil.copy(STRATEGY_FILE, BEST_CONFIG_FILE)
    
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