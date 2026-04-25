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

# ==========================================
# 📂 STRICT PATHING ARCHITECTURE
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if "worker_node" in CURRENT_DIR:
    ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
else:
    ROOT_DIR = CURRENT_DIR

# LOCAL TO WORKER: Each clone gets its own config file to test
STRATEGY_FILE = "ai_config.py" 
BEST_CONFIG_FILE = "best_ai_config.py" 

# GLOBAL SHARED: All workers write to the root database and results folder
RESULTS_FILE = os.path.join(ROOT_DIR, "results.tsv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

TRAIN_CMD = [sys.executable, "train.py"]

def get_code_hash(code_string):
    return hashlib.md5(code_string.encode('utf-8')).hexdigest()[:7]

def get_history_and_best():
    tested_hashes = set()
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("trial_id\tfinal_result\tstatus\tdescription\n")
        return -999.0, tested_hashes
    
    best_score = -999.0
    with open(RESULTS_FILE, "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                tested_hashes.add(parts[0]) 
                try:
                    score = float(parts[1])
                    status = parts[2]
                    if status == "keep" and score > best_score:
                        best_score = score
                except ValueError:
                    pass
    return best_score, tested_hashes

def log_result(trial_id, score, status, desc):
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{trial_id}\t{score}\t{status}\t{desc}\n")
        f.flush()
        os.fsync(f.fileno())

def get_memory_context(memory_bank):
    try:
        results = memory_bank.collection.get(include=["metadatas", "documents"])
        if not results or not results['metadatas'] or len(results['metadatas']) == 0:
            return "No historical data yet. You are the first generation."
            
        trials = []
        for i in range(len(results['metadatas'])):
            meta = results['metadatas'][i]
            doc = results['documents'][i] if results['documents'] else ""
            score = float(meta.get('score', -999.0))
            status = meta.get('status', 'discard')
            trials.append({'score': score, 'status': status, 'doc': doc})
            
        winners = sorted([t for t in trials if t['status'] == 'keep'], key=lambda x: x['score'], reverse=True)
        losers = sorted([t for t in trials if t['status'] in ['discard', 'crash']], key=lambda x: x['score'])
        
        context = "--- HISTORICAL MEMORY BANK ---\n"
        if winners:
            context += "🏆 TOP PERFORMING STRATEGIES (Analyze these for patterns):\n"
            for w in winners[:3]:
                context += f"- Winning Score {w['score']}: {w['doc']}\n"
            context += "\n"
        if losers:
            context += "☠️ PAST LANDMINES (DO NOT REPEAT THESE):\n"
            for l in losers[:3]:
                context += f"- Failed Score {l['score']}: {l['doc']}\n"
        return context
    except Exception as e:
        return "Could not fetch memory."

def generate_hypothesis(best_score, memory_context):
    print("🤔 Lead Quant is analyzing the historical data and current state...")
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
                        "Your job is to tune the XGBoost hyperparameters, select from the available features, "
                        "and tune the Risk Management variables to maximize the Out-Of-Sample score.\n"
                        "CRITICAL: You must constrain model complexity to prevent overfitting."
                    )
                },
                {
                    "role": "user", 
                    "content": (
                        f"Our current best Out-Of-Sample score is {best_score}.\n"
                        "WARNING: If the score is -999.0, your model is acting cowardly. Lower the threshold to force trades.\n\n"
                        f"{memory_context}\n\n"
                        "ANTI-OVERFITTING RULES (STRICT):\n"
                        "1. TARGET_LOOKAHEAD MUST be strictly 2.\n"
                        "2. 'max_depth' MUST be a raw integer, exactly 3 or 4. DO NOT wrap it in brackets, LaTeX, or escape characters (e.g., no \\(3\\) or |3|).\n"
                        "3. 'n_estimators' MUST be strictly between 100 and 200.\n"
                        "4. 'reg_alpha' (L1) and 'reg_lambda' (L2) MUST be strictly between 1.5 and 2.2.\n"
                        "5. THRESHOLD_PERCENTILE MUST be between 95 and 99.\n\n"
                        "FEATURE EXPANSION RULE: You MUST retain the core profitable trinity: ['cvd_trend', 'rsi_14', 'macd_line']. However, to break our high score, you are now AUTHORIZED to add 1 or 2 experimental features to this list. You MUST ONLY choose from this exact list of available features: ['atr_14', 'close_zscore_50', 'volume_zscore_24', 'bb_upper', 'bb_lower', 'log_return']. DO NOT invent feature names or the system will crash.\n\n"
                        "You MUST format your response EXACTLY like this (valid multi-line Python code):\n"
                        "THINKING: [Explain your logic]\n"
                        "HYPOTHESIS:\n"
                        "FEATURES=['cvd_trend', 'rsi_14', 'macd_line', 'experimental_feature_here']\n"
                        "TARGET_LOOKAHEAD=2\n"
                        "THRESHOLD_PERCENTILE=97\n"
                        "MODEL_PARAMS={'max_depth': 3, 'learning_rate': 0.05, 'n_estimators': 125, 'reg_alpha': 1.6, 'reg_lambda': 1.6}\n"
                        "SL_ATR_MULTIPLIER=1.5\n"
                        "TP_ATR_MULTIPLIER=3.0\n\n"
                        "RULES FOR RISK MULTIPLIERS:\n"
                        "- SL_ATR_MULTIPLIER: Float between 1.0 and 4.0. Controls the Stop Loss.\n"
                        "- TP_ATR_MULTIPLIER: Float between 2.0 and 8.0. Controls the Take Profit.\n"
                        "- Higher TP means larger potential returns but lower win rates. Optimize these based on the market regime!"
                    )
                }
            ]
        )
        
        content = response.choices[0].message.content
        if not content:
            return "API error.", ""

        content = content.strip()
        content_clean = re.sub(r'<think>.*?(</think>|$)', '', content, flags=re.DOTALL).strip()
        
        thinking_match = re.search(r'THINKING\s*[:\-]?\s*(.*?)(?=(?:HYPOTHESIS|HYPATHESIS|HYPERTUNING|ACTION)\s*[:\-]?|$)', content_clean, re.IGNORECASE | re.DOTALL)
        hypothesis_match = re.search(r'(?:HYPOTHESIS|HYPATHESIS|HYPERTUNING|ACTION)\s*[:\-]?\s*(.*)', content_clean, re.IGNORECASE | re.DOTALL)
        
        thinking = thinking_match.group(1).strip() if thinking_match else "Failed to parse thinking."
        hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else ""
        
        return thinking.replace("**", "").replace("*", "").replace("`", ""), hypothesis.replace("**", "").replace("*", "").replace("`", "")
        
    except Exception as e:
        print(f"⚠️ Hypothesis generation failed: {e}")
        return "System error.", ""

def run_experiment(memory_bank):
    local_best, tested_hashes = get_history_and_best() 
    global_best = memory_bank.get_global_best_score()
    best_score = max(local_best, global_best)
    
    print(f"\n" + "="*50)
    print(f"🚀 STARTING NEW ITERATION | Target to beat: {best_score}")
    print("="*50)

    memory_context = get_memory_context(memory_bank)
    thinking, hypothesis = generate_hypothesis(best_score, memory_context)
    
    if not hypothesis or len(hypothesis) < 5:
        print("\n⚠️ Invalid hypothesis generated. Skipping iteration.")
        time.sleep(3)
        return
        
    print(f"\n🧠 AI Reasoning:\n   > {thinking}")
    print(f"💡 AI Hypothesis:\n   > {hypothesis}")
    
    print(f"\n⚡ Direct Code Injection: Extracting variables and writing to {STRATEGY_FILE}...")
    try:
        features_match = re.search(r"FEATURES\s*=\s*\[.*?\]", hypothesis, re.DOTALL)
        lookahead_match = re.search(r"TARGET_LOOKAHEAD\s*=\s*\d+", hypothesis)
        threshold_match = re.search(r"THRESHOLD_PERCENTILE\s*=\s*\d+", hypothesis)
        params_match = re.search(r"MODEL_PARAMS\s*=\s*\{.*?\}", hypothesis, re.DOTALL)
        
        # 🔥 FIX: Extract the new Risk Multipliers
        sl_match = re.search(r"SL_ATR_MULTIPLIER\s*=\s*[0-9.]+", hypothesis)
        tp_match = re.search(r"TP_ATR_MULTIPLIER\s*=\s*[0-9.]+", hypothesis)

        if not (features_match and lookahead_match and threshold_match and params_match):
            print("⚠️ AI provided incomplete Python variables. Skipping injection.")
            time.sleep(3)
            return

        # 🔥 FIX: Sanitize MODEL_PARAMS to remove rogue LaTeX/Markdown formatting
        safe_params = params_match.group(0).replace(r"\(", "").replace(r"\)", "").replace("|", "").replace("\\", "")

        # 🔥 FIX: Provide safe defaults if the AI hallucinated and forgot them
        sl_str = sl_match.group(0) if sl_match else "SL_ATR_MULTIPLIER=1.5"
        tp_str = tp_match.group(0) if tp_match else "TP_ATR_MULTIPLIER=3.0"

        clean_code = (
            f"{features_match.group(0)}\n"
            f"{lookahead_match.group(0)}\n"
            f"{threshold_match.group(0)}\n"
            f"{safe_params}\n"
            f"{sl_str}\n"
            f"{tp_str}\n"
        )
        
        trial_id = get_code_hash(clean_code)
        if trial_id in tested_hashes:
            print(f"⚠️ Duplicate Hash Detected ({trial_id}). We have already tested this exact combination! Skipping to force new ideas...")
            time.sleep(1)
            return
        
        with open(STRATEGY_FILE, "w", encoding="utf-8") as f:
            f.write(clean_code)
            
    except Exception as e:
        print(f"⚠️ Failed to parse or write: {e}")
        time.sleep(3)
        return

    print(f"\n📈 Running the Walk-Forward Judge...") 
    result = subprocess.run(TRAIN_CMD, capture_output=True, text=True, encoding="utf-8")
    full_output = result.stdout + "\n" + result.stderr
    
    match = re.search(r"FINAL_RESULT:([-\d.]+)", full_output)
    if match:
        score = float(match.group(1))
        
        if score > best_score:
            status = "keep_new_best"
        elif score >= (best_score - 5.0) and score >= 25.0: 
            status = "keep_runner_up"
        else:
            status = "discard"
            
        print(f"\n📊 OOS Result: {score}")
        
        if score != -999.0 and score != 0.0:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            filename = os.path.join(RESULTS_DIR, f"{score:.4f}_{trial_id}.txt")
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"=== AI HYPOTHESIS ===\n{hypothesis}\n\n")
                    f.write(f"=== DETAILED REPORT ===\n{full_output.strip()}\n")
            except Exception as e:
                print(f"⚠️ Could not save result file: {e}")

        if score == -999.0:
            print("\n--- 🚨 JUDGE SILENT VETO LOGS 🚨 ---")
            print(full_output.strip())
            print("------------------------------------\n")
    else:
        score = 0.0
        status = "crash"
        print(f"\n⚠️ Judge crashed or returned invalid output.")
        print("--- 🚨 CRASH LOGS 🚨 ---")
        print(full_output.strip())
        print("------------------------\n")


    if score == 0.0 or score == -999.0:
        print(f"\n🗑️ GUARDRAIL TRIGGERED: Score is {score}. Restoring {BEST_CONFIG_FILE}...")
        if os.path.exists(BEST_CONFIG_FILE):
            shutil.copy(BEST_CONFIG_FILE, STRATEGY_FILE)
        time.sleep(3)
        return

    if status == "keep_new_best":
        print(f"🏆 NEW GLOBAL BEST! (Score {score} > {best_score})")
        shutil.copy(STRATEGY_FILE, BEST_CONFIG_FILE) 
        log_result(trial_id, score, "keep", "Auto-experiment new high score")
        memory_bank.log_trial(trial_id, hypothesis, score, "keep") 
        
    elif status == "keep_runner_up":
        print(f"🥈 RUNNER UP: Highly Profitable Variant! (Score {score})")
        if os.path.exists(BEST_CONFIG_FILE):
            shutil.copy(BEST_CONFIG_FILE, STRATEGY_FILE)
        log_result(trial_id, score, "keep", "Highly profitable runner-up")
        memory_bank.log_trial(trial_id, hypothesis, score, "keep") 

    else:
        print(f"❌ FAILED (Score {score} is too low). Restoring base...")
        if os.path.exists(BEST_CONFIG_FILE):
            shutil.copy(BEST_CONFIG_FILE, STRATEGY_FILE)
        log_result(trial_id, score, "discard", "Failed attempt logged")
        memory_bank.log_trial(trial_id, hypothesis, score, "discard") 

    print("Waiting 3 seconds before the next loop...")
    time.sleep(3)

if __name__ == "__main__":
    from fetch_data import fetch_historical_data
    fetch_historical_data() 
    
    if not os.path.exists(BEST_CONFIG_FILE) and os.path.exists(STRATEGY_FILE):
        shutil.copy(STRATEGY_FILE, BEST_CONFIG_FILE)
    
    import chromadb
    client = chromadb.PersistentClient(path=os.path.join(ROOT_DIR, "shared_chroma_db"))
    
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