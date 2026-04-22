import os
import shutil
import subprocess
import time

# Configuration
NUM_WORKERS = 4
BASE_DIR = os.path.abspath("strategy_trainer")
WORKER_PREFIX = "worker_node_"

def setup_and_launch_swarm():
    print(f"🐝 INITIALIZING BARE-METAL SWARM ({NUM_WORKERS} WORKERS)...")
    
    processes = []
    
    for i in range(1, NUM_WORKERS + 1):
        worker_dir = os.path.abspath(f"{WORKER_PREFIX}{i}")
        
        # 1. Clean up old worker directories if they exist
        if os.path.exists(worker_dir):
            print(f"   [+] Wiping old {worker_dir}...")
            shutil.rmtree(worker_dir, ignore_errors=True)
            
        # 2. Clone the base directory (This isolates strategy.py, results.tsv, and .git)
        print(f"   [+] Cloning base logic to {worker_dir}...")
        shutil.copytree(BASE_DIR, worker_dir)
        
        # 3. Launch the background process
        print(f"   [+] Starting Worker {i}...")
        
        # We use creationflags to open them in separate command prompt windows 
        # so you can watch them all work side-by-side if you want.
        p = subprocess.Popen(
            ["python", "auto_loop.py"], 
            cwd=worker_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE 
        )
        processes.append(p)
        
        # Stagger the starts slightly so they don't all hit the LLM API at the exact same millisecond
        time.sleep(2) 

    print("\n✅ SWARM LAUNCHED SUCESSFULLY.")
    print("All workers are now autonomously coding and backtesting in parallel.")
    print("They are all writing to the shared ChromaDB brain.")
    print("Close this master window to let them run, or close their individual windows to stop them.")

if __name__ == "__main__":
    setup_and_launch_swarm()