import os
import shutil
import subprocess
import time
import sys

# Configuration
NUM_WORKERS = 2
BASE_DIR = os.path.abspath("strategy_trainer")
WORKER_PREFIX = "worker_node_"

def setup_and_launch_swarm():
    print(f"🐝 INITIALIZING BARE-METAL SWARM ({NUM_WORKERS} WORKERS)...")
    
    # This automatically grabs the Python from your active (.venv)
    python_exe = sys.executable 
    
    processes = []
    
    for i in range(1, NUM_WORKERS + 1):
        worker_dir = os.path.abspath(f"{WORKER_PREFIX}{i}")
        
        if os.path.exists(worker_dir):
            print(f"   [+] Wiping old {worker_dir}...")
            shutil.rmtree(worker_dir, ignore_errors=True)
            
        print(f"   [+] Cloning base logic to {worker_dir}...")
        # Ignore pycache and local venvs to speed up the cloning process
        shutil.copytree(BASE_DIR, worker_dir, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
        
        print(f"   [+] Starting Worker {i}...")
        
        # 'cmd.exe /k' forces the terminal to stay open after the script finishes or crashes
        p = subprocess.Popen(
            ["cmd.exe", "/k", python_exe, "auto_loop.py"], 
            cwd=worker_dir,
            creationflags=subprocess.CREATE_NEW_CONSOLE 
        )
        processes.append(p)
        time.sleep(2) 

    print("\n✅ SWARM LAUNCHED SUCESSFULLY.")

if __name__ == "__main__":
    setup_and_launch_swarm()