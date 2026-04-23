import os
from dotenv import load_dotenv

# 1. Load the keys from .env
load_dotenv()

# 2. Disable SSL Verification globally for the proxy
# This fixes the SSLCertVerificationError when using the direct IP (34.36.133.15)
os.environ["LITELLM_VERIFY_SSL"] = "False"

# 3. Verify and Start
# if not os.environ.get("KEY_1"):
#     print("[!] ERROR: KEY_1 not found in .env!")
# else:
#     print("[OK] Keys loaded. SSL Verification disabled. Initializing Proxy...")
#     os.system("litellm --config litellm_config.yaml")



# 3. Verify and Start (Check all keys)
# Advanced way: Automatically find ALL keys starting with "KEY_"
available_keys = [k for k in os.environ if k.startswith("KEY_")]

if not available_keys:
    print("[!] ERROR: No variables starting with 'KEY_' found in .env!")
else:
    print(f"[OK] Found {len(available_keys)} keys: {', '.join(available_keys)}. Initializing Proxy...")
    os.system("litellm --config litellm_config.yaml")