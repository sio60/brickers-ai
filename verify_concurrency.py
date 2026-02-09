import asyncio
import time
import concurrent.futures
from pathlib import Path
import sys
import os

# Add project root to path
# Assuming verify_concurrency.py is in brickers-ai/
sys.path.append(str(Path(__file__).parent))

try:
    # Importing route manually to avoid circular imports or config loading issues if simple script
    # But route.kids_render imports service.kids_config which loads Env.
    # We might need .env loaded.
    from dotenv import load_dotenv
    load_dotenv()
    
    from route.kids_render import _process_executor, _run_brickify_worker
except ImportError as e:
    print(f"Could not import kids_render: {e}")
    print("Make sure dependencies are installed and you are running this from brickers-ai/")
    sys.exit(1)

def verify_concurrency():
    print("Running concurrency configuration check...")
    
    executor = _process_executor
    if isinstance(executor, concurrent.futures.ProcessPoolExecutor):
        print("✅ ProcessPoolExecutor is initialized correctly.")
        # Accessing internal _max_workers depends on implementation, but usually safe for debugging
        max_workers = getattr(executor, '_max_workers', 'Unknown')
        print(f"   Max workers: {max_workers}")
        
        print("\nTo fully verify parallel execution:")
        print("1. Start the server: `python app.py`")
        print("2. Send multiple concurrent requests.")
        print("3. Observe if 'Brickify' logs appear simultaneously.")
    else:
        print("❌ Executor is not a ProcessPoolExecutor. Parallelism will fail.")

if __name__ == "__main__":
    verify_concurrency()
