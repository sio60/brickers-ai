import sys
import os
import asyncio
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config # Load .env

from service.background_composer import generate_background_async

async def main():
    print("Testing generate_background_async...")
    subject = "Lego Car"
    try:
        bg_bytes = await generate_background_async(subject)
        print(f"Success! Generated {len(bg_bytes)} bytes.")
        with open("test_bg.png", "wb") as f:
            f.write(bg_bytes)
        print("Saved to test_bg.png")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
