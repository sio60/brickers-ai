
import os
import sys
from pathlib import Path

# Add current directory to sys.path to allow importing service modules
sys.path.append(str(Path(__file__).parent))

def load_env():
    env_path = Path(".env")
    if not env_path.exists():
        print("No .env file found")
        return
    
    print(f"Loading .env from {env_path.absolute()}")
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                # Helper to remove quotes if present
                value = value.strip()
                if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                os.environ[key.strip()] = value

def main():
    load_env()
    
    # Check if key is loaded
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Error: GEMINI_API_KEY not found in environment")
        return
    else:
        print(f"GEMINI_API_KEY loaded: {key[:5]}...{key[-5:]}")

    model = os.environ.get("NANO_BANANA_MODEL")
    print(f"NANO_BANANA_MODEL: {model}")

    print("Importing background_composer...")
    try:
        from service.background_composer import _generate_background_sync
        
        print("Starting generation test...")
        subject = "lego dinosaur"
        try:
            image_bytes = _generate_background_sync(subject)
            print(f"Success! Generated {len(image_bytes)} bytes.")
            
            # Save to verify
            with open("test_output.png", "wb") as f:
                f.write(image_bytes)
            print("Saved to test_output.png")
            
        except Exception as e:
            print(f"Generation failed with error: {e}")
            import traceback
            traceback.print_exc()

    except ImportError as e:
        print(f"Import failed: {e}")
        print("sys.path:", sys.path)

if __name__ == "__main__":
    main()
