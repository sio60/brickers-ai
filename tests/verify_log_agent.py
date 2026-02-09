import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from brick_engine.agent.log_agent import app
except ImportError as e:
    print(f"Failed to import agent: {e}")
    sys.exit(1)

def main():
    print("🚀 Starting Log Analysis Agent Verification...")
    
    # Check if we are running in an environment with Docker
    try:
        import docker
        client = docker.from_env()
        client.ping()
        print("✅ Docker is available.")
    except Exception:
        print("⚠️ Docker not available or not running. This test might fail if it tries to connect to Docker.")
        # Optional: mock docker if needed, but better to fail if this is an integration test

    # Input state
    initial_state = {
        "container_name": "brickers-ai-container", # Change this if testing locally to a known container
        "logs": "",
        "analysis_result": None,
        "error_count": 0
    }

    print(f"📊 Invoking agent for container: {initial_state['container_name']}...")
    
    try:
        output = app.invoke(initial_state)
        
        result_json_str = output.get("analysis_result")
        if result_json_str:
            try:
                result = json.loads(result_json_str)
                print("\n✅ Analysis Result:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print("\n⚠️ Raw Result (Not JSON):")
                print(result_json_str)
        else:
            print("\n❌ No analysis result returned.")
            
        if output.get("error_count", 0) > 0:
            print(f"\n⚠️ Errors encountered: {output['error_count']}")

    except Exception as e:
        print(f"\n❌ Agent execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
