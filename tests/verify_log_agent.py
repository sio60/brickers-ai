import sys
import os
from pathlib import Path
import json
import logging

# Add 'brick-engine' to path so we can import 'agent'
project_root = Path(__file__).resolve().parent.parent
brick_engine_path = project_root / "brick-engine"
sys.path.append(str(brick_engine_path))

try:
    from agent.log_agent import app
except ImportError as e:
    print(f"Failed to import agent: {e}")
    # Fallback: Try project root if structure is different
    sys.path.append(str(project_root))
    try:
        from brick_engine.agent.log_agent import app
    except ImportError:
        print(f"Fatal: Could not import agent. Check sys.path: {sys.path}")
        sys.exit(1)

def main():
    # Configure logging to show in terminal
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*60)
    print("🚀 Starting Log Analysis Agent Verification (Looped Graph)")
    print("="*60)
    
    # Selection of test scenario
    simulated_log_db = """
    [2024-05-20 10:00:05] ERROR: Connection to MongoDB failed.
    pymongo.errors.ServerSelectionTimeoutError: localhost:27017: [Errno 111] Connection refused
    """
    
    simulated_log_sqs = """
    [2024-05-20 10:05:00] ERROR: Boto3 Error while polling SQS.
    botocore.exceptions.ClientError: An error occurred (AccessDenied) when calling the GetQueueAttributes operation.
    """
    
    # Select scenario
    current_log = simulated_log_db 
    print(f"📝 Testing Scenario: {'DB Connection' if current_log == simulated_log_db else 'SQS Error'}")
    
    initial_state = {
        "container_name": "brickers-ai-container", 
        "logs": current_log, 
        "analysis_result": None,
        "error_count": 0,
        "messages": [],
        "iteration": 0
    }

    # Docker Check
    print("\n🔍 Checking environment...")
    try:
        import docker
        client = docker.from_env()
        client.ping()
        print("✅ Docker is available and responding.")
    except Exception as e:
        print(f"⚠️ Docker connectivity issue: {e}")
        print("   (Test will proceed using simulated logs if organic fetch fails)")

    print(f"\n⚙️ Invoking LangGraph 'app.invoke()'...")
    print("-" * 40)
    
    try:
        # Run the graph
        output = app.invoke(initial_state)
        print("-" * 40)
        print("✅ Invocation complete.")
        
        # Output contains 'analysis_result' which is a JSON string
        raw_result_str = output.get("analysis_result")
        
        if raw_result_str:
            try:
                result_json = json.loads(raw_result_str)
                print("\n📄 FINAL ANALYSIS REPORT:")
                print(json.dumps(result_json, indent=2, ensure_ascii=False))
                
                # Extract Analysis
                analysis = result_json.get("analysis", result_json)
                
                if analysis.get("error_found"):
                    print(f"\n🔥 [ROOT CAUSE FOUND]: {analysis.get('root_cause')}")
                    print(f"💡 [SUGGESTION]: {analysis.get('suggestion')}")
                else:
                    print("\n✅ AI Summary: No critical errors found in logs.")

            except json.JSONDecodeError:
                print(f"\n⚠️ Raw Result (Not JSON): {raw_result_str}")
        else:
            print("\n❌ Fatal: No analysis result returned from agent.")
            
        print(f"\n🔄 Total Transitions (Iterations): {output.get('iteration', 0)}")
        print("="*60 + "\n")
            
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
