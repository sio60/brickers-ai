import docker
import json
import re
import os
import sys
import platform
import subprocess

def test_fetch_logs(container_name="brickers-ai-container"):
    print(f"--- Environment Info ---")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    try:
        docker_version = subprocess.check_output(["docker", "--version"]).decode().strip()
        print(f"Docker CLI: {docker_version}")
    except:
        print("Docker CLI: Not found in PATH")
    print(f"------------------------\n")

    print(f"🔍 Testing log fetching for container: {container_name}")
    try:
        # Try default connection
        client = docker.from_env()
        # Test connection early
        client.version()
    except Exception as e:
        print(f"⚠️ Default connection failed: {e}")
        print("🔄 Attempting to connect via Windows Named Pipe...")
        try:
            client = docker.DockerClient(base_url='npipe:////./pipe/docker_engine')
            client.version()
        except Exception as e2:
            print(f"❌ Named Pipe connection failed: {e2}")
            print("🔄 Attempting to connect via localhost (if exposed)...")
            try:
                client = docker.DockerClient(base_url='tcp://localhost:2375')
                client.version()
            except Exception as e3:
                print(f"❌ All connection attempts failed.")
                raise e
    
    try:
        container = client.containers.get(container_name)
        raw_logs = container.logs(tail=100).decode("utf-8", errors="replace")
        
        print("✅ Successfully fetched logs from Docker!")
        print("-" * 40)
        print(raw_logs)
        print("-" * 40)
        
        # Test Job ID extraction logic
        failure_matches = re.findall(r"요청 실패! \| jobId=([a-f0-9-]+)", raw_logs)
        if failure_matches:
            print(f"🕵️ Found failed job: {failure_matches[-1]}")
        else:
            print("ℹ️ No failed jobs found in the last 100 lines.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("1. Docker is running.")
        print(f"2. A container named '{container_name}' is active.")
        print("3. You have the 'docker' python package installed (pip install docker).")

if __name__ == "__main__":
    test_fetch_logs()
