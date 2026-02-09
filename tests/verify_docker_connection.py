# tests/verify_docker_connection.py
import docker
import sys

def verify_connection():
    try:
        print("Checking Docker connection...")
        client = docker.from_env()
        print(f"Docker version: {client.version()['Version']}")
        
        print("\nListing running containers:")
        containers = client.containers.list()
        if not containers:
            print("No running containers found (but connection successful).")
        for c in containers:
            print(f" - {c.name} ({c.status})")
            
        print("\n\u2705 Docker connection verified!")
        return True
    except Exception as e:
        print(f"\n\u274c Failed to connect to Docker: {e}")
        print("Make sure /var/run/docker.sock is mounted and accessible.")
        return False

if __name__ == "__main__":
    success = verify_connection()
    sys.exit(0 if success else 1)
