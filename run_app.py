import subprocess
import sys
import time
import webbrowser
import os
from pathlib import Path

# Add backend directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
os.environ["PYTHONPATH"] = str(project_root)


def check_qdrant():
    """Check if Qdrant is running, if not start it using Docker"""
    try:
        # Check if Qdrant container is running
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=qdrant", "--format", "{{.Names}}"], capture_output=True, text=True
        )

        if "qdrant" not in result.stdout:
            print("Starting Qdrant container...")
            subprocess.run(
                ["docker", "run", "-d", "--name", "qdrant", "-p", "6333:6333", "-p", "6334:6334", "qdrant/qdrant"]
            )
            # Wait for Qdrant to start
            time.sleep(5)
            print("Qdrant is ready!")
        else:
            print("Qdrant is already running")
    except Exception as e:
        print(f"Error starting Qdrant: {e}")
        sys.exit(1)


def start_backend():
    """Start the FastAPI backend server"""
    print("Starting backend server...")
    return subprocess.Popen(["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])


def start_frontends():
    """Start Streamlit frontends"""
    print("Starting Streamlit frontends...")
    user_matching = subprocess.Popen(["streamlit", "run", "frontend/pages/user_matching.py", "--server.port", "8501"])
    image_recommendation = subprocess.Popen(
        ["streamlit", "run", "frontend/pages/image_recommendation.py", "--server.port", "8502"]
    )
    product_search = subprocess.Popen(["streamlit", "run", "frontend/pages/product_search.py", "--server.port", "8503"])
    return user_matching, image_recommendation, product_search


def main():
    # Ensure we're in the project root directory
    project_root = Path(__file__).parent

    # Check/start Qdrant
    check_qdrant()

    # Start backend and frontends
    backend_process = start_backend()
    user_matching_process, image_recommendation_process, product_search_process = start_frontends()

    try:
        # Open the applications in the default browser
        time.sleep(3)  # Wait for servers to start
        webbrowser.open("http://localhost:8501")  # User Matching UI
        webbrowser.open("http://localhost:8502")  # Image Recommendation UI
        webbrowser.open("http://localhost:8503")  # Product Search UI

        print("\nApplications are running!")
        print("User Matching UI: http://localhost:8501")
        print("Image Recommendation UI: http://localhost:8502")
        print("Product Search UI: http://localhost:8503")
        print("Backend API: http://localhost:8000")
        print("\nPress Ctrl+C to stop all services...")

        # Keep the script running
        backend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend_process.terminate()
        user_matching_process.terminate()
        image_recommendation_process.terminate()
        product_search_process.terminate()
        print("Services stopped")


if __name__ == "__main__":
    main()
