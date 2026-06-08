import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        "uvicorn", "app.main:app",
        "--host", "0.0.0.0",
        "--port", "8080"
    ])
