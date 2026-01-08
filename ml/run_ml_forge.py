"""
ML-FORGE Runner Script

This script ensures the proper environment setup before running the ML-FORGE application.
It checks for required dependencies and launches the main application.
"""

import sys
import subprocess
import os

def check_and_install_dependencies():
    """Check for required dependencies and install if missing"""
    required_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "tensorflow"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing required package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    check_and_install_dependencies()
    print("Starting ML-FORGE application...")
    
    # Run the Streamlit application
    subprocess.call(["streamlit", "run", "app.py"] + sys.argv[1:])
