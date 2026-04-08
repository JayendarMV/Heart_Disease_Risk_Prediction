"""
run.py
========
Master runner script that simplifies starting the project.
It will first ensure the models are trained, and then launch the Flask server.
"""

import os
import sys
import subprocess
import time

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(root_dir, "model")
    model_path = os.path.join(model_dir, "heart_disease_model.pkl")

    # Step 1: Check if models exist, if not, train them.
    if not os.path.exists(model_path):
        print("Model files not found. Initiating training pipeline...")
        train_script = os.path.join(model_dir, "train.py")
        subprocess.run([sys.executable, train_script], check=True)
        time.sleep(2)
        print("\n\n")

    # Step 2: Start Flask Server
    print("=" * 60)
    print("  LAUNCHING HEARTGUARD FLASK SERVER")
    print("=" * 60)
    app_script = os.path.join(root_dir, "backend", "app.py")
    subprocess.run([sys.executable, app_script])

if __name__ == "__main__":
    main()
