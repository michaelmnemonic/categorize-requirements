import sys
import os
import subprocess
from pathlib import Path
from setfit import SetFitModel

def main():
    # Ensure we are in the src directory for relative paths in other scripts to work
    script_dir = Path(__file__).parent.resolve()
    
    model_path = script_dir.parent / "model"
    
    print(f"Checking for model in {model_path}...")
    sys.stdout.flush()
    
    try:
        # Try to load the model to verify it's usable
        model = SetFitModel.from_pretrained(str(model_path))
        print("Model loaded successfully.")
        sys.stdout.flush()
    except Exception as e:
        print(f"Model could not be loaded: {e}")
        print("Starting training...")
        sys.stdout.flush()
        try:
            # Use sys.executable to ensure we use the same python interpreter
            subprocess.run([sys.executable, "train.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training failed with error code {e.returncode}")
            sys.exit(1)
            
    print("-" * 30)
    print("Running categorization...")
    sys.stdout.flush()
    try:
        subprocess.run([sys.executable, "categorize.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Categorization failed with error code {e.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
