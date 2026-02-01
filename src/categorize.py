from pathlib import Path
from setfit import SetFitModel
import argparse

def predict_category(requirement: str):
    script_dir = Path(__file__).parent.resolve()
    model_path = script_dir.parent / "model"
    
    model = SetFitModel.from_pretrained(model_path)
    category = model.predict(requirement)
    
    return category

def main():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--requirement', help='Requirement to categorize', type=str, required=True)
    args = parser.parse_args()
    requirement = args.requirement

    # Determine category
    category = predict_category(requirement)
    print(category)

if __name__ == "__main__":
    main()
