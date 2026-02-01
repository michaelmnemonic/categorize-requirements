import json
from pathlib import Path
from setfit import SetFitModel, SetFitTrainer, TrainingArguments, sample_dataset, Trainer
from datasets import Dataset, load_dataset
import argparse

def main():
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--requirement', help='Requirement to categorize', type=str, required=True)
    args = parser.parse_args()
    requirement = args.requirement

    # Load model
    script_dir = Path(__file__).parent.resolve()
    model = SetFitModel.from_pretrained(script_dir.parent / "model")

    # Determine category
    preds = model.predict(requirement)
    print(preds)

    #probs = model.predict_proba(requirement)

    # print("-" * 30)
    # print("CATEGORIZATION RESULTS:")
    # print("-" * 30)

    # preds = model.predict(new_requirements)
    
    # scores = probs[0].tolist()
    # results = []

    # for label_id, score in enumerate(scores):
    #     results.append({
    #         "category": category_map[label_id],
    #         "score": score
    #     })

    # # Sort by score (highest first)
    # results.sort(key=lambda x: x["score"], reverse=True)

    # # Single choice
    # for req, label_id in zip(new_requirements, preds):
    #     category_name = category_map[int(label_id)]
    #     print(f"Requirement: '{req}'")
    #     print(f"-> Category: **{category_name}**\n")

    # # Multiple choice
    # print(f"Requirement: '{new_requirements[0]}'\n")
    # print("Matches:")
    # for res in results:
    #     # Convert score to percentage
    #     percentage = res['score'] * 100
    #     print(f"- {res['category']}: {percentage:.1f}%")

if __name__ == "__main__":
    main()
