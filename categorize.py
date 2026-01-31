import json
from setfit import SetFitModel, SetFitTrainer, TrainingArguments, sample_dataset, Trainer
from datasets import Dataset, load_dataset

def main():
    model = SetFitModel.from_pretrained("model")

    requirements = [
        "The login page must block IP addresses after 5 failed attempts.",
        "The video stream delay should be less than 50 milliseconds.",
        "If the cooling system fails, the reactor must shut down automatically."
    ]

    preds = model.predict(requirements)
    probs = model.predict_proba(requirements)
    print(preds)

    print("-" * 30)
    print("CATEGORIZATION RESULTS:")
    print("-" * 30)

    preds = model.predict(new_requirements)
    
    scores = probs[0].tolist()
    results = []

    for label_id, score in enumerate(scores):
        results.append({
            "category": category_map[label_id],
            "score": score
        })

    # Sort by score (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)

    # Single choice
    for req, label_id in zip(new_requirements, preds):
        category_name = category_map[int(label_id)]
        print(f"Requirement: '{req}'")
        print(f"-> Category: **{category_name}**\n")

    # Multiple choice
    print(f"Requirement: '{new_requirements[0]}'\n")
    print("Matches:")
    for res in results:
        # Convert score to percentage
        percentage = res['score'] * 100
        print(f"- {res['category']}: {percentage:.1f}%")

if __name__ == "__main__":
    main()
