import json
from setfit import SetFitModel, SetFitTrainer, TrainingArguments, sample_dataset, Trainer
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import Dataset, load_dataset

def load_training_data(file_path):
    """Loads training data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # extract labels
    labels = data['labels']

    # extract dataset
    train_dataset = Dataset.from_list(data['data'])

    return train_dataset, labels

def main():
    # 2. LOAD A PRE-TRAINED MODEL
    # 'all-MiniLM-L6-v2' is small, fast, and excellent for English technical text.
    model = SetFitModel.from_pretrained("intfloat/multilingual-e5-small")

    train_dataset, labels = load_training_data("training_data.json")
        
    model.labels = labels

    args = TrainingArguments(
        batch_size=32,
        num_epochs=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    trainer.train()

    trainer.evaluate(train_dataset)

    model.save_pretrained("trained")

    #model = SetFitModel.from_pretrained("trained")

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
