import json
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from datasets import Dataset

def load_training_data(file_path):
    """Loads training data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JSON keys are strings, so we convert label IDs back to integers for the map.
    category_map = {int(k): v for k, v in data['category_map'].items()}
    
    # Re-structure the list of objects into a dictionary of lists for the Dataset.
    texts = [item['text'] for item in data['training_data']]
    labels = [item['label'] for item in data['training_data']]
    training_data_dict = {"text": texts, "label": labels}
    
    train_dataset = Dataset.from_dict(training_data_dict)
    
    return train_dataset, category_map

def main():
    # 1. LOAD THE "FEW-SHOT" DATA FROM JSON
    train_dataset, category_map = load_training_data('training_data.json')

    print("Loading model (this happens once)...")

    # 2. LOAD A PRE-TRAINED MODEL
    # 'all-MiniLM-L6-v2' is small, fast, and excellent for English technical text.
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # 3. TRAIN THE MODEL (FINE-TUNING)
    # SetFit uses contrastive learning (comparing pairs) to learn efficiently from small data.
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        loss_class=CosineSimilarityLoss,
        batch_size=16,
        num_epochs=1,
        num_iterations=20  # Generates 20 pairs per sentence to learn "similarity"
    )

    print("Training on user examples...")
    trainer.train()

    # 4. TEST WITH NEW, UNSEEN REQUIREMENTS
    new_requirements = [
        "The login page must block IP addresses after 5 failed attempts.",
        "The video stream delay should be less than 50 milliseconds.",
        "If the cooling system fails, the reactor must shut down automatically."
    ]

    print("-" * 30)
    print("CATEGORIZATION RESULTS:")
    print("-" * 30)

    preds = model.predict(new_requirements)

    for req, label_id in zip(new_requirements, preds):
        category_name = category_map[int(label_id)]
        print(f"Requirement: '{req}'")
        print(f"-> Category: **{category_name}**\n")

if __name__ == "__main__":
    main()
