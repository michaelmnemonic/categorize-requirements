from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset

def main():
    # 1. PREPARE THE "FEW-SHOT" DATA
    # In a real app, this would come from your user's input.
    # We map category names to numbers (Label ID) for the model.
    category_map = {0: "Functional-Safety", 1: "Performance", 2: "Security"}

    training_data = {
        "text": [
            # Functional-Safety Examples
            "The emergency brake must engage immediately if the sensor signal is lost.",
            "System must ensure redundant power supply activation within 10ms of failure.",
            
            # Performance Examples
            "The dashboard must render the main view in under 1.5 seconds.",
            "API latency must not exceed 200ms for 99% of requests.",
            
            # Security Examples
            "All user passwords must be hashed and salted before storage.",
            "The system must enforce a 15-minute session timeout for inactivity."
        ],
        "label": [0, 0, 1, 1, 2, 2]  # Corresponding label IDs for the text above
    }

    # Convert to a Hugging Face Dataset format
    train_dataset = Dataset.from_dict(training_data)

    print("Loading model (this happens once)...")

    # 2. LOAD A PRE-TRAINED MODEL
    # 'all-MiniLM-L6-v2' is small, fast, and excellent for English technical text.
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # 3. TRAIN THE MODEL (FINE-TUNING)
    # SetFit uses contrastive learning (comparing pairs) to learn efficiently from small data.
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        loss_class="CosineSimilarityLoss",
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
