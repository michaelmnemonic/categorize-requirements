import json
from datasets import Dataset

def load_training_data(file_path):
    """Loads training data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # extract labels
    labels = data['labels']

    # extract dataset
    train_dataset = Dataset.from_list(data['data'])

    return train_dataset, labels
