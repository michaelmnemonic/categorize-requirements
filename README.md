# Categorize Requirements

This project uses machine learning (SetFit - Sentence Transformer Fine-tuning) to automatically categorize software requirements into specific types, such as **Security**, **Performance**, and **Functional-Safety**.

## Project Structure

- **`src/`**: Contains the source code.
  - `categorize.py`: CLI tool to predict the category of a given requirement.
  - `train.py`: Script to train the SetFit model using data from `data/training_data.json`.
  - `main.py`: Entry point that checks for a trained model and triggers training if one is not found.
  - `helpers.py`: Helper functions for data loading.
  - `test_*.py`: Unit and integration tests.
- **`data/`**: Contains the training dataset.
  - `training_data.json`: Labeled requirements for training.
- **`model/`**: Directory where the trained model is saved (auto-generated).

## Setup

This project is set up with a `flake.nix` for a reproducible development environment.

```bash
nix develop
```

This will enter a shell with Python 3.13, SetFit, and all necessary dependencies installed.

## Usage

### Training the Model

To train the model, run:

```bash
python src/train.py
```

Alternatively, you can run the main script, which will verify if a model exists and train it if necessary:

```bash
python src/main.py
```

The model will be saved to the `model/` directory.

### Categorizing Requirements

To categorize a specific requirement, use the `categorize.py` script:

```bash
python src/categorize.py --requirement "The login page must block IP addresses after 5 failed attempts."
```

**Output:**
```
Security
```

## Testing

The project uses `pytest` for testing. To run the tests:

```bash
pytest
```

## Data Format

The `data/training_data.json` file should follow this structure:

```json
{
  "labels": ["Functional-Safety", "Performance", "Security"],
  "data": [
    { "text": "Requirement text...", "label": 0 },
    ...
  ]
}
```

- `text`: The requirement string.
- `label`: Integer index corresponding to the `labels` array (e.g., 0 for "Functional-Safety").
