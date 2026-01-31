from setfit import SetFitModel, SetFitTrainer, TrainingArguments, sample_dataset, Trainer
from datasets import Dataset, load_dataset
from helpers import load_training_data

def main():
    model = SetFitModel.from_pretrained("intfloat/multilingual-e5-small")
    train_dataset, model.labels  = load_training_data("training_data.json")
    
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

    model.save_pretrained("model")

if __name__ == "__main__":
    main()