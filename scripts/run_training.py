import torch

from src.data.dataset import EmotionDataset
from src.data.dataloader import create_dataloaders
from src.models.transformer_classifier import load_model
from src.training.train import train


def main():
    # --------------------
    # Configuration
    # --------------------
    model_name = "bert-base-uncased"
    num_labels = 6
    batch_size = 16
    max_length = 128
    epochs = 3
    learning_rate = 2e-5

    train_csv = "data/raw/train.csv"
    val_csv = "data/raw/validation.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------
    # Dataset & DataLoader
    # --------------------
    train_dataset = EmotionDataset(
        csv_path=train_csv,
        model_name=model_name,
        max_length=max_length
    )

    val_dataset = EmotionDataset(
        csv_path=val_csv,
        model_name=model_name,
        max_length=max_length
    )

    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size
    )

    # --------------------
    # Model
    # --------------------
    model = load_model(
        model_name=model_name,
        num_labels=num_labels
    )

    # --------------------
    # Training
    # --------------------
    model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        freeze_bert=freeze_bert
    )

    # --------------------
    # Save model
    # --------------------
    output_path = "models/bert_emotion_classifier.pt"
    torch.save(model.state_dict(), output_path)

    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    main()
