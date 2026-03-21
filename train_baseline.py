from pathlib import Path

import torch
from torch.optim import Adam

from src.datasets import get_fashion_mnist_loaders
from src.model import BaselineCNN
from src.train import train_one_epoch, evaluate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    batch_size = 64
    learning_rate = 1e-3
    epochs = 10

    train_loader, val_loader, test_loader = get_fashion_mnist_loaders(
        batch_size=batch_size
    )

    model = BaselineCNN().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    Path("results").mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "results/best_baseline.pt")

    model.load_state_dict(torch.load("results/best_baseline.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, device)

    print("\nBest validation accuracy:", round(best_val_acc, 4))
    print("Test accuracy:", round(test_acc, 4))


if __name__ == "__main__":
    main()
