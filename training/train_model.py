import argparse
import glob
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


DEFAULT_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "blank"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


class CNNModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_dataset(
    data_dir: str,
    class_names: list[str],
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []

    for class_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        image_files = glob.glob(os.path.join(class_dir, "*.jpg"))

        print(f"[INFO] Loading class '{class_name}' from: {class_dir}")
        print(f"[INFO] Found {len(image_files)} images")

        for file_path in image_files:
            img = Image.open(file_path).convert("L")
            img = img.resize((image_width, image_height))
            array = np.array(img, dtype=np.float32) / 255.0

            images.append(array)
            labels.append(class_index)

    if len(images) == 0:
        raise ValueError(f"No images found in dataset directory: {data_dir}")

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int64),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def save_training_curves(
    train_acc_history: list[float],
    valid_acc_history: list[float],
    train_loss_history: list[float],
    valid_loss_history: list[float],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_acc_history, label="Train")
    plt.plot(valid_acc_history, label="Validation")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.grid()
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(train_loss_history, label="Train")
    plt.plot(valid_loss_history, label="Validation")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.grid()
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss.png"), dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a digit classifier for 7-segment OCR using PyTorch."
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="training_dataset_3300",
        help="Path to training dataset directory.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=28,
        help="Input image width.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=28,
        help="Input image height.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.10,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/model.pth",
        help="Path to save trained model.",
    )
    parser.add_argument(
        "--save-plots-dir",
        type=str,
        default="outputs/training",
        help="Directory to save training curves.",
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Do not save accuracy/loss plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    class_names = DEFAULT_CLASSES
    num_classes = len(class_names)

    print("[INFO] Loading dataset...")
    x_image, y_label = load_dataset(
        data_dir=args.train_data_path,
        class_names=class_names,
        image_width=args.image_width,
        image_height=args.image_height,
    )

    train_images, valid_images, train_labels, valid_labels = train_test_split(
        x_image,
        y_label,
        test_size=args.valid_ratio,
        random_state=args.seed,
        stratify=y_label,
    )

    print(f"[INFO] Train samples: {len(train_images)}")
    print(f"[INFO] Valid samples: {len(valid_images)}")

    train_dataset = ImageDataset(train_images, train_labels)
    valid_dataset = ImageDataset(valid_images, valid_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = CNNModel(num_classes=num_classes).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_acc_history = []
    valid_acc_history = []
    train_loss_history = []
    valid_loss_history = []

    start_time = time.time()

    print("[INFO] Start training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        valid_loss, valid_acc = evaluate(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device,
        )

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_acc)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}"
        )

    final_valid_loss, final_valid_acc = evaluate(
        model=model,
        dataloader=valid_loader,
        criterion=criterion,
        device=device,
    )

    elapsed = time.time() - start_time

    print(f"[RESULT] Validation Loss: {final_valid_loss:.4f}")
    print(f"[RESULT] Validation Accuracy: {final_valid_acc * 100:.2f}%")
    print(f"[RESULT] Training time: {elapsed:.3f} sec")

    output_model_path = Path(args.output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    print(f"[INFO] Model saved to: {output_model_path}")

    if not args.no_save_plots:
        save_training_curves(
            train_acc_history=train_acc_history,
            valid_acc_history=valid_acc_history,
            train_loss_history=train_loss_history,
            valid_loss_history=valid_loss_history,
            output_dir=args.save_plots_dir,
        )
        print(f"[INFO] Training curves saved to: {args.save_plots_dir}")


if __name__ == "__main__":
    main()

    