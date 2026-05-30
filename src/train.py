import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from config import (
    TRAIN_ROOT,
    VAL_ROOT,
    TRAIN_CSV,
    VAL_CSV,
    CHECKPOINT_DIR,
    NUM_DISEASE_CLASSES,
    NUM_SEVERITY_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE
)

from dataset import DurianLeafDataset
from mobilenetv3_custom import MobileNetV3SmallMultiTask
from utils import set_seed, get_device


def main():
    set_seed(42)

    device = get_device(DEVICE)
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = DurianLeafDataset(
        root_dir=TRAIN_ROOT,
        csv_file=TRAIN_CSV,
        transform=train_transform,
        use_severity=True
    )

    val_dataset = DurianLeafDataset(
        root_dir=VAL_ROOT,
        csv_file=VAL_CSV,
        transform=val_transform,
        use_severity=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = MobileNetV3SmallMultiTask(
        num_disease_classes=NUM_DISEASE_CLASSES,
        num_severity_classes=NUM_SEVERITY_CLASSES
    ).to(device)

    criterion_disease = nn.CrossEntropyLoss()
    criterion_severity = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_model_path = CHECKPOINT_DIR / "best_mobilenetv3_multitask.pth"

    for epoch in range(NUM_EPOCHS):
        model.train()

        train_loss = 0.0
        train_disease_correct = 0
        train_severity_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for images, disease_labels, severity_labels in loop:
            images = images.to(device)
            disease_labels = disease_labels.to(device)
            severity_labels = severity_labels.to(device)

            optimizer.zero_grad()

            disease_outputs, severity_outputs = model(images)

            loss_disease = criterion_disease(disease_outputs, disease_labels)
            loss_severity = criterion_severity(severity_outputs, severity_labels)

            loss = loss_disease + loss_severity

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            disease_preds = disease_outputs.argmax(dim=1)
            severity_preds = severity_outputs.argmax(dim=1)

            train_disease_correct += (disease_preds == disease_labels).sum().item()
            train_severity_correct += (severity_preds == severity_labels).sum().item()
            train_total += images.size(0)

            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_dataset)
        train_disease_acc = train_disease_correct / train_total
        train_severity_acc = train_severity_correct / train_total

        model.eval()

        val_loss = 0.0
        val_disease_correct = 0
        val_severity_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, disease_labels, severity_labels in val_loader:
                images = images.to(device)
                disease_labels = disease_labels.to(device)
                severity_labels = severity_labels.to(device)

                disease_outputs, severity_outputs = model(images)

                loss_disease = criterion_disease(disease_outputs, disease_labels)
                loss_severity = criterion_severity(severity_outputs, severity_labels)
                loss = loss_disease + loss_severity

                val_loss += loss.item() * images.size(0)

                disease_preds = disease_outputs.argmax(dim=1)
                severity_preds = severity_outputs.argmax(dim=1)

                val_disease_correct += (disease_preds == disease_labels).sum().item()
                val_severity_correct += (severity_preds == severity_labels).sum().item()
                val_total += images.size(0)

        val_loss /= len(val_dataset)
        val_disease_acc = val_disease_correct / val_total
        val_severity_acc = val_severity_correct / val_total

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Disease Acc: {train_disease_acc:.4f} | "
            f"Train Severity Acc: {train_severity_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Disease Acc: {val_disease_acc:.4f} | "
            f"Val Severity Acc: {val_severity_acc:.4f}"
        )
        epoch_model_path = CHECKPOINT_DIR / f"mobilenetv3_epoch_{epoch + 1:03d}.pth"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_disease_acc": train_disease_acc,
                "train_severity_acc": train_severity_acc,
                "val_loss": val_loss,
                "val_disease_acc": val_disease_acc,
                "val_severity_acc": val_severity_acc
            },
            epoch_model_path
        )

        print(f"Saved epoch checkpoint: {epoch_model_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_disease_acc": val_disease_acc,
                    "val_severity_acc": val_severity_acc
                },
                best_model_path
            )
            print(f"Saved best model: {best_model_path}")


if __name__ == "__main__":
    main()