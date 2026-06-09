import argparse
import json
import time
from pathlib import Path

import pandas as pd
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

from dataset_single import DurianLeafDatasetSingle
from utils import set_seed, get_device, count_parameters, calculate_flops

# Import single-task models
from mobilenetv3_single import (
    MobileNetV2SingleTask,
    MobileNetV3SmallSingleTask,
    MobileNetV3LargeSingleTask
)
from resnet_single import (
    ResNet18SingleTask,
    ResNet50SingleTask
)
from densenet_single import DenseNet121SingleTask
from ticknet_single import TickNetLargeSingleTask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train single-task durian leaf disease/severity model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "mobilenetv2",
            "mobilenetv3_small",
            "mobilenetv3_large",
            "resnet18",
            "resnet50",
            "densenet121",
            "ticknet_large"
        ],
        help="Model architecture to train"
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["disease", "severity"],
        help="Target prediction task"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size"
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=IMAGE_SIZE,
        help="Input image size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=WEIGHT_DECAY,
        help="Weight decay"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate"
    )

    parser.add_argument(
        "--use_class_weight",
        action="store_true",
        help="Use class-weighted CrossEntropyLoss"
    )

    parser.add_argument(
        "--save_all",
        action="store_true",
        help="Save checkpoint for every epoch"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers"
    )

    return parser.parse_args()


def build_model(model_name, num_classes, dropout):
    if model_name == "mobilenetv2":
        return MobileNetV2SingleTask(
            num_classes=num_classes,
            dropout=dropout
        )

    if model_name == "mobilenetv3_small":
        return MobileNetV3SmallSingleTask(
            num_classes=num_classes,
            dropout=dropout
        )

    if model_name == "mobilenetv3_large":
        return MobileNetV3LargeSingleTask(
            num_classes=num_classes,
            dropout=dropout
        )

    if model_name == "resnet18":
        return ResNet18SingleTask(
            num_classes=num_classes,
            dropout=dropout
        )

    if model_name == "resnet50":
        return ResNet50SingleTask(
            num_classes=num_classes,
            dropout=dropout
        )

    if model_name == "densenet121":
        return DenseNet121SingleTask(
            num_classes=num_classes,
            dropout=dropout
        )

    if model_name == "ticknet_large":
        return TickNetLargeSingleTask(
            num_classes=num_classes,
            dropout=dropout
        )

    raise ValueError(f"Unsupported model: {model_name}")


def remove_thop_buffers(model):
    """
    THOP may add total_ops and total_params buffers to modules.
    Remove them before saving model.state_dict().
    """
    for module in model.modules():
        if hasattr(module, "total_ops"):
            delattr(module, "total_ops")
        if hasattr(module, "total_params"):
            delattr(module, "total_params")


def build_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
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
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


def get_class_weights(task, device):
    train_df = pd.read_csv(TRAIN_CSV)

    if task == "disease":
        col = "label_id"
        num_classes = NUM_DISEASE_CLASSES
    else:
        col = "severity"
        num_classes = NUM_SEVERITY_CLASSES

    train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

    if train_df[col].isna().any():
        bad_rows = train_df[train_df[col].isna()]
        raise ValueError(
            f"Found missing/invalid {col} values in TRAIN_CSV.\n"
            f"{bad_rows[['file_name', 'file_path', 'disease_type', col]].head(20)}"
        )

    train_df[col] = train_df[col].astype(int)

    counts = train_df[col].value_counts().sort_index()

    class_counts = torch.tensor(
        [counts.get(i, 0) for i in range(num_classes)],
        dtype=torch.float
    )

    class_counts = torch.clamp(class_counts, min=1.0)

    weights = 1.0 / class_counts
    weights = weights / weights.sum() * num_classes
    weights = weights.to(device)

    return class_counts, weights


def save_json_log(log_path, history):
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


def main():
    args = parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_classes = NUM_DISEASE_CLASSES if args.task == "disease" else NUM_SEVERITY_CLASSES

    run_name = (
        f"{args.model}"
        f"_task_{args.task}"
        f"_img{args.image_size}"
        f"_bs{args.batch_size}"
        f"_lr{args.lr}"
    )

    if args.use_class_weight:
        run_name += "_classweight"

    run_dir = CHECKPOINT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run name: {run_name}")
    print(f"Checkpoint directory: {run_dir}")

    train_transform, val_transform = build_transforms(args.image_size)

    train_dataset = DurianLeafDatasetSingle(
        root_dir=TRAIN_ROOT,
        csv_file=TRAIN_CSV,
        transform=train_transform,
        task=args.task
    )

    val_dataset = DurianLeafDatasetSingle(
        root_dir=VAL_ROOT,
        csv_file=VAL_CSV,
        transform=val_transform,
        task=args.task
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = build_model(
        model_name=args.model,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    flops, thop_params, flops_readable, params_readable = calculate_flops(
        model=model,
        input_size=(1, 3, args.image_size, args.image_size),
        device=device
    )

    remove_thop_buffers(model)

    print("=" * 60)
    print("Model complexity")
    print(f"Model:                {args.model}")
    print(f"Task:                 {args.task} ({num_classes} classes)")
    print(f"Input size:           {args.image_size}x{args.image_size}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"THOP parameters:      {params_readable}")
    print(f"FLOPs:                {flops_readable}")
    print("=" * 60)

    if args.use_class_weight:
        class_counts, class_weights = get_class_weights(args.task, device)

        print(f"Task {args.task} class counts:", class_counts.cpu().tolist())
        print(f"Task {args.task} class weights:", class_weights.detach().cpu().tolist())

        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_loss = float("inf")
    best_model_path = run_dir / f"best_{args.model}_{args.task}.pth"

    history = {
        "run_name": run_name,
        "model": args.model,
        "task": args.task,
        "num_classes": num_classes,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "use_class_weight": args.use_class_weight,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "flops": int(flops),
        "flops_readable": flops_readable,
        "params_readable": params_readable,
        "epochs_log": []
    }

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        loop = tqdm(
            train_loader,
            desc=f"{args.model} | Epoch {epoch + 1}/{args.epochs}"
        )

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)

            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_dataset)
        train_acc = train_correct / train_total

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                preds = outputs.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        val_loss /= len(val_dataset)
        val_acc = val_correct / val_total

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }

        history["epochs_log"].append(epoch_log)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        checkpoint_data = {
            "model_name": args.model,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,

            "train_loss": train_loss,
            "train_acc": train_acc,

            "val_loss": val_loss,
            "val_acc": val_acc,

            "total_params": total_params,
            "trainable_params": trainable_params,
            "flops": flops,
            "flops_readable": flops_readable,
            "params_readable": params_readable,

            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "task": args.task,
            "num_classes": num_classes,
            "use_class_weight": args.use_class_weight
        }

        if args.save_all:
            epoch_model_path = run_dir / f"{args.model}_epoch_{epoch + 1:03d}.pth"
            torch.save(checkpoint_data, epoch_model_path)
            print(f"Saved epoch checkpoint: {epoch_model_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_data, best_model_path)
            print(f"Saved best model: {best_model_path}")

        save_json_log(run_dir / "training_log.json", history)

    total_time = time.time() - start_time
    history["total_training_time_seconds"] = total_time
    save_json_log(run_dir / "training_log.json", history)

    print("=" * 60)
    print("Training finished")
    print(f"Best model saved at: {best_model_path}")
    print(f"Training log saved at: {run_dir / 'training_log.json'}")
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()
