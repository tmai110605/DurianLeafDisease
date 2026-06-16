import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

import matplotlib.pyplot as plt
import numpy as np

from training.config import (
    TEST_ROOT,
    TEST_CSV,
    CHECKPOINT_DIR,
    RESULT_DIR,
    NUM_DISEASE_CLASSES,
    NUM_SEVERITY_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    DEVICE
)

from training.dataset_single import DurianLeafDatasetSingle
from training.utils import get_device

# Import single-task models
from models.mobilenetv3_single import (
    MobileNetV2SingleTask,
    MobileNetV3SmallSingleTask,
    MobileNetV3LargeSingleTask
)
from models.resnet_single import (
    ResNet18SingleTask,
    ResNet50SingleTask
)
from models.densenet_single import DenseNet121SingleTask
from models.ticknet_single import TickNetLargeSingleTask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate single-task durian leaf disease/severity model"
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
        help="Model architecture to evaluate"
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["disease", "severity"],
        help="Target task to evaluate"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pth file"
    )

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory containing best checkpoint"
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=IMAGE_SIZE,
        help="Input image size"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate used when building model"
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


def resolve_checkpoint_path(args):
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)

        if not run_dir.exists():
            run_dir = CHECKPOINT_DIR / args.run_dir

        checkpoint_path = run_dir / f"best_{args.model}_{args.task}.pth"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        return checkpoint_path

    # Default case: find checkpoint in CHECKPOINT_DIR
    checkpoint_path = CHECKPOINT_DIR / f"best_{args.model}_{args.task}.pth"

    if not checkpoint_path.exists():
        # Fall back to checking folder inside checkpoint dir matching naming conventions
        # For simplicity, prompt user if default path isn't found
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Please provide --checkpoint or --run_dir."
        )

    return checkpoint_path


def clean_state_dict(state_dict):
    """
    Remove THOP buffers if they exist.
    """
    return {
        k: v for k, v in state_dict.items()
        if not k.endswith("total_ops") and not k.endswith("total_params")
    }


def plot_confusion_matrix(cm, class_names, save_path, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_metrics_json(save_path, metrics):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def main():
    args = parse_args()

    device = get_device(DEVICE)
    print(f"Using device: {device}")

    checkpoint_path = resolve_checkpoint_path(args)
    print(f"Using checkpoint: {checkpoint_path}")

    result_dir = RESULT_DIR / f"eval_single_{args.model}_{args.task}_{checkpoint_path.parent.name}"
    result_dir.mkdir(parents=True, exist_ok=True)

    num_classes = NUM_DISEASE_CLASSES if args.task == "disease" else NUM_SEVERITY_CLASSES

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = DurianLeafDatasetSingle(
        root_dir=TEST_ROOT,
        csv_file=TEST_CSV,
        transform=test_transform,
        task=args.task
    )

    test_loader = DataLoader(
        test_dataset,
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = clean_state_dict(checkpoint["model_state_dict"])

    model.load_state_dict(state_dict)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(predictions)
            y_true.extend(labels.numpy())

    class_names = [
        "healthy",
        "algal",
        "allocaridara_attack",
        "blight",
        "phomopsis"
    ] if args.task == "disease" else [
        "healthy",
        "mild",
        "moderate",
        "severe"
    ]

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    print(f"=== {args.task.capitalize()} Classification Report ===")
    print(report)

    acc = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    metrics = {
        "model": args.model,
        "task": args.task,
        "checkpoint": str(checkpoint_path),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }

    # Extract dynamic stats from checkpoint if available
    for key in [
        "total_params",
        "trainable_params",
        "flops",
        "flops_readable",
        "params_readable",
        "epoch",
        "val_loss",
        "val_acc"
    ]:
        if key in checkpoint:
            metrics[key] = checkpoint[key]

    report_path = result_dir / f"classification_report_{args.task}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n\n")
        f.write(f"{args.task.capitalize()} Classification Report\n")
        f.write(report)

    save_metrics_json(result_dir / "metrics.json", metrics)

    cm = confusion_matrix(y_true, y_pred)

    plot_confusion_matrix(
        cm,
        class_names,
        result_dir / f"confusion_matrix_{args.task}.png",
        f"{args.task.capitalize()} Confusion Matrix"
    )

    print(f"Saved report to: {report_path}")
    print(f"Saved metrics to: {result_dir / 'metrics.json'}")
    print(f"Saved confusion matrix to: {result_dir / f'confusion_matrix_{args.task}.png'}")


if __name__ == "__main__":
    main()
