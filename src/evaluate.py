import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from config import (
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

from dataset import DurianLeafDataset
from mobilenetv3_custom import MobileNetV3SmallMultiTask
from utils import get_device


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


def main():
    device = get_device(DEVICE)
    print(f"Using device: {device}")

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = DurianLeafDataset(
        root_dir=TEST_ROOT,
        csv_file=TEST_CSV,
        transform=test_transform,
        use_severity=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = MobileNetV3SmallMultiTask(
        num_disease_classes=NUM_DISEASE_CLASSES,
        num_severity_classes=NUM_SEVERITY_CLASSES
    ).to(device)

    checkpoint_path = CHECKPOINT_DIR / "best_mobilenetv3_multitask.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    disease_true = []
    disease_pred = []
    severity_true = []
    severity_pred = []

    with torch.no_grad():
        for images, disease_labels, severity_labels in test_loader:
            images = images.to(device)

            disease_outputs, severity_outputs = model(images)

            disease_predictions = disease_outputs.argmax(dim=1).cpu().numpy()
            severity_predictions = severity_outputs.argmax(dim=1).cpu().numpy()

            disease_pred.extend(disease_predictions)
            severity_pred.extend(severity_predictions)

            disease_true.extend(disease_labels.numpy())
            severity_true.extend(severity_labels.numpy())

    disease_names = [
        "healthy",
        "algal",
        "allocaridara_attack",
        "blight",
        "phomopsis"
    ]

    severity_names = [
        "healthy",
        "mild",
        "moderate",
        "severe"
    ]

    disease_report = classification_report(
        disease_true,
        disease_pred,
        target_names=disease_names,
        digits=4
    )

    severity_report = classification_report(
        severity_true,
        severity_pred,
        target_names=severity_names,
        digits=4
    )

    print("Disease Classification Report")
    print(disease_report)

    print("Severity Classification Report")
    print(severity_report)

    report_path = RESULT_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Disease Classification Report\n")
        f.write(disease_report)
        f.write("\n\nSeverity Classification Report\n")
        f.write(severity_report)

    disease_cm = confusion_matrix(disease_true, disease_pred)
    severity_cm = confusion_matrix(severity_true, severity_pred)

    plot_confusion_matrix(
        disease_cm,
        disease_names,
        RESULT_DIR / "confusion_matrix_disease.png",
        "Disease Confusion Matrix"
    )

    plot_confusion_matrix(
        severity_cm,
        severity_names,
        RESULT_DIR / "confusion_matrix_severity.png",
        "Severity Confusion Matrix"
    )

    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()