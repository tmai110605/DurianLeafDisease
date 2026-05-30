from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DurianLeafDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, use_severity=True):
        self.root_dir = Path(root_dir)
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.use_severity = use_severity

        self.df["label_id"] = self.df["label_id"].astype(int)

        if self.use_severity:
            self.df["severity"] = self.df["severity"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.root_dir / row["file_path"]
        image = Image.open(image_path).convert("RGB")

        disease_label = int(row["label_id"])

        if self.transform:
            image = self.transform(image)

        if self.use_severity:
            severity_label = int(row["severity"])
            return image, disease_label, severity_label

        return image, disease_label