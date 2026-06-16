from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class DurianLeafDatasetSingle(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, task="disease"):
        self.root_dir = Path(root_dir)
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        if task not in ["disease", "severity"]:
            raise ValueError(f"Unsupported task: {task}. Choose 'disease' or 'severity'.")
            
        self.task = task

        if self.task == "disease":
            self.df["label_id"] = self.df["label_id"].astype(int)
        elif self.task == "severity":
            self.df["severity"] = self.df["severity"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.root_dir / row["file_path"]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.task == "disease":
            label = int(row["label_id"])
        else:  # severity
            label = int(row["severity"])

        return image, label
