import os
import glob
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image

from typing import List, Any, Optional
from torch.utils.data import Dataset, DataLoader


class AIC19Dataset(Dataset):

    def __init__(self,
                 classes_paths: List[str],
                 transform: Optional[Any] = None,
                 ):
        self.transform = transform
        self.classes_paths = classes_paths
        self.image_paths = []
        self.targets = []
        self.labels = []

        for i, class_path in enumerate(classes_paths):
            for image_path in glob.glob(os.path.join(class_path, "*.jpg")):
                self.image_paths.append(image_path)
                self.targets.append(i)
                self.labels.append(os.path.basename(class_path))

    def __len__(self):
        return len(self.image_paths)

    @property
    def num_classes(self):
        return len(self.classes_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert("RGB")).transpose(2, 0, 1)
        image = torch.tensor(image).float() / 255

        if self.transform:
            image = self.transform(image).squeeze()

        return image, self.targets[idx]
    

def create_dataloader(
    dataset_path: str,
    batch_size: int,
    inference: bool = False,
    input_size: int = 224,
    ):
    train_dirs = glob.glob(os.path.join(dataset_path, "*"))
    train_dirs.sort()
    # Sequence S03 is used for validation
    test_dirs = [dir for dir in train_dirs if "S03" in dir]
    test_dirs.sort()

    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size), antialias=True),
            # transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]),
            # transforms.ToTensor(),
        ])

    train_dataset = AIC19Dataset(train_dirs, transform)
    test_dataset = AIC19Dataset(test_dirs, transform)

    if not inference:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        train_dataset = test_dataset
        test_dataloader = test_dataset = None
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    return train_dataloader, test_dataloader