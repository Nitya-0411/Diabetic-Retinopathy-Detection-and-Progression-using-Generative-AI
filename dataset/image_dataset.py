from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import random

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, unaligned=False):
        self.transform = transform
        self.unaligned = unaligned

        self.root_A = os.path.join(root, "trainA")
        self.root_B = os.path.join(root, "trainB")

        # Recursively load all image files from subfolders
        self.files_A = sorted(glob.glob(os.path.join(self.root_A, "**", "*.*"), recursive=True))
        self.files_B = sorted(glob.glob(os.path.join(self.root_B, "**", "*.*"), recursive=True))

    def __getitem__(self, index):
        img_A_path = self.files_A[index % len(self.files_A)]
        img_A = Image.open(img_A_path).convert("RGB")

        if self.unaligned:
            img_B_path = random.choice(self.files_B)
        else:
            img_B_path = self.files_B[index % len(self.files_B)]
        img_B = Image.open(img_B_path).convert("RGB")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
