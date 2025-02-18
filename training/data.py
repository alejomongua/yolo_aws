import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from train_config import BATCH_SIZE


class PreprocessedVOCDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory containing preprocessed .npz files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(data_dir, '*.npz'))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)
        image = data['image']  # Expected shape: (H, W, 3)
        # Expected shape: (N, 5) where each row is [x_center, y_center, width, height, class_id]
        boxes = data['boxes']

        # Convert image (NumPy array) to a PIL image for transformation
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor without normalization.
            image = transforms.ToTensor()(image)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        return image, boxes


def get_datasets(base_dir):
    """
    Assumes base_dir contains three subdirectories: 'train', 'val', and 'test',
    each holding the corresponding preprocessed .npz files.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PreprocessedVOCDataset(
        os.path.join(base_dir, 'train'), transform=transform)
    val_dataset = PreprocessedVOCDataset(
        os.path.join(base_dir, 'val'), transform=transform)
    test_dataset = PreprocessedVOCDataset(
        os.path.join(base_dir, 'test'), transform=transform)
    return train_dataset, val_dataset, test_dataset


def get_data_loaders(data_dir, batch_size=BATCH_SIZE, num_workers=4):
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
