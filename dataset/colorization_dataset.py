import glob

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab
import numpy as np


SIZE = 256


class ColorizationDataset(Dataset):
    def __init__(self, paths, split="train"):
        compose = [transforms.Resize((SIZE, SIZE), Image.BICUBIC)]
        if split == "train":
            compose.append(transforms.RandomHorizontalFlip())
        self.transforms = transforms.Compose(compose)
        self.split = split
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = np.array(self.transforms(img))
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50 - 1.
        ab = img_lab[[1, 2], ...] / 110.

        return {"L": L, "ab": ab}

    def __len__(self):
        return len(self.paths)


def make_dataloader(batch_size=1, n_workers=1, pin_memory=True):
    valid_paths = glob.glob("valid_path/*.jpg")
    dataset = ColorizationDataset(paths=valid_paths, split="valid")
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=pin_memory)
    return dataloader
