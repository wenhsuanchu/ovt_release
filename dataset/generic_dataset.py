import os

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

class GenericDataset(Dataset):

    def __init__(self, image_root):

        self.root_dir = image_root
        self.video_dirs = os.listdir(image_root)

        self.img_transform = transforms.Resize(480, interpolation=InterpolationMode.BILINEAR, antialias=True)

    def __getitem__(self, idx):

        images = []
        frames = os.listdir(os.path.join(self.root_dir, self.video_dirs[idx]))
        frames.sort()

        for i, f in enumerate(frames):
            frame_path = os.path.join(self.root_dir, self.video_dirs[idx], f)
            img = Image.open(frame_path).convert("RGB")
            img = np.array(self.img_transform(img))
            images.append(np.array(img))

        images = np.stack(images[::3], 0)

        info = {
            "name": str(self.video_dirs[idx]),
            "id": self.video_dirs[idx],
            "size": images.shape[1:3],
            "size_480p": images.shape[1:3]
        }

        data = {
            "rgb": images,
            "info": info
        }

        return data

    def __len__(self):
        return len(self.video_dirs)
    
if __name__ == "__main__":

    PATH = "/projects/katefgroup/datasets/robotap/all_frames"
    test = GenericDataset(PATH)
    print(test[0]['rgb'].shape)