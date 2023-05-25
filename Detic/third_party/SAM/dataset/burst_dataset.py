from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from dataset.burstapi.dataset import BURSTDataset


class BURSTTestDataset(Dataset):
    def __init__(self, annotation_json, image_root, load_backward=False, annotated_only=False):
        self.dataset = BURSTDataset(annotation_json, image_root)

        self.annotated_only = annotated_only
        print("Load annotated only: ", self.annotated_only)
        if load_backward:
            self.dataset._videos = list(reversed(self.dataset._videos))

    def __getitem__(self, idx):
        video = self.dataset[idx]

        info = {}
        info["id"] = video.id
        info["name"] = video.name
        info["size"] = video.image_size

        images = []
        orig_images = []
        paths = video.annotated_image_paths if self.annotated_only else video.all_images_paths[::10]
        for i, f in enumerate(paths):
            img = Image.open(path.join(video._images_dir, f)).convert("RGB")
            orig_images.append(torch.from_numpy(np.array(img)))
            images.append(np.array(img))

        orig_images = torch.stack(orig_images, 0)
        images = np.stack(images, 0)

        # No groundtruth masks provided for testing.
        masks = torch.zeros(images.shape[0], 1, info["size"][0], info["size"][1])
        masks = masks.unsqueeze(2)

        data = {
            "orig_rgb": orig_images,
            "rgb": images,
            "gt": masks,
            "info": info,
        }

        return data

    def __len__(self):
        return self.dataset.num_videos