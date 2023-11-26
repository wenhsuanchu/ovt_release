from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
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

        self.img_transform = transforms.Resize(480, interpolation=InterpolationMode.BILINEAR, antialias=True)

    def __getitem__(self, idx):
        video = self.dataset[idx]

        info = {}
        info["id"] = video.id
        info["dataset"] = video.dataset
        info["name"] = video.name
        info["negative_category_ids"] = video.negative_category_ids
        info["not_exhaustive_category_ids"] = video.not_exhaustive_category_ids
        info["size"] = video.image_size
        info['all_frames'] = video.all_images_paths
        info['required_frames'] = video.annotated_image_paths

        images = []
        paths = video.annotated_image_paths if self.annotated_only else video.all_images_paths[::5]
        paths = list(set(paths + video.annotated_image_paths))
        paths.sort()
        info['processed_frames'] = paths
        for i, f in enumerate(paths):
            img = Image.open(path.join(video._images_dir, f)).convert("RGB")
            img = np.array(self.img_transform(img))
            if i == 0:
                info['size_480p'] = img.shape[:2]
            images.append(np.array(img))

        images = np.stack(images, 0)

        data = {
            "rgb": images,
            "info": info,
        }

        return data

    def __len__(self):
        return self.dataset.num_videos