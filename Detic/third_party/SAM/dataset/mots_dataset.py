from os import path
import os

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

class MOTSDataset(Dataset):
    def __init__(self, data_root, split='train'):
        self.image_dir = path.join(data_root, split)

        self.videos = []
        self.shape = {}
        self.frames = {}

        vid_list = sorted(os.listdir(self.image_dir))
        # Pre-reading
        for vid in vid_list:
            vid_im_dir = os.path.join(self.image_dir, vid, 'img1')
            frames = sorted(os.listdir(vid_im_dir))
            self.frames[vid] = frames

            self.videos.append(vid)
            first_frame = np.array(Image.open(path.join(vid_im_dir, frames[0])))
            self.shape[vid] = np.shape(first_frame)[:2]

        self.img_transform = transforms.Resize(480, interpolation=InterpolationMode.BILINEAR, antialias=True)

    def __getitem__(self, idx):

        video = self.videos[idx]
        info = {}
        info['name'] = video
        info['id'] = idx
        info['frames'] = self.frames[video] 
        info['size'] = self.shape[video] # Real sizes

        frames = self.frames[video]
        vid_im_dir = os.path.join(self.image_dir, video, 'img1')

        images = []
        for i, f in enumerate(frames):

            img = Image.open(path.join(vid_im_dir, f)).convert('RGB')
            img = np.array(self.img_transform(img))
            if i == 0:
                info['size_480p'] = img.shape[:2]
            images.append(img)

        images = np.stack(images, 0)

        data = {
            'rgb': images,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)