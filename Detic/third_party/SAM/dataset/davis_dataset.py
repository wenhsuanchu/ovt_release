"""
Modified from https://github.com/seoungwugoh/STM/blob/master/dataset.py
"""

import os
from os import path
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.ops import masks_to_boxes
from torchvision.transforms import InterpolationMode
from torch.utils.data.dataset import Dataset
from dataset.util import all_to_onehot


class DAVISTestDataset(Dataset):
    def __init__(self, root, imset='2017/val.txt', resolution=480, target_name=None):
        self.root = root
        self.mask_dir = path.join(root, 'Annotations_480')
        self.image_dir = path.join(root, 'JPEGImages_480')
        self.resolution = resolution
        _imset_dir = path.join(root, 'ImageSets')
        _imset_f = path.join(_imset_dir, imset)

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=InterpolationMode.NEAREST),
        ])

        p_img = Image.new("P", (1,1))
        p_img.putpalette(np.load('/home/wenhsuac/ovt/Detic/color_palette_davis.npy').flatten())
        self.palette = p_img

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        with open(path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                if target_name is not None and target_name != _video:
                    continue
                self.videos.append(_video)
                self.num_frames[_video] = len(os.listdir(path.join(self.image_dir, _video)))
                _mask = Image.open(path.join(self.mask_dir, _video, '00000.png'))
                self.shape[_video] = np.shape(self.mask_transform(_mask))[:2]
                _mask = np.array(_mask.convert("P", palette=Image.Palette.ADAPTIVE, colors=len(_mask.getcolors())))
                self.num_objects[_video] = np.max(_mask)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['frames'] = []
        info['num_frames'] = self.num_frames[video]
        info['shape'] = self.shape[video]
        info['num_detected_objs'] = 0
        info['palette'] = None

        orig_images = []
        images = []
        masks = []
        scores = []
        for f in range(self.num_frames[video]):
            
            # Load RGBs
            img_file = path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            img = Image.open(img_file).convert('RGB')
            orig_images.append(torch.from_numpy(np.array(img)))
            images.append(np.array(img.resize((self.resolution[1], self.resolution[0]))))
            info['frames'].append('{:05d}.jpg'.format(f))
            
            # Load masks, quantize using the given DAVIS palette
            mask_file = path.join(self.mask_dir, video, '{:05d}.png'.format(f))
            if path.exists(mask_file):
                mask = Image.open(mask_file).convert("RGB")
                mask = mask.resize((info['shape'][1], info['shape'][0]), Image.NEAREST)
                mask = mask.quantize(palette=self.palette, dither=0)
                if f == 0:
                    info['palette'] = mask.getpalette()
                masks.append(mask)
            else:
                # Test-set maybe?
                masks.append(np.zeros_like(masks[0]))

        orig_images = torch.stack(orig_images, 0)
        images = np.stack(images, 0)
        masks = np.stack(masks, 0)

        # For masks, we need to convert to one hot labels
        gt_labels = np.unique(masks[0])
        gt_labels = gt_labels[gt_labels!=0]
        masks = torch.from_numpy(all_to_onehot(masks, gt_labels)).float()
        masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        data = {
            'orig_rgb': orig_images,
            'rgb': images,
            'gt': masks,
            'scores': scores,
            'info': info,
        }

        return data