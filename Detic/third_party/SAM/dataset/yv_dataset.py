import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot


class YouTubeVOSTestDataset(Dataset):
    def __init__(self, data_root, split='valid', load_backward=False):
        self.image_dir = path.join(data_root, 'all_frames', split+'_all_frames', 'JPEGImages')
        self.orig_image_dir = path.join(data_root, 'all_frames', split+'_all_frames', 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations')

        self.videos = []
        self.shape = {}
        self.frames = {}

        vid_list = sorted(os.listdir(self.image_dir))
        if load_backward:
            print("Loading backwards")
            vid_list = vid_list[::-1]
        # Pre-reading
        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.image_dir, vid)))
            self.frames[vid] = frames

            self.videos.append(vid)
            first_mask = os.listdir(path.join(self.mask_dir, vid))[0]
            _mask = np.array(Image.open(path.join(self.mask_dir, vid, first_mask)).convert("P"))
            self.shape[vid] = np.shape(_mask)

        self.img_transform = transforms.Resize(480, interpolation=InterpolationMode.BILINEAR, antialias=True)
        self.mask_transform = transforms.Resize(480, interpolation=InterpolationMode.NEAREST)

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video
        info['frames'] = self.frames[video] 
        info['size'] = self.shape[video] # Real sizes
        info['gt_obj'] = {} # Frames with labelled objects

        vid_im_path = path.join(self.image_dir, video)
        vid_gt_path = path.join(self.mask_dir, video)

        frames = self.frames[video]

        images = []
        masks = []
        for i, f in enumerate(frames):

            img = Image.open(path.join(vid_im_path, f)).convert('RGB')
            orig_shape = np.array(img).shape[:2]
            img = np.array(self.img_transform(img))
            images.append(np.array(img))

            if i == 0:
                info['shape'] = np.array(img).shape[:2]
            
            mask_file = path.join(vid_gt_path, f.replace('.jpg','.png'))
            if path.exists(mask_file):
                masks.append(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8))
                this_labels = np.unique(masks[-1])
                this_labels = this_labels[this_labels!=0]
                info['gt_obj'][i] = this_labels
            else:
                # Mask not exists -> nothing in it
                masks.append(np.zeros((orig_shape[0], orig_shape[1])))
            
        
        images = np.stack(images, 0)
        masks = np.stack(masks, 0)
        
        # Construct the forward and backward mapping table for labels
        # this is because YouTubeVOS's labels are sometimes not continuous
        # while we want continuous ones (for one-hot)
        # so we need to maintain a backward mapping table
        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels!=0]
        info['label_convert'] = {}
        info['label_backward'] = {}
        idx = 1
        for l in labels:
            info['label_convert'][l] = idx
            info['label_backward'][idx] = l
            idx += 1
        masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        # Resize to 480p
        masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)
    
    def _readFlow(self, filename):
        """ Read .flo file in Middlebury format"""
        # Code adapted from:
        # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

        # WARNING: this will work on little-endian architectures (eg Intel x86) only!
        # print 'fn = %s'%(fn)
        with open(filename, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print 'Reading %d x %d flo file\n' % (w, h)
                data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
                # Reshape testdata into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                return np.resize(data, (int(h), int(w), 2))