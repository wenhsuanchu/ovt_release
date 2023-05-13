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
    def __init__(self, root, imset='2017/val.txt', resolution=480, target_name=None, load_saved_detections=False):
        self.root = root
        self.mask_dir = path.join(root, 'Annotations_480')
        self.detection_dir = path.join(root, 'Detected_Annotations_NPZ_480')
        self.flow_dir = path.join(root, 'Flow_480')
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

        self.load_saved_detections = load_saved_detections

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
        detections = []
        detected_boxes = []
        scores = []
        fwd_flows = []
        bwd_flows = []
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

            # Load detections
            if self.load_saved_detections:
                detection_file = path.join(self.detection_dir, video, '{:05d}.npz'.format(f))
                if path.exists(detection_file):
                    detection_npz = np.load(detection_file)

                    # Detectron should already sort based on confidence, but just in case
                    score = torch.from_numpy(detection_npz['scores'])
                    sorted_score, sort_idxes = torch.sort(score, descending=True)
                    scores.append(sorted_score)

                    detection = torch.from_numpy(detection_npz['mask']).unsqueeze(1).float() # [N, H, W] -> [N, 1, H, W]
                    detection = torch.nn.functional.interpolate(detection, (info['shape'][0], info['shape'][1]), mode='nearest') # [N, 1, H, W]
                    detection = (detection > 0.5).float()
                    detections.append(detection[sort_idxes])

                    boxes = masks_to_boxes(detection[sort_idxes].squeeze(1))
                    detected_boxes.append(boxes)
                else:
                    print(f"Missing detection file at {detection_file}.")
                    detections.append(torch.zeros_like(detections[0]))
                    detected_boxes.append(torch.zeros_like(detected_boxes[0]))
                    scores.append(torch.zeros_like(scores[0]))

            # Load flow
            fwd_flow_file = path.join(self.flow_dir, video, '{:05d}_fwd.flo'.format(f))
            if path.exists(fwd_flow_file):
                fwd_flow = self._readFlow(fwd_flow_file)
                fwd_flows.append(fwd_flow)
            else:
                print(f"Missing flow file at {fwd_flow_file}.")
                fwd_flows.append(np.zeros_like(fwd_flows[0]))
            bwd_flow_file = path.join(self.flow_dir, video, '{:05d}_bwd.flo'.format(f))
            if path.exists(bwd_flow_file):
                bwd_flow = self._readFlow(bwd_flow_file)
                bwd_flows.append(bwd_flow)
            else:
                print(f"Missing flow file at {bwd_flow_file}.")
                bwd_flows.append(np.zeros_like(bwd_flows[0]))

        orig_images = torch.stack(orig_images, 0)
        images = np.stack(images, 0)
        masks = np.stack(masks, 0)
        fwd_flows = np.stack(fwd_flows, 0)
        bwd_flows = np.stack(bwd_flows, 0)

        # For masks, we need to convert to one hot labels
        gt_labels = np.unique(masks[0])
        gt_labels = gt_labels[gt_labels!=0]
        masks = torch.from_numpy(all_to_onehot(masks, gt_labels)).float()
        masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        data = {
            'orig_rgb': orig_images,
            'rgb': images,
            'fwd_flow': fwd_flows,
            'bwd_flow': bwd_flows,
            'gt': masks,
            'detections': detections,
            'detected_boxes': detected_boxes,
            'scores': scores,
            'info': info,
        }

        return data

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