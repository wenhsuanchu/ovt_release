from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from ytvostools.ytvos import YTVOS
from ytvostools.mask import decode as rle_decode

#frame_dir = '/projects/katefgroup/datasets/UVO/uvo_videos_dense_frames'
#annFile = '/projects/katefgroup/datasets/UVO/VideoDenseSet/UVO_video_val_dense.json'
#annFile = '/projects/katefgroup/datasets/UVO/VideoDenseSet/UVO_video_val_dense_with_label.json'

class UVOTestDataset(Dataset):
    def __init__(self, data_root, load_backward=False):
        self.image_dir = path.join(data_root, 'uvo_videos_dense_frames')
        self.annFile = path.join(data_root, 'VideoDenseSet', 'UVO_video_val_dense_with_label.json')

        self.uvo = YTVOS(self.annFile)
        self.vid_ids = self.uvo.getVidIds()
        if load_backward:
            print("Loading backwards")
            self.vid_ids = list(reversed(self.vid_ids))

    def __getitem__(self, idx):
        vid = self.uvo.loadVids(self.vid_ids[idx])[0]
        #annIds = self.uvo.getAnnIds(vidIds=vid['id'])
        #anns = self.uvo.loadAnns(annIds)

        info = {}
        info['id'] = vid['id']
        info['name'] = vid['ytid']
        info['size'] = (vid['height'], vid['width']) # Real sizes

        images = []
        orig_images = []
        #masks = np.stack([rle_decode(anns[i]['segmentations']) for i in range(len(anns))], 0) # [N, H, W, S]
        for i, f in enumerate(vid['file_names']):

            img = Image.open(path.join(self.image_dir, f)).convert('RGB')
            orig_images.append(torch.from_numpy(np.array(img)))
            images.append(np.array(img))

        orig_images = torch.stack(orig_images, 0)
        images = np.stack(images, 0)

        #masks = torch.from_numpy(masks).permute(3,0,1,2).float() # [N, H, W, S] -> [S, N, H, W]
        masks = torch.zeros(images.shape[0], 1, info['size'][0], info['size'][1])
        masks = masks.unsqueeze(2)

        data = {
            'orig_rgb': orig_images,
            'rgb': images,
            'gt': masks,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.vid_ids)