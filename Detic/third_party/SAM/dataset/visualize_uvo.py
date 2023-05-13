from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import os

from ytvostools.ytvos import YTVOS
from ytvostools.mask import decode as rle_decode

out_path = "/projects/katefgroup/datasets/UVO/vis_gt"
davis_palette = np.load('/home/wenhsuac/ovt/Detic/color_palette_davis.npy')

def mask2rgb(mask, palette, max_id):

    # Mask: [H, W]
    # palette: 256 * 3
    H, W = mask.shape
    rgb = np.zeros((H, W, 3))
    for i in range(max_id+1):
        rgb[mask==i] = palette[i]

    return rgb

class UVOTestDataset(Dataset):
    def __init__(self, data_root, load_backward=False):
        self.image_dir = path.join(data_root, 'uvo_videos_dense_frames')
        self.annFile = path.join(data_root, 'VideoDenseSet', 'UVO_video_val_dense_with_label.json')

        self.uvo = YTVOS(self.annFile)
        self.vid_ids = self.uvo.getVidIds()
        if load_backward:
            print("Loading backwards")
            self.vid_ids = reversed(self.vid_ids)

    def __getitem__(self, idx):
        vid = self.uvo.loadVids(self.vid_ids[idx])[0]
        annIds = self.uvo.getAnnIds(vidIds=vid['id'])
        anns = self.uvo.loadAnns(annIds)

        info = {}
        info['id'] = vid['id']
        info['name'] = vid['ytid']
        info['size'] = (vid['height'], vid['width']) # Real sizes

        images = []
        orig_images = []
        masks = []
        for i in range(len(anns)):
            inst_masks = []
            for j in range(len(anns[i]['segmentations'])):
                if anns[i]['segmentations'][j] is not None:
                    inst_masks.append(rle_decode(anns[i]['segmentations'][j]))
                else:
                    inst_masks.append(np.zeros((vid['height'], vid['width'])))
            inst_masks = np.stack(inst_masks, -1)
            masks.append(inst_masks)
        masks = np.stack(masks, 0)
        for i, f in enumerate(vid['file_names']):

            img = Image.open(path.join(self.image_dir, f)).convert('RGB')
            orig_images.append(torch.from_numpy(np.array(img)))
            images.append(np.array(img))

        orig_images = torch.stack(orig_images, 0)
        images = np.stack(images, 0)

        masks = torch.from_numpy(masks).permute(3,0,1,2).float() # [N, H, W, S] -> [S, N, H, W]
        #masks = torch.zeros(images.shape[0], 1, info['size'][0], info['size'][1])
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
    
test_dataset = UVOTestDataset("/projects/katefgroup/datasets/UVO/")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

for test_id, data in enumerate(test_loader):
    
    rgb = data['rgb'][0]
    msk = data['gt'][0]
    info = data['info']
    name = data['info']['name'][0]
    size = data['info']['size']

    os.makedirs(out_path, exist_ok=True)
    
    if os.path.exists(os.path.join(out_path, name+".gif")):
        continue

    gif_frames = []
    for f in range(rgb.shape[0]):

        merged_labels = torch.zeros((1, *size), dtype=torch.uint8)
        prob = msk[f]
        for prob_id in range(prob.shape[0]):
            mask = prob[prob_id] > 0.5
            merged_labels = mask * (mask * (prob_id+1)) + (~mask) * merged_labels
        
        img_E = Image.fromarray(mask2rgb(merged_labels[0], davis_palette, prob.shape[0]).astype(np.uint8))

        img_E = img_E.convert('RGBA')
        img_E.putalpha(127)
        img_E = img_E.resize((info['size'][1].item()//2, info['size'][0].item()//2))
        img_O = Image.fromarray(data['orig_rgb'][0][f].numpy().astype(np.uint8))
        img_O = img_O.resize((info['size'][1].item()//2, info['size'][0].item()//2))
        img_O.paste(img_E, (0, 0), img_E)

        gif_frames.append(img_O)

    gif_save_path = os.path.join(out_path, name+".gif")
    gif_frames[0].save(gif_save_path, format="GIF", append_images=gif_frames, save_all=True, interlace=False, duration=100, loop=0)

    del gif_frames