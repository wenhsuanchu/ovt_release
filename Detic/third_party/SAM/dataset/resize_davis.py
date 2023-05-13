import numpy as np
import os
from os import path
from PIL import Image

from progressbar import progressbar

root = "/projects/katefgroup/datasets/OVT-DAVIS"
imset = "2017/debug.txt"
resolution = (480, 720)

mask_dir = path.join(root, 'Annotations')
image_dir = path.join(root, 'JPEGImages')
imset_dir = path.join(root, 'ImageSets')
imset_f = path.join(imset_dir, imset)

out_image_dir = path.join(root, 'JPEGImages_480')
out_mask_dir = path.join(root, 'Annotations_480')

videos = []
num_frames = {}
with open(path.join(imset_f), "r") as lines:
    for line in lines:
        _video = line.rstrip('\n')
        videos.append(_video)
        num_frames[_video] = len(os.listdir(path.join(image_dir, _video)))

for video in progressbar(videos, max_value=len(videos), redirect_stdout=True):

    os.makedirs(path.join(out_image_dir, video), exist_ok=True)
    os.makedirs(path.join(out_mask_dir, video), exist_ok=True)

    for f in range(num_frames[video]):
        img_file = path.join(image_dir, video, '{:05d}.jpg'.format(f))
        img = Image.open(img_file).convert('RGB')
        img = img.resize((resolution[1], resolution[0]))
        img.save(path.join(out_image_dir, video, '{:05d}.jpg'.format(f)))

        mask_file = path.join(mask_dir, video, '{:05d}.png'.format(f))
        if path.exists(mask_file):
            mask = Image.open(mask_file).convert("RGB")
            mask = mask.resize((resolution[1], resolution[0]), Image.NEAREST)
            if f == 0:
                first_mask = mask
        else:
            # Test-set maybe?
            mask = Image.fromarrayy(np.zeros_like(first_mask))
        mask.save(path.join(out_mask_dir, video, '{:05d}.png'.format(f)))
        #print(np.unique(np.array(mask).reshape(-1, 3), axis=0))