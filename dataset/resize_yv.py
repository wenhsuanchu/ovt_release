import numpy as np
import os
from os import path
from PIL import Image

from progressbar import progressbar

root = "/projects/katefgroup/datasets/OVT-Youtube/"
resolution = (480, 720)

mask_dir = path.join(root, 'valid', 'Annotations')
image_dir = path.join(root, 'all_frames', 'valid_all_frames', 'JPEGImages')

out_image_dir = path.join(root, 'all_frames', 'valid_all_frames', 'JPEGImages_480')
out_mask_dir = path.join(root, 'valid', 'Annotations_480')

videos = []
num_frames = {}

'''videos = sorted(os.listdir(image_dir))

for video in progressbar(videos, max_value=len(videos), redirect_stdout=True):

    os.makedirs(path.join(out_image_dir, video), exist_ok=True)

    frames = sorted(os.listdir(os.path.join(image_dir, video)))

    for f, frame_name in enumerate(frames):
        img_file = path.join(image_dir, video, frame_name)
        img = Image.open(img_file).convert('RGB')
        img = img.resize((resolution[1], resolution[0]))
        img.save(path.join(out_image_dir, video, frame_name))'''

videos = sorted(os.listdir(mask_dir))

for video in progressbar(videos, max_value=len(videos), redirect_stdout=True):

    if video != "119739ba0b":
        continue
        
    os.makedirs(path.join(out_mask_dir, video), exist_ok=True)

    frames = sorted(os.listdir(os.path.join(mask_dir, video)))

    for f, frame_name in enumerate(frames):
        mask_file = path.join(mask_dir, video, frame_name)
        
        if path.exists(mask_file):
            mask = Image.open(mask_file).convert('P')
            mask = mask.resize((resolution[1], resolution[0]), Image.NEAREST)

            mask.save(path.join(out_mask_dir, video, frame_name))