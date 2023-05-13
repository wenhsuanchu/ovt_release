import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from progressbar import progressbar

from davis_test_dataset_npz import DAVISTestDataset
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

CKPT_PATH = "/home/wenhsuac/ovt/Detic/third_party/SAM/pretrained/sam_vit_h_4b8939.pth"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--davis_path', default='/projects/katefgroup/datasets/OVT-DAVIS')
parser.add_argument('--output')
parser.add_argument('--split', help='val/testdev', default='val')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()

davis_path = args.davis_path
out_path = args.output

# Simple setup
if args.output: os.makedirs(out_path, exist_ok=True)

# Simple setup
#if args.output: os.makedirs(out_path, exist_ok=True)

torch.autograd.set_grad_enabled(False)

model_type = "vit_h"

device = "cuda"

sam = sam_model_registry["vit_h"](checkpoint=CKPT_PATH)
sam.to(device=device)
predictor = SamPredictor(sam)

test_dataset = DAVISTestDataset(davis_path, resolution=(480,720), imset='2017/debug.txt')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    torch.cuda.empty_cache()

    rgb = data['rgb'] # [B, S, C, H, W]
    detections = data['detections']
    detected_boxes = data['detected_boxes'] # list([B, N, 4])
    msk = data['gt'][0].to(sam.device)
    info = data['info']
    name = info['name'][0]
    k = min(msk.shape[0], detections[0].shape[1])
    size = info['shape']
    torch.cuda.synchronize()
    process_begin = time.time()

    #rgb = np.array(Image.open("truck.jpg").convert('RGB'))
    #print(rgb.shape)
    '''im_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    rgb = im_transform(rgb).permute(1,2,0)'''
    '''input_boxes = torch.tensor([
        [75, 275, 1725, 850],
        [425, 600, 700, 875],
        [1375, 550, 1650, 800],
        [1240, 675, 1400, 750],
    ], device=predictor.device)'''

    print(rgb.shape)
    print(data['detected_boxes'][0].shape)

    predictor.set_image(rgb[0, 0].numpy().astype(np.uint8))
    #predictor.set_image(rgb.astype(np.uint8))
    transformed_boxes = predictor.transform.apply_boxes_torch(detected_boxes[0][0][:10].to(sam.device), rgb[0][0].shape[:2])
    #transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes.to(sam.device), rgb.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    print(detected_boxes[0][0][:10])
    print(masks.shape)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(rgb[0, 0].cpu().numpy())
    #plt.imshow(rgb)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in detected_boxes[0][0][:10]:
    #for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    plt.axis('off')
    fig.savefig('temp.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.0)
