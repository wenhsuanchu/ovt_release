import numpy as np
import sys
import torch
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, '../GroundingDINO/')
import matplotlib.pyplot as plt
import time
import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from scipy.optimize import linear_sum_assignment
import groundingdino.datasets.transforms as T
import torch.multiprocessing
from groundingdino.util.inference import load_model, load_image, predict, annotate
from torchvision.ops import box_convert
torch.multiprocessing.set_sharing_strategy('file_system')

from progressbar import progressbar

from dataset.davis_dataset import DAVISTestDataset
from segment_anything import sam_model_registry, SamPredictor, SamCustomPredictor

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

gdino = load_model("/home/wenhsuac/ovt/Detic/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                   "/home/wenhsuac/ovt/Detic/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth")

sam = sam_model_registry["vit_h"](checkpoint=CKPT_PATH)
sam.to(device=device)
predictor = SamPredictor(sam)

test_dataset = DAVISTestDataset(davis_path, resolution=(480,720), imset='2017/debug2.txt')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    torch.cuda.empty_cache()

    rgb = data['rgb'] # [B, S, C, H, W]
    msk = data['gt'][0].to(sam.device)
    info = data['info']
    name = info['name'][0]
    size = info['shape']
    torch.cuda.synchronize()
    process_begin = time.time()

    #rgb = np.array(Image.open("truck.jpg").convert('RGB'))
    #print(rgb.shape)

    img = rgb[0, 0].numpy()
    #height, width = img.shape[:2]
    gdino_transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img_transformed, _ = gdino_transform(Image.fromarray(rgb[0, 0].numpy()), None)

    boxes, logits, phrases = predict(
            model=gdino,
            image=img_transformed,
            caption="dog most to the right",
            box_threshold=0.35,
            text_threshold=0.25
        )

    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xyxy[:, ::2] = xyxy[:, ::2] * img.shape[1]
    xyxy[:, 1::2] = xyxy[:, 1::2] * img.shape[0]

    predictor.set_image(rgb[0, 0].numpy().astype(np.uint8))
    #predictor.set_image(rgb.astype(np.uint8))
    transformed_boxes = predictor.transform.apply_boxes_torch(torch.from_numpy(xyxy).to(sam.device), rgb[0][0].shape[:2])
    #transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes.to(sam.device), rgb.shape[:2])

    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    print(xyxy)
    print(masks.shape)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(rgb[0, 0].cpu().numpy())
    #plt.imshow(rgb)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in xyxy:
    #for box in input_boxes:
        show_box(box, plt.gca())
    plt.axis('off')
    fig.savefig('temp.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.0)

    assert(False)
