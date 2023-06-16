import numpy as np
import torch
import torch.nn.functional as F
import time
import os
import sys
import copy
from os import path
import argparse
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch.multiprocessing
import warnings

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

from progressbar import progressbar

sys.path.insert(0, '../../../dinov2')
sys.path.insert(0, '../../')
sys.path.insert(0, '../CenterNet2/')
sys.path.insert(0, '../gmflow/')
from gmflow.gmflow import GMFlow
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detectron2.config import get_cfg
from detectron2.modeling import build_model, GeneralizedRCNNWithTTA
from detic.custom_tta import CustomRCNNWithTTA
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer

#from STCN.model.eval_network import STCN
import hubconf

from dataset.davis_dataset import DAVISTestDataset
from dataset.davis_metrics import db_eval_boundary, db_eval_iou
from segment_anything import sam_model_registry, SamCustomPredictor
#from segment_anything_hq import sam_model_registry, SamCustomPredictor
#from sam_propagator import Propagator
from sam_propagator_local2 import Propagator
from ytvostools.mask import decode as rle_decode

CKPT_PATH = "/home/wenhsuac/ovt/Detic/third_party/SAM/pretrained/sam_vit_h_4b8939.pth"
#CKPT_PATH = "/home/wenhsuac/ovt/Detic/third_party/SAM/pretrained/sam_hq_vit_h.pth"

def setup_cfg(args):
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    print(args.config_file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = "/home/wenhsuac/ovt/Detic/datasets/metadata/lvis_v1_train_cat_info.json"
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if args.pred_one_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.TEST.AUG.FLIP = True
    cfg.TEST.AUG.MIN_SIZES = [int(cfg.INPUT.MIN_SIZE_TEST*0.75), cfg.INPUT.MIN_SIZE_TEST, int(cfg.INPUT.MIN_SIZE_TEST*1.25)]
    cfg.TEST.AUG.MAX_SIZE = cfg.INPUT.MAX_SIZE_TEST
    # cfg.vocabulary = args.vocabulary
    # if args.custom_vocabulary is not None:
    #     cfg.custom_vocabulary = args.custom_vocabulary
    cfg.freeze()
    return cfg

def get_unique_masks(predictions):

    raw_boxes = predictions["instances"].pred_boxes.tensor
    unique_masks = [predictions["instances"].pred_masks[0]]
    mask_labels = [predictions["instances"].pred_classes[0]]
    for idx in range(1, len(raw_boxes)):
        if not torch.allclose(raw_boxes[idx], raw_boxes[idx-1]):
            unique_masks.append(predictions["instances"].pred_masks[idx])
            mask_labels.append(predictions["instances"].pred_classes[idx])

    unique_masks = torch.stack(unique_masks)
    mask_labels = torch.stack(mask_labels)

    return unique_masks, mask_labels

def find_matching_masks(detected, gt, iou_thresh=0.0):

    # Detected: [N, 1, H, W]
    # GT: [N2, 1, H, W]
    #print(detected.shape)
    #print(gt.shape)

    # Convert to boolean masks
    detected = detected.bool().cpu() # [N1, 1, H, W]
    gt = gt.bool().permute(1,0,2,3).cpu() # [1, N2, H, W]

    intersection = torch.logical_and(detected, gt).float().sum((-2, -1))
    union = torch.logical_or(detected, gt).float().sum((-2, -1))
    
    iou = (intersection + 1e-6) / (union + 1e-6) # [N1, N2]
    #print(iou)
    #input()
    thresholded = iou # When we add semantic label thresholding
    row_idx, col_idx = linear_sum_assignment(-thresholded) # Score -> cost
    matched_idxes = np.stack([row_idx, col_idx], axis=1)
    matched_score = iou[row_idx, col_idx]
    matched_idxes = matched_idxes[torch.nonzero(matched_score>iou_thresh).flatten()] # [N, 2] or [2]
    # This happens when we only have one pair of masks matched
    # It makes subsequent functions so we unsqueeze for an extra dimension
    if len(matched_idxes.shape) == 1:
        matched_idxes = matched_idxes[None, :]

    return matched_idxes

def calc_iou(detected, gt):

    # Detected: [N, 1, H, W]
    # GT: [N, 1, H, W]
    assert (detected.shape == gt.shape)

    # Convert to boolean masks
    detected = detected.squeeze(1).bool().cpu() # [N, H, W]
    gt = gt.squeeze(1).bool().cpu() # [N, H, W]

    intersection = torch.logical_and(detected, gt).float().sum((-2, -1))
    union = torch.logical_or(detected, gt).float().sum((-2, -1))
    
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou

def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for k, l in enumerate(labels):
        Ms[k] = (masks == l).astype(np.uint8)
        
    return Ms

def mask2rgb(mask, palette, max_id):

    # Mask: [H, W]
    # palette: list([1]) * 768
    H, W = mask.shape
    palette = torch.stack(palette).reshape((256, 3))
    rgb = np.zeros((H, W, 3))
    for i in range(max_id+1):
        rgb[mask==i] = palette[i%255]

    return rgb

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
parser.add_argument(
    "--config-file",
    metavar="FILE",
    help="path to config file",
)
parser.add_argument(
    "--vocabulary",
    default="lvis",
    choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
    help="",
)
parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
)
parser.add_argument(
    "--confidence-threshold",
    type=float,
    default=0.5,
    help="Minimum score for instance predictions to be shown",
)
parser.add_argument(
    "--save-format",
    default="png",
    choices=['png', 'npz', 'None'],
    help="Save format for the detected masks",
)
parser.add_argument(
    "--pred_one_class",
    action='store_true',
)
parser.add_argument(
    "--opts",
    help="Modify config options using the command-line 'KEY VALUE' pairs",
    default=[],
    nargs=argparse.REMAINDER,
)
args = parser.parse_args()

davis_path = args.davis_path
out_path = args.output

# Simple setup
if args.output: os.makedirs(out_path, exist_ok=True)

torch.autograd.set_grad_enabled(False)
davis_palette = np.load('/home/wenhsuac/ovt/Detic/color_palette_davis.npy')

model_type = "vit_h_custom"

device = "cuda"

cfg = setup_cfg(args)

flow_predictor = GMFlow(feature_channels=128,
                        num_scales=2,
                        upsample_factor=4,
                        num_head=1,
                        attention_type='swin',
                        ffn_dim_expansion=4,
                        num_transformer_layers=6,
                        ).to(device)
flow_predictor.eval()

#dino = hubconf.dinov2_vits14().cuda().eval()

checkpoint = torch.load('/home/wenhsuac/ovt/Detic/third_party/gmflow/pretrained/gmflow_with_refine_sintel-3ed1cf48.pth')
weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
flow_predictor.load_state_dict(weights)

detector = build_model(cfg)
detector.eval()

checkpointer = DetectionCheckpointer(detector)
checkpointer.load(cfg.MODEL.WEIGHTS)

det_aug = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
)

tta_detector = GeneralizedRCNNWithTTA(cfg, detector)
#tta_detector = CustomRCNNWithTTA(cfg, detector)

sam = sam_model_registry[model_type](checkpoint=CKPT_PATH)
sam.to(device=device)
predictor = SamCustomPredictor(sam)

test_dataset = DAVISTestDataset(davis_path, resolution=(480,720), imset='2017/val.txt')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

total_process_time = 0
total_frames = 0
mious = []
j_metrics = np.array([])
f_metrics = np.array([])

for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    #torch.cuda.empty_cache()

    rgb = data['rgb'][0] # [B, S, H, W, C] -> [S, H, W, C]
    msk = data['gt'][0].to(sam.device)
    info = data['info']
    name = info['name'][0]
    k = msk.shape[0]
    size = info['shape']
    torch.cuda.synchronize()
    process_begin = time.time()
    print(name, rgb.shape)

    # Run the detector for frame 0
    img = rgb[0].numpy()
    height, width = img.shape[:2]
    image = det_aug.get_transform(img).apply_image(img)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": image, "height": height, "width": width}]

    with torch.no_grad():
        predictions = tta_detector(inputs)[0]
        if len(predictions["instances"].pred_masks) > 0:
            unique_masks, mask_labels = get_unique_masks(predictions)
        else:
            unique_masks = torch.zeros((0, height, width), device=device)
            mask_labels = []
        first_detection = unique_masks.unsqueeze(1) # [N, H, W] -> [N, 1, H, W]
        first_detection_labels = mask_labels

    # Find corresponding mask for each annotation in GT
    matched_idxes = find_matching_masks(msk[:, 0], first_detection, 0.0)
    first_detection = first_detection[matched_idxes[:, 1]]
    first_detection_labels = first_detection_labels[matched_idxes[:, 1]]

    '''detections = []
    detection_labels = []

    for img in rgb:
        img = img.numpy()
        height, width = img.shape[:2]
        image = det_aug.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]

        with torch.no_grad():
            predictions = tta_detector(inputs)[0]
            if len(predictions["instances"].pred_masks) > 0:
                unique_masks, mask_labels = get_unique_masks(predictions)
            else:
                unique_masks = torch.zeros((0, height, width), device=device)
                mask_labels = []
            detections.append(unique_masks.unsqueeze(0).unsqueeze(2)) # [N, H, W] -> [1, N, 1, H, W]
            detection_labels.append(mask_labels)
    
    torch.cuda.empty_cache()'''

    # Find corresponding mask for each annotation in GT
    '''matched_detections = []
    matched_idxes = find_matching_masks(msk[:, 0], detections[0][0], 0.0)
    matched_detections.append(detections[0][:, matched_idxes[:, 1]])
    matched_labels = detection_labels[0][matched_idxes[:, 1]]

    for i, (det, labels) in enumerate(zip(detections, detection_labels)):
        # Skip if first frame since we already have it
        if i > 0:
            # Get the matching detections based on labels
            matched_idxes = []
            for j in range(det.shape[1]):
                if labels[j] in matched_labels:
                    matched_idxes.append(j)
            matched_detection = detections[i][:, matched_idxes]
            matched_detections.append(matched_detection)

    first_detection = matched_detections[0][0]'''
    '''matched_detections = []
    for i, det in enumerate(detections):
        #print(i)
        matched_idxes = find_matching_masks(msk[:, i], detections[i][0], 0.0 if i == 0 else 0.75)
        matched_detection = detections[i][:, matched_idxes[:, 1]]
        if i == 0:
            matched_labels = detection_labels[i][matched_idxes[:, 1]]
        matched_detections.append(matched_detection)

    first_detection = matched_detections[0][0]'''
    # Run inference model
    if first_detection.shape[0] > 0:
        processor = Propagator(predictor, detector, flow_predictor, None, rgb, det_aug)
        with torch.no_grad():
            boxes = processor.interact(first_detection, first_detection_labels, 0, rgb.shape[0])

    # Postprocess predicted masks
    out_masks = torch.zeros((processor.t, *size), dtype=torch.uint8)
    for ti in range(processor.t):
        if len(processor.prob[ti]) > 0:
            prob = torch.from_numpy(rle_decode(processor.prob[ti])).float().permute(2,0,1).unsqueeze(1) # [H, W, N] -> [N, 1, H, W]
        else:
            prob = torch.zeros((1, 1, *size))
        prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
        num_instances = prob.shape[0]
        
        bg_mask = torch.ones((1,)+prob.shape[1:]) * 0.01
        prob = torch.cat([bg_mask, prob], dim=0)
        out_labels = torch.argmax(prob, dim=0)
        replaced_labels = out_labels.clone()
        #print("UNIQ1", ti, torch.unique(out_labels))
        for idx in range(num_instances):
            #print("REPLACING ", idx+1, "WITH", processor.valid_instances[ti][idx])
            replaced_labels[out_labels==idx+1] = processor.valid_instances[ti][idx]
        
        for i, (new, old) in enumerate(reversed(processor.reid_instance_mappings.items())):
            replaced_labels[replaced_labels==new] = old
 
        out_masks[ti] = replaced_labels[0]
    
    out_masks = out_masks.numpy().astype(np.uint8)

    #torch.cuda.synchronize()
    total_process_time += time.time() - process_begin
    total_frames += rgb.shape[0]

    # Calc stats
    out_labels = np.unique(out_masks[0])
    out_labels = out_labels[out_labels!=0]
    one_hot_detected = torch.from_numpy(all_to_onehot(out_masks, out_labels)).float().unsqueeze(2)
    if one_hot_detected.shape[0] < msk.shape[0]:
        zero_padding = torch.zeros((msk.shape[0] - one_hot_detected.shape[0], *one_hot_detected.shape[1:]))
        one_hot_detected = torch.cat([one_hot_detected, zero_padding], axis=0)
    matched_idxes = find_matching_masks(msk[:, 0], one_hot_detected[:, 0])
    '''if len(matched_idxes) > 0:
        for ti in range(1, rgb.shape[0]):
            miou = calc_iou(one_hot_detected[matched_idxes[:, 1], ti], msk[matched_idxes[:, 0], ti].cpu())
            mious.append(miou.sum()/msk.shape[0])'''
    # Need NSHW
    all_gt_masks = msk.cpu().numpy().squeeze(2) # [N, S, 1, H, W] -> [N, S, H, W]
    all_res_masks = one_hot_detected[matched_idxes[:, 1]].cpu().numpy().squeeze(2)
    if all_res_masks.shape[0] < all_gt_masks.shape[0]:
        print("SHAPE MISMATCH")
    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
    for ii in range(all_gt_masks.shape[0]):
        j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...])
        f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...])
    j_metrics = np.append(j_metrics, j_metrics_res.flatten())
    f_metrics = np.append(f_metrics, f_metrics_res.flatten())

    # Save the results
    if args.output:
        this_out_path = path.join(out_path, "pred", name)
        this_out_path2 = path.join(out_path, "vis", name)
        this_out_path3 = path.join(out_path, "prompt", name)
        os.makedirs(this_out_path, exist_ok=True)
        os.makedirs(this_out_path2, exist_ok=True)
        os.makedirs(this_out_path3, exist_ok=True)
        gif_frames = []
        for f in range(rgb.shape[0]):
            
            img_E = Image.fromarray(mask2rgb(out_masks[f], info['palette'], torch.max(processor.valid_instances[f])).astype(np.uint8))
            #img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))
            
            gt_with_bg = torch.cat([torch.ones_like(data['gt'][0][:1, f])*0.01, data['gt'][0][:, f]], dim=0)
            gt_mask = torch.argmax(gt_with_bg, dim=0).squeeze(0)
            img_GT = Image.fromarray(gt_mask.cpu().numpy().astype(np.uint8), mode='P')
            img_GT.putpalette(info['palette'])

            img_E = img_E.convert('RGBA')
            img_E.putalpha(127)
            img_GT = img_GT.convert('RGBA')
            img_GT.putalpha(127)

            img_O = Image.fromarray(data['orig_rgb'][0][f].numpy().astype(np.uint8))
            img_O = img_O.resize((size[1], size[0]))
            img_O2 = img_O.copy()
            img_O.paste(img_E, (0, 0), img_E)
            img_O2.paste(img_GT, (0, 0), img_GT)
            #img_O.save(os.path.join(this_out_path2, '{:05d}.png'.format(f)))

            if f > 0:
                if len(boxes[f-1]) > 0:
                    draw = ImageDraw.Draw(img_O)
                    for obj_id, box in reversed(list(enumerate(boxes[f-1]))):
                        inst_id = processor.valid_instances[f][obj_id].item()
                        while(inst_id in processor.reid_instance_mappings.keys()):
                            inst_id = processor.reid_instance_mappings[inst_id]
                        if inst_id != -1:
                            try:
                                draw.rectangle(box.tolist(), outline=tuple(davis_palette[inst_id%255]), width=2)
                            except ValueError:
                                # Sometimes the box is invalid, just move along and hope for the best.
                                print("Warning: Invalid box")
                                pass

            '''if f+1 < rgb.shape[0]:
                #print(name, processor.valid_instances[f+1], type(processor.valid_instances[f+1]))
                img_prompt = Image.fromarray(data['orig_rgb'][0][f+1].numpy().astype(np.uint8))
                img_prompt = img_prompt.resize((size[1], size[0]))
                img_E_next = img_prompt.copy()
                img_E_next = Image.fromarray(mask2rgb(out_masks[f+1], info['palette'], torch.max(processor.valid_instances[f+1])).astype(np.uint8))
                img_E_next = img_E_next.convert('RGBA')
                img_E_next.putalpha(127)
                img_O_next = Image.fromarray(data['orig_rgb'][0][f+1].numpy().astype(np.uint8))
                img_O_next = img_O_next.resize((size[1], size[0]))
                img_O_next.paste(img_E_next, (0, 0), img_E_next)
                if len(boxes[f]) > 0:
                    draw = ImageDraw.Draw(img_prompt)
                    for obj_id, box in enumerate(boxes[f]):
                        if processor.valid_instances[f+1][obj_id].item() in processor.reid_instance_mappings:
                            color = processor.reid_instance_mappings[processor.valid_instances[f+1][obj_id].item()]
                        else:
                            color = processor.valid_instances[f+1][obj_id]
                        draw.rectangle(box.tolist(), outline=tuple(davis_palette[color]), width=3)
                merged_prompt = Image.new('RGB', (img_O.width + img_prompt.width + img_O_next.width, img_O.height))
                merged_prompt.paste(img_O, (0, 0))
                merged_prompt.paste(img_prompt, (img_O.width, 0))
                merged_prompt.paste(img_O_next, (img_O.width+img_prompt.width, 0))
                #merged_prompt.save(os.path.join(this_out_path3, '{:05d}.png'.format(f)))'''

            merged = Image.new('RGB', (img_O.width + img_O2.width, img_O.height))
            merged.paste(img_O, (0, 0))
            merged.paste(img_O2, (img_O.width, 0))
            gif_frames.append(merged)

        gif_save_path = os.path.join(this_out_path2, "vis.gif")
        gif_frames[0].save(gif_save_path, format="GIF", append_images=gif_frames, save_all=True, interlace=False, duration=100, loop=0)

        del gif_frames

    del rgb
    del predictions
    del msk

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)
#print('mIOU: ', torch.stack(mious).mean())
print(np.mean(j_metrics), np.mean(f_metrics))
