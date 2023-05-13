import numpy as np
import torch
import torch.nn.functional as F
import time
import os
import sys
import json
from os import path
import argparse
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True

from progressbar import progressbar

sys.path.insert(0, '../../')
sys.path.insert(0, '../CenterNet2/')
sys.path.insert(0, '../gmflow/')
from gmflow.gmflow import GMFlow
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detectron2.config import get_cfg
from detectron2.modeling import build_model, GeneralizedRCNNWithTTA
import detectron2.data.transforms as T
from detectron2.structures import Instances, Boxes
from detectron2.checkpoint import DetectionCheckpointer

from dataset.yv_dataset import YouTubeVOSTestDataset
from utils.flow import run_flow_on_images
from segment_anything import sam_model_registry, SamCustomPredictor
from sam_propagator_yv import Propagator

CKPT_PATH = "/home/wenhsuac/ovt/Detic/third_party/SAM/pretrained/sam_vit_h_4b8939.pth"

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
        rgb[mask==i] = palette[i]

    return rgb

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--yv_path', default='/projects/katefgroup/datasets/OVT-Youtube')
parser.add_argument('--output')
parser.add_argument('--load_backward', action='store_true')
parser.add_argument('--split', default='valid')
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

yv_path = args.yv_path
out_path = args.output

# Simple setup
if args.output: os.makedirs(out_path, exist_ok=True)

torch.autograd.set_grad_enabled(False)
palette = Image.open(path.expanduser(args.yv_path + '/valid/Annotations/0a49f5265b/00000.png')).getpalette()

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

sam = sam_model_registry[model_type](checkpoint=CKPT_PATH)
sam.to(device=device)
predictor = SamCustomPredictor(sam)

test_dataset = YouTubeVOSTestDataset(yv_path, split='valid', load_backward=args.load_backward)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

total_process_time = 0
total_frames = 0

with open(path.join(yv_path, args.split, 'meta.json')) as f:
    meta = json.load(f)['videos']

for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True):

    # Just skip if we already have what we need
    if os.path.exists(path.join(out_path, 'vis', data['info']['name'][0])):
        continue

    #torch.cuda.empty_cache()

    rgb = data['rgb'][0] # [B, S, H, W, C] -> [S, H, W, C]
    msk = data['gt'][0].to(sam.device)
    info = data['info']
    name = info['name'][0]
    num_objects = len(info['labels'][0])
    gt_obj = info['gt_obj']
    size = info['shape']
    #torch.cuda.synchronize()
    process_begin = time.time()
    print(name, rgb.shape)

    #if name == 'c16d9a4ade' or name == '6eaf926e75':
    #    continue

    detections = []
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
    
    fwd_flow, bwd_flow = run_flow_on_images(flow_predictor, rgb)
    fwd_flow = torch.from_numpy(fwd_flow)
    bwd_flow = torch.from_numpy(bwd_flow)
    
    #torch.cuda.empty_cache()

    # Frames with labels, but they are not exhaustively labeled
    frames_with_gt = sorted(list(gt_obj.keys()))
    print("FRAMES WITH GT", frames_with_gt)
    # Find corresponding mask for each annotation in GT
    gt_matched_detections = []
    matched_detections_label = []
    for frame_with_gt in frames_with_gt:
        matched_idxes = find_matching_masks(msk[:, frame_with_gt], detections[frame_with_gt][0], 0.0)
        # Check if we actually matched something (sometimes there are no detections)
        if len(matched_idxes) > 0:
            # Sometimes we don't have enough detections to match, so we don't get enough indexes
            # In this case we'll pad with some dummy values
            num_obj_in_gt = len(msk[:, frame_with_gt])
            num_matched_obj = len(matched_idxes)
            print(frame_with_gt, num_obj_in_gt, num_matched_obj)
            while num_matched_obj < num_obj_in_gt:
                padded_idxes = np.array([[num_matched_obj, matched_idxes[0, 1]]])
                matched_idxes = np.concatenate([matched_idxes, padded_idxes], axis=0)
                num_matched_obj += 1
            # Index into the detections
            matched_detection = detections[frame_with_gt][:, matched_idxes[:, 1]]
            gt_matched_detections.append(matched_detection)
            matched_labels = detection_labels[frame_with_gt][matched_idxes[:, 1]]
        else:
            # If nothing is matched, we just have no detections, so just append everything
            gt_matched_detections.append(detections[frame_with_gt]) # [1, 0, 1, H, W]
            matched_labels = torch.tensor([], device=device)

        if len(matched_detections_label) == 0:
            matched_detections_label.append({"t": frame_with_gt,
                                            "labels": matched_labels})
        else:
            matched_detections_label.append({"t": frame_with_gt,
                                            "labels": torch.cat([matched_detections_label[-1]["labels"],
                                                                matched_labels])})
    #print("MATCHED DET LABELS", matched_detections_label)
    # Store detections with the same class labels
    matched_detections = []
    curr_frame_with_gt_idx = 0
    for i, (det, labels) in enumerate(zip(detections, detection_labels)):
        # Skip if we don't even need to match anything yet
        if i > matched_detections_label[curr_frame_with_gt_idx]["t"]:
            # Check if we need to move onto the next set of matched labels
            if curr_frame_with_gt_idx < len(frames_with_gt) - 1:
                if i >= frames_with_gt[curr_frame_with_gt_idx+1]:
                    curr_frame_with_gt_idx += 1

            # Get the matching detections based on labels
            matched_labels = matched_detections_label[curr_frame_with_gt_idx]["labels"]
            matched_idxes = []
            for j in range(det.shape[1]):
                if labels[j] in matched_labels:
                    matched_idxes.append(j)
            matched_detection = detections[i][:, matched_idxes]
            matched_detections.append(matched_detection)
        else:
            matched_detections.append([])
    # Replace detections in GT frames with the original matched detections
    for i, frame_with_gt in enumerate(frames_with_gt):
        matched_detections[frame_with_gt] = gt_matched_detections[i]
    
    # Run inference model
    processor = Propagator(predictor, detector, rgb, num_objects, det_aug)
    #print("NUM OBJ", num_objects)
    with torch.no_grad():
        # min_idx tells us the starting point of propagation
        # Propagating before there are labels is not useful
        min_idx = 99999
        for i, frame_idx in enumerate(frames_with_gt):
            min_idx = min(frame_idx, min_idx)
            # Note that there might be more than one label per frame
            obj_idx = gt_obj[frame_idx][0].tolist()
            # Map the possibly non-continuous labels into a continuous scheme
            obj_idx = [info['label_convert'][o].item() for o in obj_idx]

            start_detection = matched_detections[frame_idx][0]
            print(start_detection.shape, len(obj_idx))
            if start_detection.shape[0] > 0:
                # We perform propagation from the current frame to the next frame with label
                start_detection = start_detection[[x - 1 for x in obj_idx]]
                if i == len(frames_with_gt) - 1:
                    boxes = processor.interact(start_detection, frame_idx, rgb.shape[0], torch.tensor(obj_idx, device=processor.device),
                                            fwd_flow, bwd_flow, matched_detections)
                else:
                    boxes = processor.interact(start_detection, frame_idx, frames_with_gt[i+1]+1, torch.tensor(obj_idx, device=processor.device),
                                            fwd_flow, bwd_flow, matched_detections)

    # Postprocess predicted masks
    out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
    for ti in range(processor.t):
        prob = processor.prob[ti].float()
        prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)

        out_labels = torch.zeros((1, *size), dtype=torch.uint8, device='cuda')
        labels_in_gt = np.arange(1, num_objects+1)
        #print(ti, "VALID", processor.valid_instances[ti])
        # Swap Re-ID labels, do this in a backwards manner as that's how things get linked
        for i, label in enumerate(reversed(processor.valid_instances[ti])):
            if label.item() in processor.reid_instance_mappings.keys():
                new = label
                new_prob_id = processor.valid_instances[ti].index(new)
                old = processor.reid_instance_mappings[new.item()]
                # If both old and new are valid, we need to merge labels
                if torch.tensor(old, device='cuda') in processor.valid_instances[ti]:
                    old_prob_id = processor.valid_instances[ti].index(torch.tensor(old, device='cuda'))
                    # Hack: if both ids are in GT, we probably made a mistake, so skip it
                    if new.item() in labels_in_gt and old in labels_in_gt:
                        continue
                    # This mapping goes both ways, we need to reverse the mapping if needed so the GT labels don't get replaced
                    if new.item() in labels_in_gt:
                        #print(ti, "INSTEAD REID REPLACING ", old, "WITH", new.item())
                        prob[new_prob_id] = torch.clamp(prob[new_prob_id] + prob[old_prob_id], max=1.0)
                    elif old in labels_in_gt:
                        #print(ti, "REID REPLACING ", new.item(), "WITH", old)
                        prob[old_prob_id] = torch.clamp(prob[new_prob_id] + prob[old_prob_id], max=1.0)
                # Otherwise we can just replace labels by replacing entries in valid_instances
                else:
                    # Keep replacing until we hit a GT label
                    while(processor.valid_instances[ti][new_prob_id].item() in processor.reid_instance_mappings.keys()):
                        new = processor.valid_instances[ti][new_prob_id]
                        old = processor.reid_instance_mappings[new.item()]
                        # Make sure we're not replacing GT labels
                        if new.item() not in labels_in_gt:
                            #print(ti, "SIMPLE REID REPLACING ", new.item(), "WITH", old)
                            processor.valid_instances[ti][new_prob_id] = torch.tensor(old, device='cuda')
                        else:
                            break

        # Merge instance masks
        for prob_id, label in enumerate(processor.valid_instances[ti]):
            if label.item() in labels_in_gt:
                mask = prob[prob_id] > 0.5
                out_labels = mask * (mask * label) + (~mask) * out_labels
 
        out_masks[ti] = out_labels
    
    out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

    #torch.cuda.synchronize()
    total_process_time += time.time() - process_begin
    total_frames += rgb.shape[0]
    
    # Remap the indices to the original domain
    #print("LABEL BACK", info['label_backward'])
    idx_masks = np.zeros_like(out_masks)
    for i in range(1, num_objects+1):
        backward_idx = info['label_backward'][i].item()
        idx_masks[out_masks==i] = backward_idx
    
    # Save the results
    # Get the required set of frames
    req_frames = []
    objects = meta[name]['objects']
    for key, value in objects.items():
        req_frames.extend(value['frames'])

    # Map the frame names to indices
    req_frames_names = set(req_frames)
    req_frames = []
    for fi in range(rgb.shape[0]):
        frame_name = info['frames'][fi][0][:-4]
        if frame_name in req_frames_names:
            req_frames.append(fi)
    req_frames = sorted(req_frames)

    this_out_path = path.join(out_path, 'Annotations_Debug', name)
    this_out_path2 = path.join(out_path, 'vis', name)
    os.makedirs(this_out_path, exist_ok=True)
    os.makedirs(this_out_path2, exist_ok=True)
    gif_frames = []
    for f in range(idx_masks.shape[0]):
        if f >= min_idx:
            img_E = Image.fromarray(idx_masks[f])
            img_E.putpalette(palette)
            if f in req_frames:
                img_E.save(os.path.join(this_out_path, info['frames'][f][0].replace('.jpg','.png')))

            img_E = img_E.convert('RGBA')
            img_E.putalpha(127)
            img_E = img_E.resize((360, 240))
            img_O = Image.fromarray(data['orig_rgb'][0][f].numpy().astype(np.uint8))
            img_O = img_O.resize((360, 240))
            img_O.paste(img_E, (0, 0), img_E)

            gif_frames.append(img_O)

    gif_save_path = os.path.join(this_out_path2, "vis.gif")
    gif_frames[0].save(gif_save_path, format="GIF", append_images=gif_frames, save_all=True, interlace=False, duration=100, loop=0)

    del gif_frames

    del rgb
    del fwd_flow
    del bwd_flow
    del predictions
    del processor
    del detections
    del msk

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)
