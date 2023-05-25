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
import warnings

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

from progressbar import progressbar

sys.path.insert(0, '../../')
sys.path.insert(0, '../CenterNet2/')
sys.path.insert(0, '../GroundingDINO/')
sys.path.insert(0, '../gmflow/')
from gmflow.gmflow import GMFlow
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detectron2.config import get_cfg
from detectron2.modeling import build_model, GeneralizedRCNNWithTTA
from detic.custom_tta import CustomRCNNWithTTA
import detectron2.data.transforms as T
import groundingdino.datasets.transforms as GDT
from detectron2.structures import Instances, Boxes
from detectron2.checkpoint import DetectionCheckpointer

from groundingdino.util.inference import load_model, load_image, predict, annotate
from torchvision.ops import box_convert

from dataset.burst_dataset import BURSTTestDataset
from dataset.davis_dataset import DAVISTestDataset
from utils.flow import run_flow_on_images
from segment_anything import sam_model_registry, SamCustomPredictor
from sam_propagator_grounded import Propagator
from ytvostools.mask import encode as rle_encode

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
    cfg.TEST.AUG.MIN_SIZES = [int(cfg.INPUT.MIN_SIZE_TEST*0.5), int(cfg.INPUT.MIN_SIZE_TEST*0.75), cfg.INPUT.MIN_SIZE_TEST, int(cfg.INPUT.MIN_SIZE_TEST*1.25), int(cfg.INPUT.MIN_SIZE_TEST*1.5)]
    cfg.TEST.AUG.MAX_SIZE = cfg.INPUT.MAX_SIZE_TEST
    cfg.vocabulary = args.vocabulary
    if args.custom_vocabulary is not None:
        cfg.custom_vocabulary = args.custom_vocabulary
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

def mask2rgb(mask, palette, max_id):

    # Mask: [H, W]
    # palette: list([1]) * 768

    H, W = mask.shape
    rgb = np.zeros((H, W, 3))
    alpha = np.zeros((H, W, 1))
    for i in range(max_id+1):
        if i ==0:
            rgb[mask==i] = palette[i%255]
            alpha[mask==i] = 0
            '''elif i == 1:
                rgb[mask==i] = np.array([255,0,212])
                alpha[mask==i] = 200'''
        else:
            rgb[mask==i] = palette[i%255]
            alpha[mask==i] = 200
    
    rgb = np.concatenate([rgb, alpha], axis=-1)

    return rgb

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--dataset_path', default='/projects/katefgroup/datasets/BURST/')
parser.add_argument('--output')
parser.add_argument('--load_backward', action='store_true')
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
        "--caption",
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

dataset_path = args.dataset_path
out_path = args.output
caption = args.caption

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

checkpoint = torch.load('/home/wenhsuac/ovt/Detic/third_party/gmflow/pretrained/gmflow_with_refine_sintel-3ed1cf48.pth')
weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
flow_predictor.load_state_dict(weights)

gdino = load_model("/home/wenhsuac/ovt/Detic/third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                   "/home/wenhsuac/ovt/Detic/third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth")

detector = build_model(cfg)
detector.eval()

checkpointer = DetectionCheckpointer(detector)
checkpointer.load(cfg.MODEL.WEIGHTS)

det_aug = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
)

#tta_detector = GeneralizedRCNNWithTTA(cfg, detector)
tta_detector = CustomRCNNWithTTA(cfg, detector)

sam = sam_model_registry[model_type](checkpoint=CKPT_PATH)
sam.to(device=device)
predictor = SamCustomPredictor(sam)

#test_dataset = BURSTTestDataset(os.path.join(dataset_path, 'val', 'all_classes.json'), os.path.join(burst_path, 'frames'), annotated_only=False)
test_dataset = DAVISTestDataset(dataset_path, resolution=(480,720), imset='2017/debug.txt')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

total_process_time = 0
total_frames = 0

for test_id, data in enumerate(progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True)):

    #torch.cuda.empty_cache()
    if test_id > 20: break

    '''rgb = data['rgb'][0] # [B, S, H, W, C] -> [S, H, W, C]
    msk = data['gt'][0].to(sam.device)
    info = data['info']
    name = info['name'][0]
    vid_id = info['id']
    size = info['size']'''
    # DAVIS
    rgb = data['rgb'][0] # [B, S, H, W, C] -> [S, H, W, C]
    msk = data['gt'][0].to(sam.device)
    info = data['info']
    name = info['name'][0]
    size = info['shape']
    info['size'] = info['shape']
    #torch.cuda.synchronize()
    process_begin = time.time()
    #print(name, vid_id, size, rgb.shape)

    detections = []
    detection_labels = []

    # Run the detector for frame 0
    img = rgb[0].numpy()
    height, width = img.shape[:2]
    gdino_transform = GDT.Compose(
        [
            GDT.RandomResize([800], max_size=1333),
            GDT.ToTensor(),
            GDT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img_transformed, _ = gdino_transform(Image.fromarray(img), None)

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=gdino,
            image=img_transformed,
            caption=caption,
            box_threshold=0.35,
            text_threshold=0.25
        )

        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        xyxy[:, ::2] = xyxy[:, ::2] * img.shape[1]
        xyxy[:, 1::2] = xyxy[:, 1::2] * img.shape[0]

        predictor.set_image(img.astype(np.uint8))
        transformed_boxes = predictor.transform.apply_boxes_torch(torch.from_numpy(xyxy).to(sam.device), img.shape[:2])

        masks, _, _, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        first_detection = masks # [N, 1, H, W]
    
    #torch.cuda.empty_cache()

    # Run inference model
    if first_detection.shape[0] > 0:
        processor = Propagator(predictor, detector, gdino, flow_predictor, rgb, 1, det_aug, caption)
        with torch.no_grad():
            boxes = processor.interact(first_detection, 0, rgb.shape[0])

    # Postprocess predicted masks
    merged_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
    out_masks = []
    for ti in range(processor.t):
        prob = processor.prob[ti].float()
        prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)

        merged_labels = torch.zeros((1, *size), dtype=torch.uint8, device='cuda')

        # Swap Re-ID labels, do this in a backwards manner as that's how things get linked
        for i, label in enumerate(reversed(processor.valid_instances[ti])):
            if label.item() in processor.reid_instance_mappings.keys():
                new_prob_id = processor.valid_instances[ti].tolist().index(label)
                # Keep replacing until we hit the end
                while(processor.valid_instances[ti][new_prob_id].item() in processor.reid_instance_mappings.keys()):
                    new = processor.valid_instances[ti][new_prob_id]
                    old = processor.reid_instance_mappings[new.item()]
                    # If both old and new are valid, we need to merge labels
                    # Then we can break the loop and handle the rest later
                    if torch.tensor(old, device='cuda') in processor.valid_instances[ti]:
                        old_prob_id = processor.valid_instances[ti].tolist().index(torch.tensor(old, device='cuda'))
                        prob[old_prob_id] = torch.clamp(prob[new_prob_id] + prob[old_prob_id], max=1.0)
                        # Mark invalid because we have the old instance already
                        processor.valid_instances[ti][new_prob_id] = -1
                        break
                    # Otherwise just replace the valid instance ID
                    else:
                        processor.valid_instances[ti][new_prob_id] = torch.tensor(old, device='cuda')
        
                '''# If both old and new are valid, we need to merge labels
                if torch.tensor(old, device='cuda') in processor.valid_instances[ti]:
                    old_prob_id = processor.valid_instances[ti].tolist().index(torch.tensor(old, device='cuda'))
                    prob[old_prob_id] = torch.clamp(prob[new_prob_id] + prob[old_prob_id], max=1.0)
                # Otherwise we can just replace labels by replacing entries in valid_instances
                else:
                    # Keep replacing until we hit a GT label
                    while(processor.valid_instances[ti][new_prob_id].item() in processor.reid_instance_mappings.keys()):
                        new = processor.valid_instances[ti][new_prob_id]
                        old = processor.reid_instance_mappings[new.item()]
                        # Make sure we're not replacing GT labels
                        processor.valid_instances[ti][new_prob_id] = torch.tensor(old, device='cuda')'''

        # Merge instance masks
        # Make sure we actually have valid instances
        if not torch.equal(processor.valid_instances[ti], torch.tensor([0], device='cuda')):
            for prob_id, label in reversed(list(enumerate(processor.valid_instances[ti]))):
                if label == -1:
                    continue
                mask = prob[prob_id] > 0.5
                merged_labels = mask * (mask * label) + (~mask) * merged_labels
 
        merged_masks[ti] = merged_labels
        out_masks.append(prob)
    
    merged_masks = (merged_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

    #torch.cuda.synchronize()
    total_process_time += time.time() - process_begin
    total_frames += rgb.shape[0]

    # Generate output json
    # We will generate one json per video so we can parallelize
    '''results = []
    for inst_id in range(1, processor.total_instances):
        instance_masks = []
        for ti, out_mask in enumerate(out_masks):
            if inst_id in processor.valid_instances[ti]:
                mask_idx = processor.valid_instances[ti].tolist().index(inst_id)
                seg = rle_encode(np.asfortranarray(out_mask[mask_idx][0].detach().cpu().numpy() > 0.5).astype(np.uint8))
                seg['counts'] = seg['counts'].decode()
                instance_masks.append(seg)
            else:
                instance_masks.append(None)
        instance_dict = {
            'video_id': vid_id.item(),
            'score': 0.9, # We should replace this with detection scores at some point
            'category_id': 1,
            'segmentations': instance_masks
        }
        results.append(instance_dict)
    #print(results)
    results_json = json.dumps(results)
    if args.output:
        json_save_path = path.join(out_path, "json")
        os.makedirs(json_save_path, exist_ok=True)
        with open(path.join(json_save_path, "{:04d}.json".format(vid_id.item())), 'w') as f:
            f.write(results_json)'''

    # Save the results
    if args.output:

        this_out_path = path.join(out_path, "pred", name)
        this_out_path2 = path.join(out_path, "vis")
        this_out_path3 = path.join(out_path, "prompt", name)
        os.makedirs(this_out_path, exist_ok=True)
        os.makedirs(this_out_path2, exist_ok=True)
        os.makedirs(this_out_path3, exist_ok=True)
        gif_frames = []
        for f in range(rgb.shape[0]):
            
            img_E = Image.fromarray(mask2rgb(merged_masks[f], davis_palette, torch.max(processor.valid_instances[f])).astype(np.uint8), mode='RGBA')
            #img_E.save(os.path.join(this_out_path, '{:05d}.png'.format(f)))

            img_E = img_E.convert('RGBA')
            #img_E.putalpha(127)
            img_E = img_E.resize((info['size'][1].item(), info['size'][0].item()))
            img_O = Image.fromarray(rgb[f].numpy().astype(np.uint8))
            img_O.putalpha(127)
            img_O = img_O.resize((info['size'][1].item(), info['size'][0].item()))
            img_O.paste(img_E, (0, 0), img_E)

            vis_ids = []
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

            gif_frames.append(img_O)

            '''if f+1 < rgb.shape[0]:
                #print(name, processor.valid_instances[f+1], type(processor.valid_instances[f+1]))
                img_prompt = Image.fromarray(data['orig_rgb'][0][f+1].numpy().astype(np.uint8))
                img_prompt = img_prompt.resize((info['size'][1].item()//2, info['size'][0].item()//2))
                img_E_next = img_prompt.copy()
                img_E_next = Image.fromarray(mask2rgb(merged_masks[f+1], davis_palette, torch.max(processor.valid_instances[f+1])).astype(np.uint8))
                img_E_next = img_E_next.convert('RGBA')
                img_E_next.putalpha(127)
                img_E_next = img_E_next.resize((info['size'][1].item()//2, info['size'][0].item()//2))
                img_O_next = Image.fromarray(data['orig_rgb'][0][f+1].numpy().astype(np.uint8))
                img_O_next = img_O_next.resize((info['size'][1].item()//2, info['size'][0].item()//2))
                img_O_next.paste(img_E_next, (0, 0), img_E_next)
                if len(boxes[f]) > 0:
                    draw = ImageDraw.Draw(img_prompt)
                    for obj_id, box in reversed(list(enumerate(boxes[f]))):
                        if processor.valid_instances[f+1][obj_id].item() in processor.reid_instance_mappings:
                            color = processor.reid_instance_mappings[processor.valid_instances[f+1][obj_id].item()]
                        else:
                            color = processor.valid_instances[f+1][obj_id]
                        draw.rectangle((torch.div(box, 2, rounding_mode='trunc')).tolist(), outline=tuple(davis_palette[color]), width=3)
                merged_prompt = Image.new('RGB', (img_O.width + img_prompt.width + img_O_next.width, img_O.height))
                merged_prompt.paste(img_O, (0, 0))
                merged_prompt.paste(img_prompt, (img_O.width, 0))
                merged_prompt.paste(img_O_next, (img_O.width+img_prompt.width, 0))
                merged_prompt.save(os.path.join(this_out_path3, '{:05d}.png'.format(f)))'''

        gif_save_path = os.path.join(this_out_path2, name+".gif")
        gif_frames[0].save(gif_save_path, format="GIF", append_images=gif_frames, save_all=True, interlace=False, duration=100, loop=0)

        del gif_frames

    del rgb
    del detections
    del processor

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)
