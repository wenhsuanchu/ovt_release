import numpy as np
import torch
import torch.nn.functional as F
import time
import os
import sys
from os import path
import argparse
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import torch.multiprocessing
import warnings

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

from progressbar import progressbar

sys.path.insert(0, '../../')
sys.path.insert(0, '../CenterNet2/')
sys.path.insert(0, '../gmflow/')
from gmflow.gmflow import GMFlow
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detic.custom_tta import CustomRCNNWithTTA
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer

from dataset.mots_dataset import MOTSDataset
from segment_anything import sam_model_registry, SamCustomPredictor
from sam_propagator_local import Propagator
from ytvostools.mask import encode as rle_encode
from ytvostools.mask import decode as rle_decode

CKPT_PATH = "/home/wenhsuac/ovt/Detic/third_party/SAM/pretrained/sam_vit_h_4b8939.pth"

CATEGORY_LUT = {
    0: 2, # person should be 2
    1: 1, # car should be 1
    2: 1 # automobile should be 1
}

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

def visualize_frame(rgb, merged_masks, valid_instances, reid_mappings, height, width, palette, boxes=[]):
        
    img_E = Image.fromarray(mask2rgb(merged_masks, palette, torch.max(valid_instances)).astype(np.uint8), mode='RGBA')
    img_E = img_E.convert('RGBA')
    img_E = img_E.resize((width, height))
    img_O = Image.fromarray(rgb.numpy().astype(np.uint8))
    img_O.putalpha(127)
    img_O = img_O.resize((width, height))
    img_O.paste(img_E, (0, 0), img_E)

    if len(boxes) > 0:
        draw = ImageDraw.Draw(img_O)
        for obj_id, box in reversed(list(enumerate(boxes))):
            inst_id = valid_instances[obj_id].item()
            while(inst_id in reid_mappings.keys()):
                inst_id = reid_mappings[inst_id]
            if inst_id != -1:
                draw.rectangle(box.tolist(), outline=tuple(palette[inst_id%255]), width=2)
    
    return img_O

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--mots_path', default='/projects/katefgroup/datasets/MOTS/MOTS')
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

mots_path = args.mots_path
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

#tta_detector = GeneralizedRCNNWithTTA(cfg, detector)
tta_detector = CustomRCNNWithTTA(cfg, detector)

sam = sam_model_registry[model_type](checkpoint=CKPT_PATH)
sam.to(device=device)
predictor = SamCustomPredictor(sam)

test_dataset = MOTSDataset(mots_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

total_process_time = 0
total_frames = 0

for test_id, data in enumerate(progressbar(test_loader, max_value=len(test_loader), redirect_stdout=True)):

    #torch.cuda.empty_cache()
    #if test_id > 5: break

    rgb = data['rgb'][0] # [B, S, H, W, C] -> [S, H, W, C]
    info = data['info']
    name = info['name'][0]
    vid_id = info['id']
    size = info['size']
    #size = info['size_480p'] # HACK for vis only!
    #torch.cuda.synchronize()
    process_begin = time.time()
    print(name, size, rgb.shape)

    #if os.path.exists(path.join(out_path, "json", "{}.json".format(name))):
    #    continue

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

    # Run inference model
    processor = Propagator(predictor, detector, flow_predictor, rgb, det_aug)
    if first_detection.shape[0] > 0:
        with torch.no_grad():
            boxes = processor.interact(first_detection, mask_labels, 0, rgb.shape[0])

    #torch.cuda.empty_cache()
        
    # Postprocess predicted masks
    gif_frames = []
    for ti in range(processor.t):
        if len(processor.prob[ti]) > 0:
            prob = torch.from_numpy(rle_decode(processor.prob[ti])).float().permute(2,0,1).unsqueeze(1) # [H, W, N] -> [N, 1, H, W]
        else:
            prob = torch.zeros((1, 1, *size))
        prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)

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

        # Merge instance masks
        merged_masks = torch.zeros((1, *size), dtype=torch.uint8)
        # Make sure we actually have valid instances
        if not torch.equal(processor.valid_instances[ti], torch.tensor([0], device='cuda')):
            for prob_id, label in reversed(list(enumerate(processor.valid_instances[ti]))):
                if label.item() == -1:
                    continue
                mask = prob[prob_id] > 0.5
                merged_masks = mask * (mask * label.item()) + (~mask) * merged_masks
        merged_masks = merged_masks[0] # [1, H, W] -> [H, W]

        # Recompress masks after re-iding
        # MOTS doesn't take overlapping masks so we use merged_masks instead of raw probs
        nonoverlapping_prob = torch.zeros_like(prob)
        for i, inst_id in enumerate(processor.valid_instances[ti]):
            if inst_id != -1:
                # It's possible that after the merging the mask has 0 size
                # So set to invalid if that happens
                if torch.sum(merged_masks == inst_id.item()) == 0:
                    processor.valid_instances[ti][i] = -1
                nonoverlapping_prob[i, 0] = (merged_masks == inst_id.item())
        nonoverlapping_prob = nonoverlapping_prob.squeeze(1).permute(1,2,0) # [N, 1, H, W] -> [H, W, N]
        processor.prob[ti] = rle_encode(np.asfortranarray(nonoverlapping_prob.numpy()).astype(np.uint8))
        #prob = prob.squeeze(1).permute(1,2,0) # [N, 1, H, W] -> [H, W, N]
        #processor.prob[ti] = rle_encode(np.asfortranarray(prob.numpy()).astype(np.uint8))

        merged_masks = merged_masks.numpy()
        
        # Generate gif frames if needed
        if args.output:
            img_O = visualize_frame(data['rgb'][0][ti],
                                    merged_masks,
                                    processor.valid_instances[ti],
                                    processor.reid_instance_mappings,
                                    info['size_480p'][0].item(),
                                    info['size_480p'][1].item(),
                                    davis_palette,
                                    boxes[ti-1] if ti > 0 else list())
            gif_frames.append(img_O)

    #torch.cuda.synchronize()
    total_process_time += time.time() - process_begin
    total_frames += rgb.shape[0]

    # Save the results
    if args.output:

        vis_save_path = path.join(out_path, "vis")
        txt_save_path = path.join(out_path, "txt")
        os.makedirs(vis_save_path, exist_ok=True)
        os.makedirs(txt_save_path, exist_ok=True)

        # Generate output txt
        # We will generate one txt per video so we can parallelize
        with open(path.join(txt_save_path, "{}.txt".format(name)), 'w') as f:
            for ti in range(processor.t):
                for inst_id in processor.valid_instances[ti]:
                    if inst_id != -1:
                        mask_idx = processor.valid_instances[ti].tolist().index(inst_id)
                        seg = processor.prob[ti][mask_idx]
                        seg = seg['counts'].decode()
                        raw_cat_id = processor.valid_labels[ti][mask_idx].item()
                        cat_id = CATEGORY_LUT[raw_cat_id] if raw_cat_id in CATEGORY_LUT.keys() else 2 # default to person category
                        height = info['size'][0].item()
                        width = info['size'][1].item()
                        out_str = f'{ti+1} {inst_id} {cat_id} {height} {width} {seg}'
                        f.write(out_str + '\n')

        # Save gif
        gif_save_path = os.path.join(vis_save_path, name+".gif")
        gif_frames[0].save(gif_save_path, format="GIF", append_images=gif_frames, save_all=True, interlace=False, duration=100, loop=0)

    del rgb
    del predictions
    del processor
    del gif_frames

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print('FPS: ', total_frames / total_process_time)
