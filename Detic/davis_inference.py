# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import mss
import torch
from pathlib import Path
from PIL import Image 
import matplotlib

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/STCN/')
sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo

# constants
WINDOW_NAME = "Detic"

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
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
    parser.add_argument("--pred_all_class", action='store_true')
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
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

#cm = plt.get_cmap('tab20c')
cm = np.load('color_palette_davis.npy') / 255.
cm = np.concatenate([cm, np.ones((256, 1))], 1)
cm = matplotlib.colors.ListedColormap(cm)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg, args)

    assert(len(args.input) == 1)
    assert(os.path.isdir(args.input[0]))

    davis_image_path = os.path.join(args.input[0], 'JPEGImages_480')
    davis_detects_path = os.path.join(args.input[0], 'Detected_Annotations_NPZ_480')

    for video_folder in tqdm.tqdm(Path(davis_image_path).iterdir()):

        video_name = os.path.basename(video_folder)

        Path(os.path.join(davis_detects_path, video_name)).mkdir(parents=True, exist_ok=True)

        frames = Path(video_folder).glob('*')
        for path in tqdm.tqdm(frames):

            seg_out_filename = os.path.join(davis_detects_path, video_name, os.path.basename(path)[:-4]+'.'+args.save_format)
            if os.path.exists(seg_out_filename): continue

            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)

            # Save detections
            if args.save_format == 'png':
                merged_seg_out = (predictions["instances"].pred_masks * predictions["instances"].scores[..., None, None])
                if merged_seg_out.shape[0] > 0:
                    merged_seg_out = torch.cat([torch.ones_like(merged_seg_out[None, 0])*0.01, merged_seg_out], dim=0) # Add "BG" class channel
                else:
                    # No detections
                    merged_seg_out = torch.ones((1, merged_seg_out.shape[-2], merged_seg_out.shape[-1]))*0.01 # Create "BG" class channel

                merged_seg_out = torch.argmax(merged_seg_out, dim=0)

                seg_img = Image.fromarray((cm(merged_seg_out.cpu().numpy().astype('uint8'))[:, :, :3] * 255).astype('uint8'))
                seg_img.save(seg_out_filename)
            elif args.save_format == 'npz':
                if predictions["instances"].pred_masks.shape[0] > 0:
                    np.savez(seg_out_filename, mask=predictions["instances"].pred_masks[:10].cpu().numpy(), 
                                               scores=predictions["instances"].scores[:10].cpu().numpy())
                else:
                    # No detections
                    H, W = predictions["instances"].pred_masks.shape[-2:]
                    np.savez(seg_out_filename, mask=torch.zeros((1, H, W)), scores=torch.zeros((1,)))
            else:
                pass
            
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            '''if os.path.basename(path)[:-4] == '00000':
                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, video_name+'.jpg')
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output"
                        out_filename = args.output
                    visualized_output.save(out_filename)
                else:
                    pass'''

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    os.makedirs(os.path.join(args.output, video_name), exist_ok=True)
                    out_filename = os.path.join(args.output, video_name, os.path.basename(path)[:-4]+'.jpg')
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                pass
