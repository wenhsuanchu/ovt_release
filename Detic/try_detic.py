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
import copy
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

from detectron2.modeling import build_model
from detectron2.modeling import DatasetMapperTTA, GeneralizedRCNNWithTTA
from detectron2.structures import ImageList, Instances, Boxes
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detic.predictor import VisualizationDemo
from typing import Dict, List, Optional, Tuple
from detectron2.structures import Instances, ROIMasks

# constants
WINDOW_NAME = "Detic"

def detector_postprocess(
    results: Instances, output_height: int, output_width: int, mask_threshold: float = 0.5
):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.
    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.
    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks):
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        results.pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold
        ).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results

def postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
    """
    Rescale the output instances to the target size.
    """
    # note: private function; subject to changes
    processed_results = []
    for results_per_image, input_per_image, image_size in zip(
        instances, batched_inputs, image_sizes
    ):
        height = input_per_image.get("height", image_size[0])
        width = input_per_image.get("width", image_size[1])
        print(height, width, image_sizes)
        r = detector_postprocess(results_per_image, height, width)
        processed_results.append({"instances": r})
    return processed_results

def create_proposals_from_boxes(boxes, image_sizes):#, objectness_logits):
    """
    Add objectness_logits
    """
    boxes = [Boxes(boxes.detach())]
    #objectness_logits = [objectness_logits]
    proposals = []
    #for boxes_per_image, image_size, objectness in zip(boxes, image_sizes, objectness_logits):
    for boxes_per_image, image_size in zip(boxes, image_sizes):
        boxes_per_image.clip(image_size)
        prop = Instances(image_size)
        prop.proposal_boxes = boxes_per_image
        prop.objectness_logits = torch.tensor([0.99]*boxes_per_image.tensor.shape[0], device=boxes_per_image.device)
        #prop.objectness_logits = objectness #torch.tensor([0.9]*boxes_per_image.tensor.shape[0], device=boxes_per_image.device)
        proposals.append(prop)
    return proposals

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
    if args.pred_one_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    #cfg.INPUT.MIN_SIZE_TEST = cfg.INPUT.MIN_SIZE_TEST // 2
    #cfg.INPUT.MAX_SIZE_TEST = cfg.INPUT.MAX_SIZE_TEST // 2
    cfg.TEST.AUG.FLIP = True
    cfg.TEST.AUG.MIN_SIZES = [int(cfg.INPUT.MIN_SIZE_TEST*0.75), cfg.INPUT.MIN_SIZE_TEST, int(cfg.INPUT.MIN_SIZE_TEST*1.25)]
    cfg.TEST.AUG.MAX_SIZE = cfg.INPUT.MAX_SIZE_TEST
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
        "--pred_one_class",
        action='store_true',
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

    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    model = GeneralizedRCNNWithTTA(cfg, model)

    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    demo = VisualizationDemo(cfg, args)

    assert(len(args.input) == 1)
    assert(os.path.isdir(args.input[0]))

    davis_image_path = os.path.join(args.input[0], 'JPEGImages_480')
    davis_detects_path = os.path.join(args.input[0], 'Debug')

    for video_folder in tqdm.tqdm(Path(davis_image_path).iterdir()):

        video_name = os.path.basename(video_folder)
        video_name = 'lab-coat'
        video_folder = os.path.join(davis_image_path, video_name)

        Path(os.path.join(davis_detects_path, video_name)).mkdir(parents=True, exist_ok=True)

        frames = Path(video_folder).glob('*')
        for path in tqdm.tqdm(frames):

            seg_out_filename = os.path.join(davis_detects_path, video_name, os.path.basename(path)[:-4]+'.'+args.save_format)
            #if os.path.exists(seg_out_filename): continue

            img = read_image(path, format="BGR")
            print(img.shape)
            print(cfg.INPUT.FORMAT)

            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            print(predictions["instances"].pred_boxes.tensor.shape)
            #input()
            #raw_boxes = predictions["instances"].pred_boxes.tensor

            img = img[:, :, ::-1]
            height, width = img.shape[:2]
            image = aug.get_transform(img).apply_image(img)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": height, "width": width}]
            #print(torch.unique(model(inputs)[0]["instances"].pred_boxes.tensor, dim=0))
            #input()
            with torch.no_grad():
                predictions = model(inputs)[0]
                processed_predictions = copy.deepcopy(predictions)
                processed_predictions["instances"] = Instances(predictions["instances"].image_size)
                raw_boxes = predictions["instances"].pred_boxes.tensor
                processed_boxes = [raw_boxes[0]]
                processed_scores = [predictions["instances"].scores[0]]
                processed_classes = [predictions["instances"].pred_classes[0]]
                processed_masks = [predictions["instances"].pred_masks[0]]
                for idx in range(1, len(raw_boxes)):
                    if not torch.allclose(raw_boxes[idx], raw_boxes[idx-1]):
                        processed_boxes.append(raw_boxes[idx])
                        processed_scores.append(predictions["instances"].scores[idx])
                        processed_classes.append(predictions["instances"].pred_classes[idx])
                        processed_masks.append(predictions["instances"].pred_masks[idx])
                processed_boxes = Boxes(torch.stack(processed_boxes))
                processed_scores = torch.stack(processed_scores)
                processed_classes = torch.stack(processed_classes)
                processed_masks = torch.stack(processed_masks)

                processed_predictions["instances"].set("pred_boxes", processed_boxes)
                processed_predictions["instances"].set("scores", processed_scores)
                processed_predictions["instances"].set("classes", processed_classes)
                processed_predictions["instances"].set("pred_masks", processed_masks)

            #images = model.preprocess_image(inputs)
            #features = model.backbone(images.tensor)
            #proposals, _ = model.proposal_generator(images, features, None)
            #print("LOGITS", proposals[0].get('objectness_logits').shape, proposals[0].get('proposal_boxes').tensor.shape)
            #input()
            #print(torch.max(proposals[0].get('proposal_boxes').tensor[:, 2]))
            #proposals = create_proposals_from_boxes(proposals[0].get('proposal_boxes').tensor, images.image_sizes, proposals[0].get('objectness_logits'))
            #print(inputs[0]["width"], inputs[0]["height"])
            #print(images.image_sizes[0][1], images.image_sizes[0][0])
            #scaled_boxes = predictions["instances"].pred_boxes
            #scaled_boxes.scale(images.image_sizes[0][1]/inputs[0]["width"],
            #                   images.image_sizes[0][0]/inputs[0]["height"])
            #print(scaled_boxes.tensor)
            #proposals = create_proposals_from_boxes(scaled_boxes.tensor, images.image_sizes)
            #print("LOGITS2", proposals[0].get('objectness_logits').shape)
            #input()
            #instances, roi_dict = model.roi_heads(images, features, proposals)
            # print(roi_dict["boxes"][0].shape)
            # boxes = Boxes(roi_dict["boxes"][0].detach())
            # boxes.scale(inputs[0]["width"]/images.image_sizes[0][1],
            #             inputs[0]["height"]/images.image_sizes[0][0])
            # boxes.clip(images.image_sizes[0])
            # box_features = [features[f] for f in model.roi_heads.box_in_features]
            # box_features = model.roi_heads.box_pooler(box_features, [boxes])
            # pooled_box_features = torch.mean(box_features, dim=(-2, -1))
            # print("FEATS", pooled_box_features.shape)
            #predictions = postprocess(instances, inputs, images.image_sizes)
            print(torch.unique_consecutive(predictions["instances"].pred_boxes.tensor, dim=0).shape, predictions["instances"].pred_boxes.tensor, predictions["instances"].scores)
            print(processed_predictions["instances"].pred_boxes.tensor.shape, processed_predictions["instances"].pred_boxes.tensor)
            predictions = processed_predictions
            #print(boxes.tensor)

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

        print("GOT HERE")
        input()
