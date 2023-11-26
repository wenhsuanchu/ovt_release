import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import collections
import numpy as np
from scipy.optimize import linear_sum_assignment
from detectron2.structures import Instances, Boxes
from ytvostools.mask import encode as rle_encode
from PIL import Image

from utils.flow import run_flow_on_images

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

def masks_to_boxes(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if mask.numel() == 0:
        return torch.zeros((4), device=mask.device, dtype=torch.float)

    bounding_boxes = torch.zeros((4), device=mask.device, dtype=torch.float)

    y, x = torch.where(mask != 0)

    if len(y) > 0:
        bounding_boxes[0] = torch.min(x)
        bounding_boxes[1] = torch.min(y)
        bounding_boxes[2] = torch.max(x)
        bounding_boxes[3] = torch.max(y)
    else:
        bounding_boxes[0] = 0
        bounding_boxes[1] = 0
        bounding_boxes[2] = 0
        bounding_boxes[3] = 0

    return bounding_boxes

class Propagator:
    def __init__(self, prop_net, detector, flow_predictor, reid_net, images, det_aug, filter_labels=True):

        self.filter_labels = filter_labels

        self.prop_net = prop_net
        self.detector = detector
        self.flow_predictor = flow_predictor
        self.reid_net = reid_net

        self.det_aug = det_aug

        # True dimensions
        self.t = images.shape[0]

        self.images = images
        self.device = prop_net.device

        nh, nw = images.shape[1:3]
        self.prob = [torch.zeros((0, 1, nh, nw), dtype=torch.float32, device=self.device)] * self.t

        # Tells us which instance ids are in self.prob (starts from 1)
        self.valid_instances = [torch.tensor([0], device=self.device)] * self.t
        self.valid_labels = [torch.tensor([-1], device=self.device)] * self.t
        self.total_instances = 0

        # Re-ID instance mappings
        self.feat_anchor = []
        self.feat_hist = []
        self.reid_instance_feats = []
        self.new_instance_feats = []
        self.new_instance_ids = []
        self.reid_instance_mappings = collections.OrderedDict()

    def do_pass(self, idx, end_idx, max_tracklets):

        # First get box feats at start_idx
        start_boxes = []
        for instance_mask in self.prob[idx]:
            self.feat_hist.append(collections.deque(maxlen=10))
            instance = instance_mask[0]
            orig_box = masks_to_boxes(instance)
            start_boxes.append(orig_box)
        start_boxes = torch.stack(start_boxes)
        start_boxes, start_box_feats, _ = self._refine_boxes(self.images[idx].numpy(), Boxes(start_boxes.cuda()), self.prob[idx])
        for i, start_box_feat in enumerate(start_box_feats):
            self.feat_hist[i].append(start_box_feat)
            self.feat_anchor.append(start_box_feat)

        self.prop_net.set_image(self.images[idx].numpy().astype(np.uint8))
        transformed_boxes = self.prop_net.transform.apply_boxes_torch(torch.round(start_boxes).long().to(self.prop_net.device),
                                                                      self.images[idx].shape[:2])
        _, _, _, _, prev_embeddings = self.prop_net.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=True,
        )

        closest_ti = end_idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1

        all_boxes = []

        for ti in this_range:

            # Move old masks to CPU and compress masks to save GPU memory
            if ti-2 >= idx:
                # [N, 1, H, W] -> [H, W, N], and move to CPU
                self.prob[ti-2] = self.prob[ti-2].squeeze(1).permute(1,2,0).to(torch.device('cpu'), non_blocking=True)
                self.prob[ti-2] = rle_encode(np.asfortranarray(self.prob[ti-2]).astype(np.uint8))

            boxes = []
            valid_instances = []
            valid_labels = []
            prev_valid_masks = []
            prev_valid_embeddings = []

            # Get flow
            fwd_flow, bwd_flow = run_flow_on_images(self.flow_predictor, self.images[ti-1:ti+1])
            fwd_flow = torch.from_numpy(fwd_flow[0]).to(self.device, non_blocking=True)
            bwd_flow = torch.from_numpy(bwd_flow[0]).to(self.device, non_blocking=True)

            for prob_id, instance_mask in enumerate(self.prob[ti-1]):

                instance = instance_mask[0]

                orig_box = masks_to_boxes(instance)

                # Skip instance and kill tracklet if not flow consistent
                is_consistent, prop_idxes = self._calc_flow_consistency(instance, fwd_flow, bwd_flow)
                if not is_consistent:
                    print(ti, "[Flow] Killing", self.valid_instances[ti-1][prob_id])
                    # If a new instance died before it got re-ided, then just remove the entry from new instances
                    if self.valid_instances[ti-1][prob_id] in self.new_instance_ids:
                        self.new_instance_ids.remove(self.valid_instances[ti-1][prob_id])

                    hist_feats = [self.feat_anchor[self.valid_instances[ti-1][prob_id]-1]] + list(self.feat_hist[self.valid_instances[ti-1][prob_id]-1])
                    if len(hist_feats) > 1:
                        hist_feats = hist_feats[:-1]
                    self.reid_instance_feats.append({"id": self.valid_instances[ti-1][prob_id],
                                                     "feat": torch.stack(hist_feats)})
                    
                    # Remove the oldest instance awaiting re-id if we have too many of them
                    if len(self.reid_instance_feats) > max_tracklets*2:
                        self.reid_instance_feats.pop(0)

                    continue

                # Use optical flow to propagate the box
                idxes = torch.nonzero(instance)
                idxes_valid = instance[prop_idxes[:, 0], prop_idxes[:, 1]]
                # Filter based on valid mask
                idxes = idxes[torch.nonzero(idxes_valid).squeeze(1)]
                sampled_pts = idxes.to(self.device, non_blocking=True)
                # Warp boxes
                warped_box = self._propagate_boxes(sampled_pts, fwd_flow, orig_box)
                boxes.append(warped_box)
                valid_instances.append(self.valid_instances[ti-1][prob_id])
                valid_labels.append(self.valid_labels[ti-1][prob_id])

                if prob_id < prev_embeddings.shape[0]:
                    prev_valid_embeddings.append(prev_embeddings[prob_id])
                prev_valid_masks.append(instance_mask)
            
            # If no instances are consistent, then we don't bother prompting with this frame
            # For now we'll just set all mask predictions to zero
            if len(boxes) == 0:
                all_boxes.append(boxes)
                out_masks = torch.zeros((0,)+self.prob[-1].shape[1:])
                embeddings = torch.zeros((0,)+prev_embeddings.shape[1:], device=self.device)
            else:
                # Refine boxes using box regressor
                boxes = torch.stack(boxes)
                boxes, _, box_scores = self._refine_boxes(self.images[ti].numpy(), Boxes(boxes.to(self.device, non_blocking=True)))
                boxes = torch.round(boxes).long()
                
                # Skip instance and kill tracklet if box scores are low
                updated_valid_instances = []
                updated_boxes = []
                updated_prev_valid_embeddings = []
                updated_prev_valid_masks = []
                
                for box_id, box_score in enumerate(box_scores):
    
                    if box_score < 0.35:
                        print(ti, "[Box Quality] Killing", valid_instances[box_id])
                        # If a new instance died before it got re-ided, then just remove the entry from new instances
                        if valid_instances[box_id] in self.new_instance_ids:
                            self.new_instance_ids.remove(valid_instances[box_id])

                        hist_feats = [self.feat_anchor[self.valid_instances[ti-1][prob_id]-1]] + list(self.feat_hist[valid_instances[box_id]-1])
                        self.reid_instance_feats.append({"id": valid_instances[box_id],
                                                         "feat": torch.stack(hist_feats)})
                        
                        # Remove the oldest instance awaiting re-id if we have too many of them
                        if len(self.reid_instance_feats) > max_tracklets*2:
                            self.reid_instance_feats.pop(0)

                        continue
                    
                    updated_valid_instances.append(valid_instances[box_id])
                    updated_boxes.append(boxes[box_id])
                    if box_id < len(prev_valid_embeddings):
                        updated_prev_valid_embeddings.append(prev_valid_embeddings[box_id])
                    updated_prev_valid_masks.append(prev_valid_masks[box_id])

                valid_instances = updated_valid_instances
                prev_valid_embeddings = updated_prev_valid_embeddings
                prev_valid_masks = updated_prev_valid_masks
                boxes = updated_boxes

                # Get segmentation masks
                if len(boxes) == 0:
                    out_masks = torch.zeros((0,)+self.prob[-1].shape[1:])
                    all_boxes.append(boxes)
                    embeddings = torch.zeros((0,)+prev_embeddings.shape[1:], device=self.device)
                else:
                    # Prompt SAM
                    boxes = torch.stack(boxes)
                    self.prop_net.set_image(self.images[ti].numpy().astype(np.uint8))
                    transformed_boxes = self.prop_net.transform.apply_boxes_torch(boxes.to(self.device, non_blocking=True),
                                                                                  self.images[ti].shape[:2])
                    out_masks, scores, _, mask_tokens, embeddings = self.prop_net.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=True,
                    )

                    # Select the best proposals
                    out_masks = self._select_best_proposal(out_masks, mask_tokens, scores,
                                                           prev_valid_masks, prev_valid_embeddings)
                    
                    # Store feats in history bank
                    new_boxes = []
                    for prob_id, instance_mask in enumerate(out_masks):
                        instance = instance_mask[0]
                        new_boxes.append(masks_to_boxes(instance))
                    _, box_feats, box_scores = self._refine_boxes(self.images[ti].numpy(),
                                                         Boxes(torch.stack(new_boxes).to(self.device, non_blocking=True)), out_masks)
                    for box_id, box_feat in enumerate(box_feats):
                        if box_scores[box_id] > 0.35:
                            self.feat_hist[valid_instances[box_id]-1].append(box_feat)
                        else:
                            print("Skipped", valid_instances[box_id], "in feat hist")

                    all_boxes.append(boxes)

            # Merge with detections
            detected_ti, detected_labels_ti = self._run_detector(self.images[ti].numpy(), self.valid_labels[idx] if self.filter_labels else None)
            detected_ti = detected_ti.cuda()
            if len(detected_ti) > 0:
                if out_masks.shape[0] > 0:
                    merged_mask, valid_instances = self._merge_detections_and_propagations(self.images[ti].numpy(),
                                                                                           detected_ti,
                                                                                           detected_labels_ti,
                                                                                           out_masks,
                                                                                           valid_instances,
                                                                                           valid_labels,
                                                                                           max_tracklets)
                else:
                    merged_mask, valid_instances = self._add_detections(self.images[ti].numpy(),
                                                                        detected_ti,
                                                                        detected_labels_ti,
                                                                        valid_instances,
                                                                        valid_labels,
                                                                        max_tracklets)
            else:
                merged_mask = out_masks

            # Do Re-ID
            #print(ti, "START REID")
            _ = self._reid()

            self.prob[ti] = merged_mask
            if len(valid_instances) > 0:
                self.valid_instances[ti] = torch.stack(valid_instances)
                self.valid_labels[ti] = torch.stack(valid_labels)
            prev_embeddings = embeddings
            #print("MERGED", ti, merged_mask.shape, self.valid_instances[ti])
        
        # Compress the final 2 timesteps
        for ti in range(closest_ti-2, closest_ti):
            self.prob[ti] = self.prob[ti].squeeze(1).permute(1,2,0).to(torch.device('cpu'), non_blocking=True)
            self.prob[ti] = rle_encode(np.asfortranarray(self.prob[ti]).astype(np.uint8))

        return closest_ti, all_boxes

    def interact(self, mask, mask_labels, frame_idx, end_idx, max_tracklets=75):

        # Only select a subset of masks so we don't track too many at once
        mask = mask[:max_tracklets]

        # Init frame 0
        self.prob[frame_idx] = mask.to(self.device)
        self.valid_instances[frame_idx] = torch.arange(mask.shape[0], device=self.device) + 1
        self.valid_labels[frame_idx] = mask_labels
        self.total_instances = mask.shape[0]

        # Track from frame 1 ~ end
        _, boxes = self.do_pass(frame_idx, end_idx, max_tracklets)

        return boxes
    
    def _calc_flow_consistency(self, mask, fwd_flow, bwd_flow, thresh=0.7):

        H, W = mask.shape
        
        # Check if the instance is still valid
        # i.e. if the forward-backward flow goes back to the same instance
        idxes = torch.nonzero(mask)
        instance_size = len(idxes)
        # If the instance is too small, just kill it
        if instance_size < 50:
            return False, None
        flow = fwd_flow[idxes[:, 0], idxes[:, 1]] # Note flows are in (x, y) format
        flow = torch.flip(flow, (1,)) # (x, y) -> (y, x) format
        prop_idxes = torch.round(idxes + flow)
        prop_idxes[:, 0] = torch.clamp(prop_idxes[:, 0], 0, H-1)
        prop_idxes[:, 1] = torch.clamp(prop_idxes[:, 1], 0, W-1)
        prop_idxes = prop_idxes.long()
        flow = bwd_flow[prop_idxes[:, 0], prop_idxes[:, 1]] # Note flows are in (x, y) format
        flow = torch.flip(flow, (1,)) # (x, y) -> (y, x) format
        prop_idxes = torch.round(prop_idxes + flow)
        prop_idxes[:, 0] = torch.clamp(prop_idxes[:, 0], 0, H-1)
        prop_idxes[:, 1] = torch.clamp(prop_idxes[:, 1], 0, W-1)
        prop_idxes = prop_idxes.long()
        in_instance_after_flow = mask.squeeze(0)[prop_idxes[:, 0], prop_idxes[:, 1]]
        consistent_size = torch.sum(in_instance_after_flow)

        if consistent_size / instance_size > thresh:
            return True, prop_idxes
        else:
            return False, None
        
    def _run_detector(self, img, filter_labels=None):

        height, width = img.shape[:2]
        image = self.det_aug.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]

        predictions = self.detector(inputs)[0]
        if len(predictions["instances"].pred_masks) > 0:
            unique_masks, mask_labels = get_unique_masks(predictions)
        else:
            unique_masks = torch.zeros((0, height, width), device=self.device)
            mask_labels = []

        detections = unique_masks.unsqueeze(1) # [N, H, W] -> [N, 1, H, W]

        if detections.shape[0] > 0 and filter_labels is not None:
            matched_idxes = []
            for j in range(detections.shape[0]):
                if mask_labels[j] in filter_labels:
                    matched_idxes.append(j)
            detections = detections[matched_idxes]
            mask_labels = mask_labels[matched_idxes]

        return detections, mask_labels

    def _select_best_proposal(self, out_masks, mask_tokens, scores, prev_valid_masks, prev_valid_embeddings):

        best_out_masks = []
        # If nothing is valid, then we just use the score argmax
        # as we don't have the associated frame embeddings
        if len(prev_valid_embeddings) == 0:
            for mask_id in range(0, mask_tokens.shape[0]):
                best_out_masks.append(out_masks[mask_id, torch.argmax(scores[mask_id], dim=0)])
        # Otherwise, choose the mask that best matches previous predictions
        else:
            prev_valid_embeddings = torch.stack(prev_valid_embeddings)
            # Decode masks using current mask tokens + previous frame embeddings
            b, c, h, w = prev_valid_embeddings.shape
            # Note that we may have a different number of tracklets in this frame than the previous, so we slice mask_tokens
            decoded_prev_masks = (mask_tokens[:b, 1:] @ prev_valid_embeddings.view(b, c, h * w)).view(b, -1, h, w)
            decoded_prev_masks = self.prop_net.model.postprocess_masks(
                decoded_prev_masks,
                self.prop_net.input_size,
                self.prop_net.original_size
            )
            decoded_prev_masks = decoded_prev_masks > self.prop_net.model.mask_threshold
            # Choose the mask that best matches previous predictions (via IOU)
            for mask_id, decoded_mask in enumerate(decoded_prev_masks):
                ious = self._calc_iou(decoded_mask.unsqueeze(1), prev_valid_masks[mask_id].unsqueeze(0))
                best_out_masks.append(out_masks[mask_id, torch.argmax(torch.flatten(ious))])
            # If we had more tracklets than the previous frame, just pick based on the score argmax, 
            # as we don't have the associated frame embeddings
            if b < mask_tokens.shape[0]:
                for mask_id in range(b, mask_tokens.shape[0]):
                    best_out_masks.append(out_masks[mask_id, torch.argmax(scores[mask_id], dim=0)])
        best_out_masks = torch.stack(best_out_masks).unsqueeze(1)
        
        return best_out_masks
    
    def _merge_detections_and_propagations(self, img, detection, detection_labels, propagation, valid_instances, valid_labels, max_tracklets):

        # High IOU requirement for mask replacement
        matched_idxes = self._find_matching_masks(propagation, detection)
        # Low IOU requirement for spawning new tracklets
        matched_idxes_low = self._find_nonoverlapping_masks(detection, propagation, thresh=0.7)

        # Figure out how many instances there are in the two frames
        num_instances = propagation.shape[0] + detection.shape[0] - len(matched_idxes_low)
        num_instances = min(num_instances, max_tracklets)
        merged_mask = torch.zeros((num_instances,)+self.prob[-1].shape[1:], device=self.prob[-1].device)
        # Handle (existing) instances from frame 1
        curr_instance_count = propagation.shape[0]
        #merged_mask[:curr_instance_count] = propagation
        for label_idx in range(propagation.shape[0]):
            if label_idx in matched_idxes[:, 0]:
                # If high IOU, use detections
                detected_idx = matched_idxes[(matched_idxes[:, 0] == label_idx).nonzero().squeeze(1), 1]
                merged_mask[label_idx] = detection[detected_idx]
            else:
                # Otherwise use propagation
                merged_mask[label_idx] = propagation[label_idx]
        # If we don't have too many objects yet, then allow new tracks to spawn
        # This handles (new) instances detected in the second frame
        for label_idx in range(detection.shape[0]):
            # Check if the detection is "new"
            if label_idx not in matched_idxes_low[:, 0]:
                # Make sure we don't have too many tracks
                if curr_instance_count < max_tracklets:
                    merged_mask[curr_instance_count] = detection[label_idx]
                    curr_instance_count += 1
                    valid_instances.append(torch.tensor(self.total_instances+1, device=self.device))
                    valid_labels.append(detection_labels[label_idx])
                    self.feat_hist.append(collections.deque(maxlen=9))

                    _, new_box_feats, _ = self._refine_boxes(img, Boxes(masks_to_boxes(detection[label_idx][0]).unsqueeze(0)), detection[label_idx].unsqueeze(0)) # [1, C]
                    self.feat_hist[self.total_instances].append(new_box_feats.squeeze(0))
                    self.feat_anchor.append(new_box_feats.squeeze(0))
                    #print("NEW", self.total_instances+1)
                    self.new_instance_ids.append(self.total_instances+1)

                    self.total_instances += 1
                # Otherwise we are done
                else:
                    break

        return merged_mask, valid_instances
    
    def _add_detections(self, img, detection, detection_labels, valid_instances, valid_labels, max_tracklets):

        num_instances = min(detection.shape[0], max_tracklets)

        merged_mask = torch.zeros((num_instances,)+self.prob[-1].shape[1:], device=self.prob[-1].device)

        # Handle (new) instances detected in the second frame
        # We match using lower thresholds to prevent over-initializing
        curr_instance_idx = 0
        for label_idx in range(num_instances):
            merged_mask[curr_instance_idx] = detection[label_idx]
            curr_instance_idx += 1
            valid_instances.append(torch.tensor(self.total_instances+1, device=self.device))
            valid_labels.append(detection_labels[label_idx])
            self.feat_hist.append(collections.deque(maxlen=9))

            _, new_box_feats, _ = self._refine_boxes(img, Boxes(masks_to_boxes(detection[label_idx][0]).unsqueeze(0)), detection[label_idx].unsqueeze(0)) # [1, C]
            self.feat_hist[self.total_instances].append(new_box_feats.squeeze(0))
            self.feat_anchor.append(new_box_feats.squeeze(0))
            self.new_instance_ids.append(self.total_instances+1)

            self.total_instances += 1

        return merged_mask, valid_instances

    #0.8/0.75 for dino
    #0.6/0.75 for detic
    def _reid(self, matching_threshold=0.8, ratio_threshold=0.55):

        reid_count = 0

        # Do Re-ID
        if len(self.new_instance_ids) > 0 and len(self.reid_instance_feats) > 0:

            no_longer_new = []
            # Aggregate new instance feats
            new_box_feat_ids = []
            new_box_feats = []
            for new_instance_id in self.new_instance_ids:
                hist_feats = list(self.feat_hist[new_instance_id-1])
                new_box_feats.append(torch.stack(hist_feats))
                new_box_feat_ids.append(torch.full((len(self.feat_hist[new_instance_id-1]),), new_instance_id))

                if len(self.feat_hist[new_instance_id-1]) == 10:
                    no_longer_new.append(new_instance_id)
            new_box_feat_ids = torch.cat(new_box_feat_ids) # [NEW]
            new_box_feats = torch.cat(new_box_feats) # [NEW, C, HW]
            new_box_feats = new_box_feats / torch.norm(new_box_feats, dim=1, keepdim=True)
            # Aggregate killed instance feats
            killed_box_feat_ids = []
            killed_box_feats = []
            for entry in self.reid_instance_feats:
                if entry["id"] not in self.reid_instance_mappings.values():
                    killed_box_feat_ids.append(torch.full((len(entry["feat"]),), entry["id"]))
                    killed_box_feats.append(torch.stack(list(entry["feat"])))
            if len(killed_box_feat_ids) == 0:
                return
            killed_box_feat_ids = torch.cat(killed_box_feat_ids) # [KILLED]
            killed_box_feats = torch.cat(killed_box_feats) # [KILLED, C, HW]
            killed_box_feats = killed_box_feats / torch.norm(killed_box_feats, dim=1, keepdim=True)

            # Calculate feature differences
            diffs = torch.einsum('aij,bik->abjk', new_box_feats, killed_box_feats) # [NEW, KILLED, HW, HW]
            #diffs = (new_box_feats.pow(2).sum(1).unsqueeze(1).unsqueeze(-1) # [NEW, 1, HW, 1]
            #         + killed_box_feats.pow(2).sum(1).unsqueeze(0).unsqueeze(-1) # [1, KILLED, HW, 1]
            #         - 2*torch.einsum('aij,bik->abjk', new_box_feats, killed_box_feats)) # [NEW, KILLED, HW, HW]
            num_feats_in_box = diffs.shape[-1]
            # Clamp match count to 1 because each feature can only match with at least one other feature
            all_matches = torch.clamp((diffs > matching_threshold).sum(-1), max=1.).sum(-1) # [NEW, KILLED, HW, HW] -> [NEW, KILLED]
            #all_matches = torch.clamp((diffs < matching_threshold).sum(-1), max=1.).sum(-1) # [NEW, KILLED, HW, HW] -> [NEW, KILLED]

            # Use a mask to keep track of what has already been matched
            matched_mask = torch.ones_like(all_matches)
            max_match = torch.max(all_matches*matched_mask)
            max_match_idx = (all_matches*matched_mask == max_match).nonzero()[0]
            reid_matches = []
            reid_match_counts = []
            while max_match / num_feats_in_box > ratio_threshold:
                # Check if we get matches on both sides
                new_id = new_box_feat_ids[max_match_idx[0]].item()
                old_id = killed_box_feat_ids[max_match_idx[1]].item()
                #if new_id > old_id:
                #print("MAXMATCH", new_id, old_id, max_match, num_feats_in_box)
                if (new_id, old_id) not in reid_matches:
                        reid_matches.append((new_id, old_id))
                        reid_match_counts.append(max_match / num_feats_in_box)
                else:
                    reid_match_counts[reid_matches.index((new_id, old_id))] += max_match / num_feats_in_box

                # Prevent dupicate entries, since it's possible that multiple pairs of feats get matched at once
                '''if new_id not in self.reid_instance_mappings.keys() and new_id > old_id:
                    self.reid_instance_mappings[new_id] = old_id
                    self.feat_hist[new_id-1] = self.feat_hist[old_id-1] + self.feat_hist[new_id-1]
                    print("REID", new_id, "->", old_id)
                    # Remove re-ided entries
                    if new_id in self.new_instance_ids:
                        self.new_instance_ids.remove(new_id)
                    for entry in self.reid_instance_feats:
                        if entry["id"].item() == old_id:
                            self.reid_instance_feats.remove(entry)'''
                #matched_mask = matched_mask * (~(killed_box_feat_ids == old_id)).cuda()
                matched_mask[max_match_idx[0], max_match_idx[1]] = 0
                max_match = torch.max(all_matches*matched_mask)
                max_match_idx = (all_matches*matched_mask == max_match).nonzero()[0]

            sorted_counts_idx = sorted(range(len(reid_match_counts)), key=lambda k: reid_match_counts[k], reverse=True)
            for idx in sorted_counts_idx:
                new_id, old_id = reid_matches[idx]
                if new_id not in self.reid_instance_mappings.keys() and old_id not in self.reid_instance_mappings.values():
                    self.reid_instance_mappings[new_id] = old_id
                    self.feat_hist[new_id-1] = self.feat_hist[old_id-1] + self.feat_hist[new_id-1]
                    self.feat_anchor[new_id-1] = self.feat_anchor[old_id-1]
                    print("REID", new_id, "->", old_id, reid_match_counts[idx])
                    # Remove re-ided entries
                    if new_id in self.new_instance_ids:
                        self.new_instance_ids.remove(new_id)
                    for entry in self.reid_instance_feats:
                        if entry["id"].item() == old_id:
                            self.reid_instance_feats.remove(entry)

            # If the entry has persisted long enough before getting re-ided, don't try to re-id it
            for entry in no_longer_new:
                if entry in self.new_instance_ids:
                    self.new_instance_ids.remove(entry)
        
        return reid_count
    
    def _calc_iou(self, masks1, masks2):

        # masks1: [N, 1, H, W]
        # masks2: [N2, 1, H, W]

        # Convert to boolean masks
        N1, _, H, W = masks1.shape
        N2, _, H, W = masks2.shape
        masks1 = (masks1.float() > 0.5).float().squeeze(1).reshape(N1, H*W)
        masks2 = (masks2.float() > 0.5).float().squeeze(1).reshape(N2, H*W)
        intersection = torch.matmul(masks1, masks2.t())

        area1 = masks1.sum(dim=1).view(1, N1)
        area2 = masks2.sum(dim=1).view(1, N2)
        union = (area1.t() + area2) - intersection
        
        iou = (intersection + 1e-6) / (union + 1e-6) # [N1, N2]

        return iou
    
    def _find_matching_masks(self, propagation, detected, iou_thresh=0.8):

        # Propagation: [N, 1, H, W]
        # Detected: [N2, 1, H, W]
        
        iou = self._calc_iou(propagation, detected) # [N1, N2]
        thresholded = iou # When we add semantic label thresholding
        row_idx, col_idx = linear_sum_assignment(-thresholded.cpu()) # Score -> cost
        matched_idxes = torch.stack([torch.from_numpy(row_idx), torch.from_numpy(col_idx)], axis=1)
        matched_score = iou[row_idx, col_idx]
        matched_idxes = matched_idxes[torch.nonzero(matched_score>iou_thresh).flatten().cpu()] # [N, 2] or [2]
        # This happens when we only have one pair of masks matched
        # It makes subsequent functions so we unsqueeze for an extra dimension
        if len(matched_idxes.shape) == 1:
            matched_idxes = matched_idxes[None, :]

        return matched_idxes
    
    def _find_nonoverlapping_masks(self, detected, masks, thresh=0.1):

        # Detected: [N, 1, H, W]
        # Masks: [N2, 1, H, W]

        iou = self._calc_iou(detected, masks) # [N1, N2]
        thresholded = iou # When we add semantic label thresholding
        max_ious, max_idxes = torch.max(thresholded, dim=1)
        matched_idxes = []
        for i, (max_iou, max_idx) in enumerate(zip(max_ious, max_idxes)):
            if max_iou > thresh:
                matched_idxes.append(torch.tensor([i, max_idx]))
        # If nothing is matched, construct a dummy tensor
        if len(matched_idxes) == 0:
            matched_idxes = torch.full((0, 2), -1)
        else:
            matched_idxes = torch.stack(matched_idxes)

        return matched_idxes

    def _propagate_boxes(self, sampled_pts, fwd_flow, orig_box):

        # sampled_pts: [K, 2]
        # flow: [H, W, 2], (x, y) format
        # orig_box: [N, 4]

        H, W, _ = fwd_flow.shape

        # Propagate points based on flow
        sampled_flow = fwd_flow[sampled_pts[:, 0], sampled_pts[:, 1]].to(self.device, non_blocking=True)
        sampled_flow = torch.flip(sampled_flow, (1,)) # (x, y) -> (y, x)
        warped_pts = torch.round(sampled_pts+sampled_flow)
        # Clamp and convert to long
        warped_pts[:, 0] = torch.clamp(warped_pts[:, 0], 0, H-1)
        warped_pts[:, 1] = torch.clamp(warped_pts[:, 1], 0, W-1)
        warped_pts = warped_pts.long()

        # Solve for the scale and translation
        A = torch.zeros((sampled_pts.shape[0]*2, 4), device=self.device)
        one_zeros = torch.ones(sampled_pts.shape[0]).reshape(-1, 1).tile(1,2)
        one_zeros[:, 1] = 0
        one_zeros = one_zeros.flatten()
        zero_ones = torch.ones(sampled_pts.shape[0]).reshape(-1, 1).tile(1,2)
        zero_ones[:, 0] = 0
        zero_ones = zero_ones.flatten()
        x_zeros = sampled_pts[:, 1].reshape(-1, 1).tile(1, 2)
        x_zeros[:, 1] = 0
        x_zeros = x_zeros.flatten()
        zero_ys = sampled_pts[:, 0].reshape(-1, 1).tile(1, 2)
        zero_ys[:, 0] = 0
        zero_ys = zero_ys.flatten()
        A[:, 0] = one_zeros
        A[:, 1] = zero_ones
        A[:, 2] = x_zeros
        A[:, 3] = zero_ys

        b = torch.flip(warped_pts, (1,)).flatten().float().reshape(-1, 1)
        sol = torch.linalg.lstsq(A, b).solution

        # Propagate the box
        box = torch.tensor([orig_box[0]*sol[2]+sol[0]-5,
                            orig_box[1]*sol[3]+sol[1]-5,
                            orig_box[2]*sol[2]+sol[0]+5,
                            orig_box[3]*sol[3]+sol[1]+5])
        box[0] = torch.clamp(box[0], 0, W-1)
        box[1] = torch.clamp(box[1], 0, H-1)
        box[2] = torch.clamp(box[2], 0, W-1)
        box[3] = torch.clamp(box[3], 0, H-1)
        box = torch.round(box).long()

        #box = torch.tensor([min_x, min_y, max_x, max_y])
        return box
    
    def _refine_boxes(self, img, boxes, masks=None):

        # img: [H, W, 3]
        # boxes: Boxes()

        # Refine boxes using detic's box regressor

        height, width = img.shape[:2]
        image = self.det_aug.get_transform(img).apply_image(img)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]

        images = self.detector.preprocess_image(inputs)
        features = self.detector.backbone(images.tensor)

        scaled_boxes = boxes
        scaled_boxes.scale(images.image_sizes[0][1]/inputs[0]["width"],
                           images.image_sizes[0][0]/inputs[0]["height"])

        proposals = self._create_proposals_from_boxes(scaled_boxes.tensor, images.image_sizes)

        _, roi_dict = self.detector.roi_heads(images, features, proposals)

        out_boxes = Boxes(torch.clone(roi_dict["boxes"][0].detach()))
        out_boxes.scale(inputs[0]["width"]/images.image_sizes[0][1],
                        inputs[0]["height"]/images.image_sizes[0][0])
        out_boxes.clip(images.image_sizes[0])

        box_features = [features[f] for f in self.detector.roi_heads.box_in_features]

        img_transform = T.Compose([
            #T.Resize((2*14*(height//14), 2*14*(width//14))),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        feature_list = []
        with torch.no_grad():
            feature_list = box_features
            #k16, _, _, _, _ = self.reid_net.encode_key(img_transform(Image.fromarray(img)).unsqueeze(0).cuda())
            #dino_features = self.dino.forward_features(img_transform(Image.fromarray(img)).unsqueeze(0).cuda())["x_norm_patchtokens"]
            #dino_features = dino_features.permute(0,2,1).reshape(1, -1, 2*height//14, 2*width//14)
            #for features in box_features:
            #    feature_list.append(F.interpolate(k16, features.shape[-2:], mode='bilinear', align_corners=False))
            #print(features.shape, feature_list[-1].shape)
        #assert(False)
        #box_features = self.detector.roi_heads.box_pooler(box_features, [Boxes(roi_dict["boxes"][0].detach())])
        box_features = self.detector.roi_heads.box_pooler(feature_list, [Boxes(roi_dict["boxes"][0].detach())])
        box_features = torch.flatten(box_features, 2)
        #mask_features = [features[f] for f in self.detector.roi_heads.in_features]
        #mask_features = self.detector.roi_heads.mask_pooler(mask_features, [Boxes(roi_dict["boxes"][0].detach())])
        #mask_features = torch.flatten(mask_features, 2)

        '''if masks is not None:
            mask_features = []
            featmap = feature_list[0].squeeze(0)
            flattened_featmap = featmap.flatten(1)
            flattened_masks = F.interpolate(masks.float(), featmap.shape[-2:], mode='bilinear', align_corners=False).flatten(1) # [N, 1, H, W] -> [N, H*W]
            for idx, instance_mask in enumerate(flattened_masks):
                pt_coords = (instance_mask > 0.5).nonzero(as_tuple=False).squeeze(1)  # [X]
                if len(pt_coords) > 0:
                    mask_features.append(flattened_featmap[:, pt_coords].mean(-1, keepdim=True))
                else:
                    mask_features.append(box_features[idx].mean(-1, keepdim=True))
            mask_features = torch.stack(mask_features)
            box_features = mask_features'''

        if masks is not None:
            img_transform = T.Normalize(mean=(103.53, 116.28, 123.675), std=(57.375, 57.12, 58.395))
            with torch.no_grad():
                reid_input = img_transform(torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()).cuda()
                fmaps = self.reid_net.feature_extractor(reid_input)
                fmaps = {scale: f.reshape(1, *f.shape[1:]) for scale, f in fmaps.items()}
                split_bg_masks = self.reid_net.encoder.query_initializer.get_patch_masks(1, height, width, device=masks.device)
                # set bg masks to zero at fg pixel locations
                fg_masks = masks.squeeze(1).unsqueeze(0).float() # {N, 1, H, W} - > [1, N, H, W]
                bg_masks = torch.where(fg_masks.sum(1).unsqueeze(1) > 0, torch.zeros_like(split_bg_masks), split_bg_masks)
                H_NEW = fmaps[self.reid_net.encoder_mask_scale].shape[-2]
                W_NEW = fmaps[self.reid_net.encoder_mask_scale].shape[-1]
                bg_masks = F.interpolate(bg_masks, (H_NEW, W_NEW))
                fg_masks = F.interpolate(fg_masks, (H_NEW, W_NEW))

                ref_encoder_output = self.reid_net.encoder(fmaps=fmaps, fg_mask=fg_masks, bg_mask=bg_masks)
                box_features = ref_encoder_output["fg_queries"].squeeze(0).unsqueeze(-1)

        scores = torch.max(roi_dict["scores"][0], 1)[0]

        return out_boxes.tensor, box_features, scores
    
    def _create_proposals_from_boxes(self, boxes, image_sizes):

        # boxes: Boxes()
        # image_size: (H, W)
        
        # Takes boxes and add dummy objectness logits to form proposals

        boxes = [Boxes(boxes.detach())]
        proposals = []

        for boxes_per_image, image_size in zip(boxes, image_sizes):

            boxes_per_image.clip(image_size)
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            prop.objectness_logits = torch.tensor([0.99]*boxes_per_image.tensor.shape[0], device=boxes_per_image.device)
            proposals.append(prop)

        return proposals