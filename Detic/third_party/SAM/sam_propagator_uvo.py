import torch
import collections
import numpy as np
from scipy.optimize import linear_sum_assignment
from detectron2.structures import Instances, Boxes

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
    def __init__(self, prop_net, detector, images, num_objects, det_aug):
        self.prop_net = prop_net
        self.detector = detector

        self.det_aug = det_aug
        
        self.k = num_objects

        # True dimensions
        self.t = images.shape[0]

        # Padded dimensions
        nh, nw = images.shape[1:3]

        self.images = images
        self.device = prop_net.device

        # Background included, not always consistent (i.e. sum up to 1)
        self.prob = [torch.zeros((self.k, 1, nh, nw), dtype=torch.float32, device=self.device)] * self.t
        self.prob[0] = 1e-7

        # Tells us which instance ids are in self.prob (starts from 1)
        self.valid_instances = [torch.tensor([0], device=self.device)] * self.t
        self.total_instances = 0

        # Re-ID instance mappings
        self.feat_hist = []
        self.reid_instance_feats = []
        self.new_instance_feats = []
        self.new_instance_ids = []
        self.reid_instance_mappings = collections.OrderedDict()

    def do_pass(self, idx, end_idx, fwd_flow, bwd_flow, detected, max_tracklets):

        # First get box feats at start_idx
        start_boxes = []
        for instance_mask in self.prob[idx]:
            self.feat_hist.append(collections.deque(maxlen=10))
            instance = instance_mask[0]
            orig_box = masks_to_boxes(instance)
            start_boxes.append(orig_box)
        start_boxes = torch.stack(start_boxes)
        start_boxes, start_box_feats = self._refine_boxes(self.images[idx].numpy(), Boxes(start_boxes.cuda()))
        for i, start_box_feat in enumerate(start_box_feats):
            self.feat_hist[i].append(start_box_feat)

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

            boxes = []
            valid_instances = []
            prev_valid_masks = []
            prev_valid_embeddings = []

            for prob_id, instance_mask in enumerate(self.prob[ti-1]):

                instance = instance_mask[0]

                orig_box = masks_to_boxes(instance)

                # Skip instance and kill tracklet if needed
                is_consistent, prop_idxes = self._calc_flow_consistency(instance, fwd_flow[ti-1], bwd_flow[ti-1])
                if not is_consistent:
                    print(ti, "Killing", self.valid_instances[ti-1][prob_id])
                    # If a new instance died before it got re-ided, then just remove the entry from new instances
                    if self.valid_instances[ti-1][prob_id] in self.new_instance_ids:
                        self.new_instance_ids.remove(self.valid_instances[ti-1][prob_id])

                    self.reid_instance_feats.append({"id": self.valid_instances[ti-1][prob_id],
                                                     "feat": torch.stack(list(self.feat_hist[self.valid_instances[ti-1][prob_id]-1]))})
                    continue

                # Use optical flow to propagate the box
                idxes = torch.nonzero(instance)
                idxes_valid = instance[prop_idxes[:, 0], prop_idxes[:, 1]]
                # Filter based on valid mask
                idxes = idxes[torch.nonzero(idxes_valid).squeeze(1)]
                sampled_pts = idxes.to(self.device)
                # Warp boxes
                warped_box = self._propagate_boxes(sampled_pts, fwd_flow[ti-1], orig_box)
                boxes.append(warped_box)
                valid_instances.append(self.valid_instances[ti-1][prob_id])

                if prob_id < prev_embeddings.shape[0]:
                    prev_valid_embeddings.append(prev_embeddings[prob_id])
                prev_valid_masks.append(instance_mask)
            
            # If no instances are consistent, then we don't bother prompting with this frame
            # For now we'll just set all mask predictions to zero
            if len(boxes) == 0:
                all_boxes.append(boxes)
                out_masks = torch.zeros((0,)+self.prob[0].shape[1:])
            else:
                # Refine boxes using box regressor
                boxes = torch.stack(boxes)
                boxes, box_feats = self._refine_boxes(self.images[ti].numpy(), Boxes(boxes.cuda()))
                boxes = torch.round(boxes).long()
                for idx, box_feat in enumerate(box_feats):
                    self.feat_hist[self.valid_instances[ti-1][idx]-1].append(box_feat)

                all_boxes.append(boxes)

                # Prompt SAM for seg masks
                self.prop_net.set_image(self.images[ti].numpy().astype(np.uint8))
                transformed_boxes = self.prop_net.transform.apply_boxes_torch(boxes.to(self.prop_net.device),
                                                                            self.images[ti].shape[:2])
                '''out_masks, _, _ = self.prop_net.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )'''
                #print("SINGLE", out_masks.shape)
                out_masks, scores, _, mask_tokens, embeddings = self.prop_net.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=True,
                )

                # Select the best proposals
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
                    decoded_prev_masks = self.prop_net.model.postprocess_masks(decoded_prev_masks, self.prop_net.input_size, self.prop_net.original_size)
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
                out_masks = best_out_masks

            # Merge with detections
            detected_ti = detected[ti][0].cuda()
            if len(detected_ti) > 0:
                if out_masks.shape[0] > 0:
                    merged_mask, valid_instances = self._merge_detections_and_propagations(self.images[ti].numpy(),
                                                                                           detected_ti,
                                                                                           out_masks,
                                                                                           valid_instances,
                                                                                           max_tracklets)
                else:
                    merged_mask, valid_instances = self._add_detections(self.images[ti].numpy(),
                                                                        detected_ti,
                                                                        valid_instances,
                                                                        max_tracklets)
            else:
                merged_mask = out_masks

            # Do Re-ID
            _ = self._reid()

            self.prob[ti] = merged_mask
            if len(valid_instances) > 0:
                self.valid_instances[ti] = torch.stack(valid_instances)
            prev_embeddings = embeddings
            #print("MERGED", ti, merged_mask.shape, self.valid_instances[ti])

        return closest_ti, all_boxes

    def interact(self, mask, frame_idx, end_idx, fwd_flow, bwd_flow, detected, max_tracklets=75):

        # Only select a subset of masks so we don't track too many at once
        mask = mask[:max_tracklets]

        # Init frame 0
        self.prob[frame_idx] = mask.to(self.device)
        self.valid_instances[frame_idx] = torch.arange(mask.shape[0], device=self.device) + 1
        self.total_instances = mask.shape[0]

        # Track from frame 1 ~ end
        _, boxes = self.do_pass(frame_idx, end_idx, fwd_flow.to(self.device), bwd_flow.to(self.device), detected, max_tracklets)

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

    def _merge_detections_and_propagations(self, img, detection, propagation, valid_instances, max_tracklets):

        # High IOU requirement for mask replacement
        matched_idxes = self._find_matching_masks(propagation, detection)
        # Low IOU requirement for spawning new tracklets
        matched_idxes_low = self._find_nonoverlapping_masks(detection, propagation, thresh=0.7)

        # Figure out how many instances there are in the two frames
        num_instances = propagation.shape[0] + detection.shape[0] - len(matched_idxes_low)
        num_instances = max(num_instances, propagation.shape[0])
        # If we don't have too many objects yet, then allow new tracklets to spawn
        if num_instances < max_tracklets:

            merged_mask = torch.zeros((num_instances,)+self.prob[0].shape[1:], device=self.prob[0].device)
            # Handle instances from the first frame
            for label_idx in range(propagation.shape[0]):
                if label_idx in matched_idxes[:, 0]:
                    # If high IOU, use detections
                    detected_idx = matched_idxes[(matched_idxes[:, 0] == label_idx).nonzero().squeeze(1), 1]
                    merged_mask[label_idx] = detection[detected_idx]
                else:
                    # Otherwise use propagation
                    merged_mask[label_idx] = propagation[label_idx]
            # Handle (new) instances detected in the second frame
            # We match using lower thresholds to prevent over-initializing
            curr_instance_idx = propagation.shape[0]
            for label_idx in range(detection.shape[0]):
                if label_idx not in matched_idxes_low[:, 0]:
                    merged_mask[curr_instance_idx] = detection[label_idx]
                    curr_instance_idx += 1
                    valid_instances.append(torch.tensor(self.total_instances+1, device=self.device))
                    self.feat_hist.append(collections.deque(maxlen=10))

                    _, new_box_feats = self._refine_boxes(img, Boxes(masks_to_boxes(detection[label_idx][0]).unsqueeze(0))) # [1, C]
                    self.feat_hist[self.total_instances].append(new_box_feats.squeeze(0))
                    self.new_instance_ids.append(self.total_instances+1)

                    self.total_instances += 1

        # Otherwise we skip the parts that generate new tracklets
        else:
            merged_mask = torch.zeros_like(propagation)
            num_instances = propagation.shape[0] # Adjust num_instances
            # Handle instances from the first frame
            for label_idx in range(propagation.shape[0]):
                if label_idx in matched_idxes[:, 0]:
                    # If high IOU, use detections
                    detected_idx = matched_idxes[(matched_idxes[:, 0] == label_idx).nonzero().squeeze(1), 1]
                    merged_mask[label_idx] = detection[detected_idx]
                else:
                    # Otherwise use propagation
                    merged_mask[label_idx] = propagation[label_idx]
        #print(ti, merged_mask.shape, out_labels.shape, detection.shape, len(matched_idxes_low))

        return merged_mask, valid_instances
    
    def _add_detections(self, img, detection, valid_instances, max_tracklets):

        num_instances = min(detection.shape[0], max_tracklets)

        merged_mask = torch.zeros((num_instances,)+self.prob[0].shape[1:], device=self.prob[0].device)

        # Handle (new) instances detected in the second frame
        # We match using lower thresholds to prevent over-initializing
        curr_instance_idx = 0
        for label_idx in range(num_instances):
            merged_mask[curr_instance_idx] = detection[label_idx]
            curr_instance_idx += 1
            valid_instances.append(torch.tensor(self.total_instances+1, device=self.device))
            self.feat_hist.append(collections.deque(maxlen=10))

            _, new_box_feats = self._refine_boxes(img, Boxes(masks_to_boxes(detection[label_idx][0]).unsqueeze(0))) # [1, C]
            self.feat_hist[self.total_instances].append(new_box_feats.squeeze(0))
            self.new_instance_ids.append(self.total_instances+1)

            self.total_instances += 1

        return merged_mask, valid_instances
    
    def _reid(self):

        reid_count = 0

        # Do Re-ID
        if len(self.new_instance_ids) > 0 and len(self.reid_instance_feats) > 0:

            no_longer_new = []
            # Aggregate new instance feats
            new_box_feat_ids = []
            new_box_feats = []
            for new_instance_id in self.new_instance_ids:
                new_box_feats.append(torch.stack(list(self.feat_hist[new_instance_id-1])))
                new_box_feat_ids.append(torch.full((len(self.feat_hist[new_instance_id-1]),), new_instance_id))

                if len(self.feat_hist[new_instance_id-1]) == 10:
                    no_longer_new.append(new_instance_id)
            new_box_feat_ids = torch.cat(new_box_feat_ids) # [NEW]
            new_box_feats = torch.cat(new_box_feats).unsqueeze(1) # [NEW, 1, C]
            new_box_feats = new_box_feats / torch.norm(new_box_feats, dim=-1, keepdim=True)
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
            killed_box_feats = torch.cat(killed_box_feats).unsqueeze(0) # [1, KILLED, C]
            killed_box_feats = killed_box_feats / torch.norm(killed_box_feats, dim=-1, keepdim=True)

            # Calculate feature differences
            diffs = torch.norm(new_box_feats - killed_box_feats, dim=-1) # [NEW, KILLED]

            for idx, diff in enumerate(diffs):
                min_diff, min_diff_idx = torch.min(diff, dim=0)
                if min_diff < 0.5:
                    new_id = new_box_feat_ids[idx].item()
                    old_id = killed_box_feat_ids[min_diff_idx].item()
                    # Preent dupicate entries, since it's possible that multiple pairs of feats get matched at once
                    if new_id not in self.reid_instance_mappings.keys():
                        self.reid_instance_mappings[new_id] = old_id
                        print("REID", new_id, "->", old_id)
                    # Remove re-ided entries
                    if new_id in self.new_instance_ids:
                        # TODO: Check why this fails on kite-surf
                        self.new_instance_ids.remove(new_id)
                    for entry in self.reid_instance_feats:
                        if entry["id"].item() == old_id:
                            self.reid_instance_feats.remove(entry)

                    reid_count += 1
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
        matched_idxes = matched_idxes[torch.nonzero(matched_score>iou_thresh).flatten()] # [N, 2] or [2]
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
    
    '''def _find_nonoverlapping_masks(self, detected, gt, thresh=0.1):

        # Detected: [N, 1, H, W]
        # GT: [N2, 1, H, W]

        # Convert to boolean masks
        detected = (detected.float() > 0.5).bool() # [N1, 1, H, W]
        gt = (gt.float() > 0.5).bool().permute(1,0,2,3) # [1, N2, H, W]

        intersection = torch.logical_and(detected, gt).float().sum((-2, -1))
        instance_size = detected.float().sum((-2, -1))
        
        iou = (intersection + 1e-6) / (instance_size + 1e-6) # [N1, N2]
        thresholded = iou # When we add semantic label thresholding
        row_idx, col_idx = linear_sum_assignment(-thresholded.cpu()) # Score -> cost
        matched_idxes = torch.stack([torch.from_numpy(row_idx), torch.from_numpy(col_idx)], axis=1)
        matched_score = iou[row_idx, col_idx]
        matched_idxes = matched_idxes[torch.nonzero(matched_score>thresh).flatten()] # [N, 2] or [2]
        # This happens when we only have one pair of masks matched
        # It makes subsequent functions so we unsqueeze for an extra dimension
        if len(matched_idxes.shape) == 1:
            matched_idxes = matched_idxes[None, :]

        return matched_idxes'''

    def _propagate_boxes(self, sampled_pts, fwd_flow, orig_box):

        # sampled_pts: [K, 2]
        # flow: [H, W, 2], (x, y) format
        # orig_box: [N, 4]

        H, W, _ = fwd_flow.shape

        # Propagate points based on flow
        sampled_flow = fwd_flow[sampled_pts[:, 0], sampled_pts[:, 1]].to(self.device)
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
    
    def _refine_boxes(self, img, boxes):

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

        out_boxes = Boxes(roi_dict["boxes"][0].detach())
        out_boxes.scale(inputs[0]["width"]/images.image_sizes[0][1],
                        inputs[0]["height"]/images.image_sizes[0][0])
        out_boxes.clip(images.image_sizes[0])

        box_features = [features[f] for f in self.detector.roi_heads.box_in_features]
        box_features = self.detector.roi_heads.box_pooler(box_features, [boxes])
        pooled_box_features = torch.mean(box_features, dim=(-2, -1))

        return out_boxes.tensor, pooled_box_features
    
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