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
        self.valid_instances = [torch.tensor([0])] * self.t
        self.total_instances = 0

        # Re-ID instance mappings
        self.feat_hist = []
        self.reid_instance_feats = []
        self.new_instance_feats = []
        self.new_instance_ids = []
        self.reid_instance_mappings = collections.OrderedDict()

        #
        self.back_prob = [ [] for _ in range(self.t) ]
        self.back_valid_instances = [ [] for _ in range(self.t) ]
        self.do_backwards_instances = []

    def do_pass(self, idx, end_idx, fwd_flow, bwd_flow, detected, max_tracklets=100):

        # First get box feats at start_idx
        start_boxes = []
        for instance_mask in self.prob[idx]:
            self.feat_hist.append(collections.deque(maxlen=10))
            instance = instance_mask[0]
            orig_box = masks_to_boxes(instance)
            start_boxes.append(orig_box)
        start_boxes = torch.stack(start_boxes)
        _, start_box_feats = self._refine_boxes(self.images[idx].numpy(), Boxes(start_boxes.cuda()))
        for i, start_box_feat in enumerate(start_box_feats):
            self.feat_hist[i].append(start_box_feat)

        closest_ti = end_idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)
        end = closest_ti - 1

        all_boxes = []

        for ti in this_range:

            boxes = []
            valid_instances = []

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
                for inst_id, box_feat in enumerate(box_feats):
                    self.feat_hist[self.valid_instances[ti-1][inst_id]-1].append(box_feat)

                all_boxes.append(boxes)

                # Prompt SAM for seg masks
                self.prop_net.set_image(self.images[ti].numpy().astype(np.uint8))
                transformed_boxes = self.prop_net.transform.apply_boxes_torch(boxes.to(self.prop_net.device),
                                                                            self.images[ti].shape[:2])
                out_masks, _, _ = self.prop_net.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                #print("SINGLE", out_masks.shape)
                '''out_masks, scores, _ = self.prop_net.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=True,
                )'''
                #print("MULTI", out_masks.shape, scores)
                #out_masks = out_masks[torch.arange(out_masks.shape[0]), torch.argmax(scores, dim=1)].unsqueeze(1)
                #print("INDEX", out_masks.shape)
                #input()
                #out_masks = out_masks[:, 2].unsqueeze(1)

            # Merge with detections
            detected_ti = detected[ti][0].cuda()
            if len(detected_ti) > 0:
                if out_masks.shape[0] > 0:
                    merged_mask, valid_instances, new_ids = self._merge_detections_and_propagations(self.images[ti].numpy(),
                                                                                           detected_ti,
                                                                                           out_masks,
                                                                                           valid_instances,
                                                                                           max_tracklets)
                else:
                    merged_mask, valid_instances, new_ids = self._add_detections(self.images[ti].numpy(),
                                                                        detected_ti,
                                                                        valid_instances,
                                                                        max_tracklets)
            else:
                merged_mask = out_masks
                new_ids = []

            # For the new tracklets, add them to the list of instances that requires backwards propagation
            if len(new_ids) > 0:
                curr_ids = [entry["id"] for entry in self.do_backwards_instances]
                for new_id in new_ids:
                    if new_id not in curr_ids:
                        self.do_backwards_instances.append({"id": new_id, "t": ti})
            
            # Do Re-ID
            _ = self._reid()

            self.prob[ti] = merged_mask
            if len(valid_instances) > 0:
                self.valid_instances[ti] = torch.stack(valid_instances)
            #print("MERGED", ti, merged_mask.shape, self.valid_instances[ti])

        print("BACK INST", self.do_backwards_instances)
        # Now do backwards prop
        for entry in self.do_backwards_instances:
            print("VALID", self.valid_instances[entry["t"]], "AT", entry["t"], "FIND", entry["id"])
            print("IDX", (self.valid_instances[entry["t"]]==entry["id"]).nonzero(as_tuple=True)[0][0], "AT", entry["t"])
            print(self.prob[entry["t"]][(self.valid_instances[entry["t"]]==entry["id"]).nonzero(as_tuple=True)[0][0]].shape)
            self.back_prob[entry["t"]].append(self.prob[entry["t"]][(self.valid_instances[entry["t"]]==entry["id"]).nonzero(as_tuple=True)[0][0]])
            self.back_valid_instances[entry["t"]].append(entry["id"])
        for ti, prob in enumerate(self.back_prob):
            if len(prob) > 0:
                self.back_prob[ti] = torch.stack(prob)

        this_range = reversed(range(idx+1, closest_ti-1))

        for ti in this_range:

            boxes = []
            if len(self.back_prob[ti]) > 0:

                for prob_id, instance_mask in enumerate(self.back_prob[ti]):

                    instance = instance_mask[0]

                    orig_box = masks_to_boxes(instance)

                    # Skip instance and kill tracklet if needed
                    is_consistent, prop_idxes = self._calc_flow_consistency(instance, bwd_flow[ti-1], fwd_flow[ti-1])
                    if not is_consistent:
                        print(ti, "[BW] Killing", self.back_valid_instances[ti][prob_id])
                        continue

                    # Use optical flow to propagate the box
                    idxes = torch.nonzero(instance)
                    idxes_valid = instance[prop_idxes[:, 0], prop_idxes[:, 1]]
                    # Filter based on valid mask
                    idxes = idxes[torch.nonzero(idxes_valid).squeeze(1)]
                    sampled_pts = idxes.to(self.device)
                    # Warp boxes
                    warped_box = self._propagate_boxes(sampled_pts, bwd_flow[ti-1], orig_box)
                    boxes.append(warped_box)
                    self.back_valid_instances[ti-1].append(self.back_valid_instances[ti][prob_id])

                # If no instances are consistent, then we don't bother prompting with this frame
                if len(boxes) > 0:
                    # Refine boxes using box regressor
                    boxes = torch.stack(boxes)
                    boxes, box_feats = self._refine_boxes(self.images[ti-1].numpy(), Boxes(boxes.cuda()))
                    boxes = torch.round(boxes).long()

                    all_boxes.append(boxes)

                    # Prompt SAM for seg masks
                    self.prop_net.set_image(self.images[ti-1].numpy().astype(np.uint8))
                    transformed_boxes = self.prop_net.transform.apply_boxes_torch(boxes.to(self.prop_net.device),
                                                                                self.images[ti-1].shape[:2])
                    out_masks, _, _ = self.prop_net.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )
                
                    if len(self.back_prob[ti-1]) == 0:
                        self.back_prob[ti-1] = out_masks
                    else:
                        self.back_prob[ti-1] = torch.cat([self.back_prob[ti-1], out_masks], dim=0)

        return closest_ti, all_boxes

    def interact(self, mask, frame_idx, end_idx, fwd_flow, bwd_flow, detected):

        # Init frame 0
        self.prob[frame_idx] = mask.to(self.device)
        self.valid_instances[frame_idx] = torch.arange(mask.shape[0], device=self.device) + 1
        self.total_instances = mask.shape[0]

        # Track from frame 1 ~ end
        _, boxes = self.do_pass(frame_idx, end_idx, fwd_flow.to(self.device), bwd_flow.to(self.device), detected)

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
        matched_idxes_low = self._find_nonoverlapping_masks(propagation, detection, thresh=0.1)

        # Figure out how many instances there are in the two frames
        num_instances = propagation.shape[0] + detection.shape[0] - len(matched_idxes_low)
        num_instances = max(num_instances, propagation.shape[0])
        new_ids = []

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
                if label_idx not in matched_idxes_low[:, 1]:
                    merged_mask[curr_instance_idx] = detection[label_idx]
                    curr_instance_idx += 1
                    valid_instances.append(torch.tensor(self.total_instances+1, device=self.device))
                    self.feat_hist.append(collections.deque(maxlen=10))

                    _, new_box_feats = self._refine_boxes(img, Boxes(masks_to_boxes(detection[label_idx][0]).unsqueeze(0))) # [1, C]
                    self.feat_hist[self.total_instances].append(new_box_feats.squeeze(0))
                    self.new_instance_ids.append(self.total_instances+1)
                    new_ids.append(self.total_instances+1) # Return the new ids generated ONLY at this function call

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

        return merged_mask, valid_instances, new_ids
    
    def _add_detections(self, img, detection, valid_instances, max_tracklets):

        # Figure out how many instances there are in the two frames
        num_instances = min(detection.shape[0], max_tracklets)
        new_ids = []

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
            new_ids.append(self.total_instances+1) # Return the new ids generated ONLY at this function call

            self.total_instances += 1

        return merged_mask, valid_instances, new_ids
    
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
                    self.reid_instance_mappings[new_id] = old_id
                    # Preent dupicate entries, since it's possible that multiple pairs of feats get matched at once
                    if new_id not in self.reid_instance_mappings.keys():
                        self.reid_instance_mappings[new_id] = old_id
                        print("REID", new_id, "->", old_id)
                    # Remove re-ided entries
                    if new_id in self.new_instance_ids:
                        # TODO: Check why this fails on kite-surf
                        self.new_instance_ids.remove(new_id)
                    for entry in self.do_backwards_instances:
                        if entry["id"] == new_id:
                            self.do_backwards_instances.remove(entry)
                    for entry in self.reid_instance_feats:
                        if entry["id"].item() == old_id:
                            self.reid_instance_feats.remove(entry)

                    reid_count += 1
            # If the entry has persisted long enough before getting re-ided, don't try to re-id it
            for entry in no_longer_new:
                if entry in self.new_instance_ids:
                    self.new_instance_ids.remove(entry)
        
        return reid_count
    
    def _find_matching_masks(self, detected, gt, iou_thresh=0.8):

        # Detected: [N, 1, H, W]
        # GT: [N2, 1, H, W]

        # Convert to boolean masks
        detected = (detected.float() > 0.5).bool() # [N1, 1, H, W]
        gt = (gt.float() > 0.5).bool().permute(1,0,2,3) # [1, N2, H, W]

        intersection = torch.logical_and(detected, gt).float().sum((-2, -1))
        union = torch.logical_or(detected, gt).float().sum((-2, -1))
        
        iou = (intersection + 1e-6) / (union + 1e-6) # [N1, N2]
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
    
    def _find_nonoverlapping_masks(self, detected, gt, thresh=0.1):

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

        return matched_idxes

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

        #min_y = torch.min(warped_pts[:, 0])
        #max_y = torch.max(warped_pts[:, 0])
        #min_x = torch.min(warped_pts[:, 1])
        #max_x = torch.max(warped_pts[:, 1])

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