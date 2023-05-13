import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from segment_anything.utils.transforms import ResizeLongestSide
from utils.morphology import dilation, erosion
from utils.flow import forward_backward_consistency_check

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

class Propagator:
    def __init__(self, prop_net, images, num_objects):
        self.prop_net = prop_net
        
        self.k = num_objects

        # True dimensions
        self.t = images.shape[0]

        # Padded dimensions
        nh, nw = images.shape[1:3]

        self.images = images
        self.device = prop_net.device

        self.prob = [torch.zeros((self.k, 1, nh, nw), dtype=torch.float32, device=self.device)] * self.t
        self.prob[0] = 1e-7
        
        # Tells us which instance ids are in self.prob (starts from 1)
        self.valid_instances = [[0]] * self.t
        self.total_instances = 0

    def do_pass(self, idx, end_idx, fwd_flow, bwd_flow, detected,
                num_fg_points=5, num_bg_points=0, max_tracklets=10):

        H, W, _ = fwd_flow[0].shape
        closest_ti = end_idx

        # Note that we never reach closest_ti, just the frame before it
        this_range = range(idx+1, closest_ti)

        all_vis_orig_fg_pts = []
        all_vis_fg_pts = []
        all_vis_orig_bg_pts = []
        all_vis_bg_pts = []
        resize_transform = ResizeLongestSide(self.prop_net.image_encoder.img_size)

        for ti in this_range:
            
            vis_orig_fg_pts = []
            vis_fg_pts = []
            vis_orig_bg_pts = []
            vis_bg_pts = []

            input_fg_pts = []
            input_bg_pts = []

            valid_instances = []

            min_fg_pts = 99999
            min_bg_pts = 99999
            # Select (random) point prompts for each instance
            for prob_id, instance in enumerate(self.prob[ti-1]):
                
                # We do some dilation/erosion so we don't sample points near edges
                dilated = dilation(instance.unsqueeze(0), torch.ones((11, 11), device=self.device))
                extra_dilated = dilation(instance.unsqueeze(0), torch.ones((27, 27), device=self.device))
                eroded = erosion(instance.unsqueeze(0), torch.ones((5, 5), device=self.device))

                # Check if the instance is still valid
                # i.e. if the forward-backward flow goes back to the same instance
                idxes = torch.nonzero(eroded[0].squeeze(0))
                instance_size = len(idxes)
                # If the instance is too small, just kill it
                if instance_size < 50:
                    continue
                flow = fwd_flow[ti-1][idxes[:, 0], idxes[:, 1]] # Note flows are in (x, y) format
                flow = torch.flip(flow, (1,)) # (x, y) -> (y, x) format
                prop_idxes = torch.round(idxes + flow)
                prop_idxes[:, 0] = torch.clamp(prop_idxes[:, 0], 0, H-1)
                prop_idxes[:, 1] = torch.clamp(prop_idxes[:, 1], 0, W-1)
                prop_idxes = prop_idxes.long()
                flow = bwd_flow[ti-1][prop_idxes[:, 0], prop_idxes[:, 1]] # Note flows are in (x, y) format
                flow = torch.flip(flow, (1,)) # (x, y) -> (y, x) format
                prop_idxes = torch.round(prop_idxes + flow)
                prop_idxes[:, 0] = torch.clamp(prop_idxes[:, 0], 0, H-1)
                prop_idxes[:, 1] = torch.clamp(prop_idxes[:, 1], 0, W-1)
                prop_idxes = prop_idxes.long()
                in_instance_after_flow = eroded[0].squeeze(0)[prop_idxes[:, 0], prop_idxes[:, 1]]
                consistent_size = torch.sum(in_instance_after_flow)
                #print(ti, prob_id, consistent_size / instance_size)
                # Skip instance and kill tracklet if needed
                if consistent_size / instance_size < 0.6:
                    continue
                
                # FG points
                # Run the flow linker to get some quality seeds for the FG
                # This checks if applying forward-backward flow puts us back in roughly the same place
                fwd_occ, _ = forward_backward_consistency_check(fwd_flow[ti-1].permute(2,0,1).unsqueeze(0),
                                                                bwd_flow[ti-1].permute(2,0,1).unsqueeze(0))
                fwd_consistent = 1 - fwd_occ[0] # [H, W]
                sampled_pts, orig_pts = self._sample_pts_with_mask_flow(eroded[0].squeeze(0),
                                                                        fwd_flow[ti-1],
                                                                        num_fg_points,
                                                                        valid=fwd_consistent)
                #sampled_pts, orig_pts = self._grid_sample_pts_with_mask_flow(eroded[0].squeeze(0),
                #                                                             fwd_flow[ti-1],
                #                                                             grid_step=5,
                #                                                             valid=fwd_consistent)
                if sampled_pts is not None:
                    vis_orig_fg_pts.append(orig_pts)
                    vis_fg_pts.append(sampled_pts)
                    input_fg_pts.append(torch.flip(sampled_pts, dims=(1,)))
                    min_fg_pts = len(sampled_pts) if len(sampled_pts) < min_fg_pts else min_fg_pts

                    valid_instances.append(self.valid_instances[ti-1][prob_id])
                else:
                    # Just move on for terminated tracklets
                    continue

                # BG points
                if num_bg_points > 0:
                    sampled_pts, orig_pts = self._sample_pts_with_mask_flow(1 - dilated[0].squeeze(0),
                                                                            fwd_flow[ti-1],
                                                                            num_bg_points)
                    #sampled_pts, orig_pts = self._grid_sample_pts_with_mask_flow(1 - dilated[0].squeeze(0),
                    #                                                             fwd_flow[ti-1],
                    #                                                             grid_step=30)
                    vis_orig_bg_pts.append(orig_pts)
                    vis_bg_pts.append(sampled_pts)
                    input_bg_pts.append(torch.flip(sampled_pts, dims=(1,)))
                    min_bg_pts = len(sampled_pts) if len(sampled_pts) < min_bg_pts else min_bg_pts

            # We need to stack these later for SAM, so they must be of the same shape
            for i, fg_pts in enumerate(input_fg_pts):
                input_fg_pts[i] = fg_pts[:min_fg_pts]
            if num_bg_points > 0:
                for i, bg_pts in enumerate(input_bg_pts):
                    input_bg_pts[i] = bg_pts[:min_bg_pts]
            
            # These are used for visualization only
            all_vis_orig_fg_pts.append(vis_orig_fg_pts)
            all_vis_fg_pts.append(vis_fg_pts)
            if num_bg_points > 0:
                all_vis_orig_bg_pts.append(vis_orig_bg_pts)
                all_vis_bg_pts.append(vis_bg_pts)
            
            # Aggregate prompts
            if num_bg_points > 0:
                input_pts = torch.cat([
                    torch.stack(input_fg_pts),
                    torch.stack(input_bg_pts)
                ], dim=1)
                input_labels = torch.cat([
                    torch.ones((len(input_fg_pts), len(input_fg_pts[0])), device=self.prop_net.device),
                    torch.zeros((len(input_bg_pts), len(input_bg_pts[0])), device=self.prop_net.device)
                ], dim=1)
            else:
                input_pts = torch.stack(input_fg_pts)
                input_labels = torch.ones((len(input_fg_pts), len(input_fg_pts[0])), device=self.prop_net.device)
            
            # Prompt SAM for seg masks
            batched_input = [{
                'image': prepare_image(self.images[ti].numpy().astype(np.uint8), resize_transform, self.prop_net),
                'point_coords': resize_transform.apply_coords_torch(input_pts.to(self.prop_net.device),
                                                                    self.images[ti].shape[:2]),
                'point_labels': input_labels,
                'original_size': self.images[ti].shape[:2]
            }]
            output = self.prop_net(batched_input, multimask_output=False)
            out_masks = output[0]['masks']

            # Merge with detections
            detected_ti = detected[ti][0].cuda()
            if len(detected_ti) > 0:
                merged_mask, valid_instances = self._merge_detections_and_propagations(detected_ti,
                                                                                       out_masks,
                                                                                       valid_instances,
                                                                                       max_tracklets)
            else:
                merged_mask = out_masks
 
            self.prob[ti] = merged_mask
            self.valid_instances[ti] = torch.stack(valid_instances)
            print("MERGED", ti, merged_mask.shape, self.valid_instances[ti])

        vis_pts = {
            "fg_pts": all_vis_fg_pts,
            "orig_fg_pts": all_vis_orig_fg_pts,
            "bg_pts": all_vis_bg_pts,
            "orig_bg_pts": all_vis_orig_bg_pts
        }
        return closest_ti, vis_pts

    def interact(self, mask, frame_idx, end_idx, fwd_flow, bwd_flow, detected):

        # Init frame 0
        self.prob[frame_idx] = mask.to(self.device)
        self.valid_instances[frame_idx] = torch.arange(mask.shape[0], device=self.device) + 1
        self.total_instances = mask.shape[0]

        # Track from frame 1 ~ end
        _, vis_pts = self.do_pass(frame_idx, end_idx,
                                  fwd_flow.to(self.device),
                                  bwd_flow.to(self.device),
                                  detected)

        return vis_pts
    
    def _check_flow_consistency(self, mask, fwd_flow, bwd_flow, thresh):
        pass
    
    def _sample_pts_with_mask_flow(self, mask, fwd_flow, num_pts, valid=None):

        # mask: [H, W]
        # flow: [H, W, 2], (x, y) format
        # valid: [H, W] or None

        H, W = mask.shape

        idxes = torch.nonzero(mask)
        # Filter based on valid mask
        if valid is not None:
            idxes_valid = valid[idxes[:, 0], idxes[:, 1]]
            idxes = idxes[torch.nonzero(idxes_valid).squeeze(1)]
        # If we have enough points, then randomly pick some
        if len(idxes) > num_pts:
            perm = torch.randperm(idxes.size(0))
            chosen_idx = perm[:num_pts]
            sampled_pts = idxes[chosen_idx]
            orig_pts = sampled_pts
            # Propagate points based on flow
            sampled_flow = fwd_flow[sampled_pts[:, 0], sampled_pts[:, 1]]
            sampled_flow = torch.flip(sampled_flow, (1,)) # (x, y) -> (y, x)
            warped_pts = torch.round(sampled_pts+sampled_flow)
            # Clamp and convert to long
            warped_pts[:, 0] = torch.clamp(warped_pts[:, 0], 0, H-1)
            warped_pts[:, 1] = torch.clamp(warped_pts[:, 1], 0, W-1)
            warped_pts = warped_pts.long()
        else:
            warped_pts = None
            orig_pts = None

        return warped_pts, orig_pts
    
    def _grid_sample_pts_with_mask_flow(self, mask, fwd_flow, grid_step=20, valid=None):

        # mask: [H, W]
        # flow: [H, W, 2], (x, y) format
        # valid: [H, W] or None

        H, W = mask.shape

        grid_y, grid_x = torch.meshgrid(torch.arange(0, H, grid_step, device=mask.device),
                                        torch.arange(0, W, grid_step, device=mask.device))

        idxes = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)
        # Filter based on valid mask
        if valid is not None:
            mask = mask * valid
        idxes_valid = mask[idxes[:, 0], idxes[:, 1]]
        idxes = idxes[torch.nonzero(idxes_valid).squeeze(1)]
        # If we have enough points, then randomly pick some
        if len(idxes) > 10:
            sampled_pts = idxes
            orig_pts = sampled_pts
            # Propagate points based on flow
            sampled_flow = fwd_flow[sampled_pts[:, 0], sampled_pts[:, 1]]
            sampled_flow = torch.flip(sampled_flow, (1,)) # (x, y) -> (y, x)
            warped_pts = torch.round(sampled_pts+sampled_flow)
            # Clamp and convert to long
            warped_pts[:, 0] = torch.clamp(warped_pts[:, 0], 0, H-1)
            warped_pts[:, 1] = torch.clamp(warped_pts[:, 1], 0, W-1)
            warped_pts = warped_pts.long()
        else:
            warped_pts = None
            orig_pts = None

        return warped_pts, orig_pts
    
    def _merge_detections_and_propagations(self, detection, propagation, valid_instances, max_tracklets):

        # High IOU requirement for mask replacement
        matched_idxes = self._find_matching_masks(propagation, detection)
        # Low IOU requirement for spawning new tracklets
        matched_idxes_low = self._find_nonoverlapping_masks(propagation, detection, thresh=0.1)

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
                if label_idx not in matched_idxes_low[:, 1]:
                    merged_mask[curr_instance_idx] = detection[label_idx]
                    curr_instance_idx += 1
                    valid_instances.append(torch.tensor(self.total_instances+1, device=self.device))
                    print("ADD VALID NEW INSTANCE", valid_instances)
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
        instance_size = gt.float().sum((-2, -1))
        
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