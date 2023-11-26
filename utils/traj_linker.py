import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sampling import backwarp_using_2d_flow, bilinear_sample2d

class Linker(nn.Module):
    def __init__(self):
        super(Linker, self).__init__()

    def meshgrid2d(self, B, Y, X, sample_step=1, margin=0, device='cpu'):

        # return grid_y, grid_x
        grid_y = torch.arange(margin, Y-margin+1, sample_step, device=device, dtype=torch.float32)
        Y_stepped = grid_y.shape[0]
        grid_y = torch.reshape(grid_y, [1, Y_stepped, 1])

        grid_x = torch.arange(margin, X-margin+1, sample_step, device=device, dtype=torch.float32)
        X_stepped = grid_x.shape[0]
        grid_x = torch.reshape(grid_x, [1, 1, X_stepped])

        grid_y = grid_y.repeat(B, 1, X_stepped) # B x Y x X
        grid_x = grid_x.repeat(B, Y_stepped, 1)

        return grid_y, grid_x

    def prune_flow_field(self, flow_f, flow_b, mult_thresh=0.01, add_thresh=0.1, device='cpu'):
        B, C, H, W = flow_f.shape

        flow_b_at_target_loc = backwarp_using_2d_flow(flow_b, flow_f)
        diff = torch.norm((flow_f + flow_b_at_target_loc), dim=1) # B x H x W

        flow_mag = 0.5*(torch.norm(flow_f, dim=1) + torch.norm(flow_b_at_target_loc, dim=1))  # [B, H, W]
        thr = mult_thresh * flow_mag + add_thresh
        off = (diff > thr).float()
        on = 1.0 - off
        return on

    def terminate_at_t(self, inds_end, s, Start, X, Y, min_lifespan=3):
        
        # inds_end are the trajs to be terminated
        # inds_end is len-N
        # s is a scalar, indicating the current timestep
        traj_XYs = []
        traj_Ts = []

        for ind_end in inds_end:
            t_start = Start[ind_end] # the starting frame of this traj
            t_end = s
            time_interval = torch.arange(t_start, t_end+1, device=X.device, dtype=torch.int64)
            if t_end-t_start >= min_lifespan:
                traj_Ts.append(time_interval)
                traj_XYs.append(torch.stack([X[time_interval, ind_end], Y[time_interval, ind_end]], dim=-1))
        return traj_XYs, traj_Ts

    def forward(self,
                flows_f, flows_b,
                margin=10, sample_step=4,
                valids=None,
                born=True,
                mult_thresh=0.01,
                add_thresh=0.1,
                min_lifespan=3,
                use_bilinear=True):
        # rgbs is B x S x 3 x H x W
        # flows_* is B x (S-1) x 2 x H x W
        B, S, C, H, W = flows_f.shape
        device = flows_f.device

        ys, xs = self.meshgrid2d(B, H, W, sample_step, margin, device=device) # B x H x W H->Y, W->X

        xs = xs.reshape(-1)
        ys = ys.reshape(-1)

        if valids is not None:
            valid_check = bilinear_sample2d(valids[:,0], xs.unsqueeze(0), ys.unsqueeze(0)).squeeze(1) # B x N
            print('valid_check', valid_check.shape)
            inds = (valid_check > 0.5).reshape(-1)
            print('inds', inds.shape)
            xs = xs[inds]
            ys = ys[inds]
        
        X = torch.zeros((S, 2*H*W//(sample_step**2)), device=device) # store all points
        Y = torch.zeros((S, 2*H*W//(sample_step**2)), device=device)
        Start = -torch.ones(2*H*W//(sample_step**2), device=device) # initialize as -1, means invalid

        X[0, :len(xs)] = xs # initalize
        Y[0, :len(ys)] = ys #
        Start[:len(ys)] = 0 # start at frame 0

        trajs_XYs = []
        trajs_Ts = []            

        for s in range(S-1):
            flow_f = flows_f[:, s] # B x 2 x H x W
            flow_b = flows_b[:, s] # B x 2 x H x W

            if valids is not None:
                valid0 = valids[:, s] # B x 1 x H x W

            # filter pts based on flow filtering
            on = self.prune_flow_field(flow_f, flow_b, mult_thresh=mult_thresh, add_thresh=add_thresh, device=device) # B x H x W
            
            inds = torch.where(Start >= 0)[0]
            xs_ori = X[s:s+1, inds] # B x N_p
            ys_ori = Y[s:s+1, inds] # B x N_p

            if use_bilinear:
                uv = bilinear_sample2d(flow_f, xs_ori, ys_ori) # B x 2 x N, forward flow at the discrete points
            else:
                uv = flow_f[0:1,:,ys_ori.squeeze().long(), xs_ori.squeeze().long()] # B x 2 x N, forward flow at the discrete points

            u = uv[:, 0] # B x N
            v = uv[:, 1]
            
            xs_tar = xs_ori + u # B x N_p
            ys_tar = ys_ori + v

            if use_bilinear:
                fb_check = bilinear_sample2d(on.unsqueeze(1), xs_ori, ys_ori).squeeze(1) # B x N
            else:
                fb_check = on[:,ys_ori.squeeze().long(), xs_ori.squeeze().long()] # B x N
            
            if valids is not None:
                valid_check = bilinear_sample2d(valid0, xs_ori, ys_ori).squeeze(1) # B x N

            margin_check = ((xs_tar > margin) &
                            (ys_tar > margin) &
                            (xs_tar < W - margin) &
                            (ys_tar < H - margin)) # choose inbound pts & pass forward-backward check
            if valids is not None:
                choose = margin_check & (fb_check > 0.5) & (valid_check > 0.5)
            else:
                choose = margin_check & (fb_check > 0.5)
            choose = choose.squeeze(0) # N_p
            inds_on = inds[choose]
            inds_off = inds[~choose]

            X[s+1, inds_on] = xs_tar[0, choose]
            Y[s+1, inds_on] = ys_tar[0, choose]
            
            # terminate
            traj_XYs, traj_Ts = self.terminate_at_t(inds_off, s, Start, X, Y, min_lifespan=min_lifespan)
            trajs_XYs.extend(traj_XYs) # each element is len_history x 2
            trajs_Ts.extend(traj_Ts) # each element is len_history
            Start[inds_off] = -1

            # sample new points at t+1-> cover with 0 the area to be sampled and 1 the area to be left untouched
            map_occupied = torch.zeros((B, 1, H, W), device=device)
            map_occupied[:, :, ys_tar[0, choose].long(), xs_tar[0, choose].long()] = 1.0
            img_dilation_kernel = torch.ones((1, 1, 5, 5), device=device)
            map_occupied = F.conv2d(map_occupied, img_dilation_kernel, padding=2)
            map_occupied = (map_occupied > 0.0).float() # B x 1 x H x W

            if use_bilinear:
                map_occ = bilinear_sample2d(map_occupied, xs.unsqueeze(0), ys.unsqueeze(0)).squeeze()
            else:
                map_occ = map_occupied[0,0,ys.squeeze().long(),xs.squeeze().long()]
                
            map_free = 1.0 - map_occ
            if valids is not None:
                valid_ok = bilinear_sample2d(valid0, xs.unsqueeze(0), ys.unsqueeze(0)).squeeze()
                map_free = map_free * valid_ok
            ind_map_free = map_free==1.0
            xs_added = xs[ind_map_free]
            ys_added = ys[ind_map_free]

            if born: # add borning of new points
                num_added = len(xs_added)
                free_inds = torch.where(Start < 0)[0] # we re-use slots in Start
                min_added = min(num_added, len(free_inds))
                X[s+1, free_inds[:min_added]] = xs_added[:min_added]
                Y[s+1, free_inds[:min_added]] = ys_added[:min_added]
                Start[free_inds[:min_added]] = s+1
       
        # terminate all remaining trajs
        inds_on = torch.where(Start >= 0)[0]
        traj_XYs, traj_Ts = self.terminate_at_t(inds_on, S-1, Start, X, Y, min_lifespan=min_lifespan)
        trajs_XYs.extend(traj_XYs) # each element is len_history x 2
        trajs_Ts.extend(traj_Ts) # each element is len_history

        return trajs_XYs, trajs_Ts

if __name__ == "__main__":
    net = Linker()
    flow_f = torch.zeros((1, 9, 2, 64, 32)).cuda()
    flow_b = torch.zeros((1, 9, 2, 64, 32)).cuda()
    trajs_XYs, trajs_Ts = net(flow_f, flow_b)
    print(len(trajs_XYs), trajs_XYs[0].shape)
    print(len(trajs_Ts), trajs_Ts[0].shape, trajs_Ts)







