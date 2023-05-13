import time
import os
import numpy as np
import random
import flyingthingsdataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from fire import Fire

random.seed(125)
np.random.seed(125)

import sys
from gmflow.gmflow import GMFlow

# from filter_trajs import filter_trajs

import traj_linker
linker = traj_linker.Linker()

if is_train:
    cur_out_dir = '/data/flyingthings/gmflow_river_%s/train' % mod
else:
    cur_out_dir = '/data/flyingthings/gmflow_river_%s/val' % mod
if not os.path.exists(cur_out_dir):
    os.makedirs(cur_out_dir)

def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ


def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def run_model(gmflow, d, sw=None, device_ids=[0], iters=8):
    device = 'cuda:%d' % device_ids[0]
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    metrics = {}

    out_f = '%s/%06d.npz' % (cur_out_dir, sw.global_step)
    if os.path.isfile(out_f) and not do_overwrite:
        sys.stdout.write(':')
        return total_loss, metrics

    rgbs = d['rgbs'].to(device).float() # B, S, C, H, W
    trajs_g = d['trajs'].to(device).float() # B, S, N, 2
    vis_g = d['visibles'].to(device).float() # B, S, N
    valids = d['valids'].to(device).float() # B, S, N

    B, S, C, H, W = rgbs.shape
    assert(C==3)
    B, S, N, D = trajs_g.shape

    assert(torch.sum(valids)==B*S*N)

    assert(B==1)

    if True:
        rgbs_ = rgbs.reshape(B*S, 3, H, W)
        scale_factor = 0.5
        rgbs_ = F.interpolate(rgbs_, scale_factor=0.5)
        H = int(H*scale_factor)
        W = int(W*scale_factor)
        rgbs = rgbs_.reshape(B, S, 3, H, W)
        # trajs_g[:,:,:,0] *= scale_factor
        trajs_g *= scale_factor
        # trajs_g[:,0] = trajs_g[:,0].round()
        trajs_g[:,0] = trajs_g[:,0].floor()

    start_time = time.time()

        
    flow_fw_vis = []
    flow_bw_vis = []
    occ_fw_vis = []
    occ_bw_vis = []

    flow_fws = []
    flow_bws = []
    occ_fws = []
    occ_bws = []
    for s in range(S-1):
        s0 = s
        s1 = s+1
        results_dict = gmflow(
            rgbs[:,s0], rgbs[:,s1],
            attn_splits_list=[2],
            corr_radius_list=[-1],
            prop_radius_list=[-1],
            pred_bidir_flow=True,
        )
        flow_pr = results_dict['flow_preds'][-1]  # [2, 2, H, W]
        # print_stats('flow_pr', flow_pr)

        flow_fw = flow_pr[0:1]
        flow_bw = flow_pr[1:2]

        flow_fws.append(flow_fw)
        flow_bws.append(flow_bw)

        # occ_fw, occ_bw = forward_backward_consistency_check(flow_fw, flow_bw)
        # occ_fw = occ_fw.unsqueeze(1)
        # occ_bw = occ_bw.unsqueeze(1)

        # occ_fws.append(occ_fw)
        # occ_bws.append(occ_bw)
    
        # print_stats('occ_fw', occ_fw)
        
        if sw is not None and sw.save_this:
            flow_fw_vis.append(sw.summ_flow('', flow_fw, clip=50.0, only_return=True))
            flow_bw_vis.append(sw.summ_flow('', flow_bw, clip=50.0, only_return=True))
            # occ_fw_vis.append(sw.summ_oned('', occ_fw, norm=False, only_return=True))
            # occ_bw_vis.append(sw.summ_oned('', occ_bw, norm=False, only_return=True))

    stage_time = time.time()-start_time
    print('flow %.2f' % (stage_time))
    start_time = time.time()
            

    # for s in range(S-2):
    #     flow_01 = flow_fws[s]
    #     flow_12 = flow_fws[s+1]
    #     flow_12_in_0 = flow_warp(flow_12, flow_01)
    #     flow_02 = flow_01 + flow_12_in_0

    #     flow_21 = flow_bws[s+1]
    #     flow_10 = flow_bws[s]
    #     flow_10_in_2 = flow_warp(flow_10, flow_21)
    #     flow_20 = flow_21 + flow_10_in_2
        
    #     # flow_21 = flow_warp(flow_12, flow_01)
    #     # flow_02 = flow_01 + flow_12

    #     # flow_20_in_0 = flow_warp(flow_20, flow_02)

    #     occ_fw, occ_bw = forward_backward_consistency_check(flow_02, flow_20)
    #     vis_ok = (occ_fw==0).reshape(H*W)
    #     grid_y, grid_x = utils.basic.meshgrid2d(B, H, W) # B, H, W
    #     coord0 = torch.stack([grid_x, grid_y], dim=1) # B, 2, H, W
    #     coord1 = coord0 + flow_01
    #     coord2 = coord0 + flow_02
    #     triplet = torch.stack([coord0.reshape(B, 2, -1),
    #                            coord1.reshape(B, 2, -1),
    #                            coord2.reshape(B, 2, -1)], dim=3) # B,2,HW,3
    #     triplet = triplet.permute(0,3,2,1) # B,S,N,2
    #     triplet = triplet[:,:,vis_ok]

    #     end = triplet[0,-1] # N,2
    #     inb = (end[:,0] >= 0).byte() & (end[:,0] < W-1).byte() & (end[:,1] >= 0).byte() & (end[:,1] < H-1).byte()
    #     # print('inb', inb.shape)
    #     triplet = triplet[:,:,inb]

    #     if False:
    #         # triplet = 
    #         # triplet = triplet.permute(0,3,2,1) # B,S,N,2
    #         # triplet = triplet[:,:,vis_ok]
    #         # vis_ok = 




    #         grid_y, grid_x = utils.basic.meshgrid2d(B, H, W) # B, H, W
    #         coord1 = torch.stack([grid_x, grid_y], dim=1) # B, 2, H, W
    #         coord0 = coord1 + flow_10
    #         coord2 = coord1 + flow_12
    #         triplet = torch.stack([coord0.reshape(B, 2, -1),
    #                                coord1.reshape(B, 2, -1),
    #                                coord2.reshape(B, 2, -1)], dim=3) # B,2,HW,3




    #         flow_10 = flow_bws[s]
    #         flow_12 = flow_fws[s+1]
    #         grid_y, grid_x = utils.basic.meshgrid2d(B, H, W) # B, H, W
    #         coord1 = torch.stack([grid_x, grid_y], dim=1) # B, 2, H, W
    #         coord0 = coord1 + flow_10
    #         coord2 = coord1 + flow_12
    #         triplet = torch.stack([coord0.reshape(B, 2, -1),
    #                                coord1.reshape(B, 2, -1),
    #                                coord2.reshape(B, 2, -1)], dim=3) # B,2,HW,3
    #         assert(B==1)
    #         vis_ok = (occ_fws[s]==0).reshape(H*W)
    #         triplet = triplet.permute(0,3,2,1) # B,S,N,2
    #         triplet = triplet[:,:,vis_ok]

    #     N = triplet.shape[2]
    #     print('s=%d; N=%d' % (s,N))
    #     if sw is not None and sw.save_this:
    #         inds = utils.misc.farthest_point_sample(triplet[:,0], 256)
    #         # perm = np.random.permutation(N)
    #         # triplet_ = triplet[:,:,perm[:256]]
    #         triplet_ = triplet[:,:,inds.reshape(-1)]
    #         rgb_ = utils.improc.preprocess_color(rgbs[:,s:s+2]).mean(dim=1)
            
    #         gt_rgb = utils.improc.preprocess_color(sw.summ_traj2ds_on_rgb('', trajs_g[0:1,s:s+3], torch.mean(utils.improc.preprocess_color(rgbs[0:1,s:s+3]), dim=1), linewidth=1, cmap='winter', only_return=True))
    #         # sw.summ_traj2ds_on_rgb('2_triplet/triplet_%d' % s, triplet, rgb_, linewidth=1, cmap='spring')
    #         sw.summ_traj2ds_on_rgb('2_triplet/triplet_%d' % s, triplet_, gt_rgb, linewidth=1, cmap='spring')
    #         sw.summ_traj2ds_on_rgbs('2_triplet/triplet_%d_on_rgbs' % s, triplet_, utils.improc.preprocess_color(rgbs[:,s:s+3]), linewidth=1, cmap='spring')

    trajs_XYs, trajs_Ts = linker(rgbs,
                                 torch.stack(flow_fws, dim=1),
                                 torch.stack(flow_bws, dim=1),
                                 sample_step=4,
                                 mult_thresh=0.05,
                                 add_thresh=0.5,
                                 min_lifespan=4) # each B x N_p x 2
    stage_time = time.time()-start_time
    print('link %.2f' % (stage_time))
    start_time = time.time()

    # trajs_XYs, trajs_Ts = utils.misc.get_filtered_trajs(
    #     trajs_XYs, trajs_Ts,
    #     min_lifespan=3,
    # )
    # stage_time = time.time()-start_time
    # print('filter %.2f' % (stage_time))
    # start_time = time.time()

    print('trajs_XYs', len(trajs_XYs))
    print('trajs_XYs[0]', trajs_XYs[0].shape)
    print('trajs_Ts[0]', trajs_Ts[0].shape)

    np_trajs_XYs = []
    np_trajs_Ts = []
    for i in range(len(trajs_XYs)):
        XYs = trajs_XYs[i].detach().cpu().numpy().astype(np.float16)
        Ts = trajs_Ts[i].detach().cpu().numpy().astype(np.float16)
        np_trajs_XYs.append(XYs)
        np_trajs_Ts.append(Ts)

    np_rgbs = rgbs[0].byte().detach().cpu().numpy().astype(np.uint8)
    np_trajs_g = trajs_g[0].detach().cpu().numpy().astype(np.float16)
    np_vis_g = vis_g[0].detach().cpu().numpy().astype(np.float16)

    np.savez_compressed(
        out_f,
        trajs_XYs=np_trajs_XYs,
        trajs_Ts=np_trajs_Ts,
        rgbs=np_rgbs,
        # img_names=img_names,
        # cur_rgb_path=cur_rgb_path,
        trajs_g=np_trajs_g,
        vis_g=np_vis_g,
    )
    sys.stdout.write('.')
        
    return total_loss, metrics
    
def main(
        exp_name='debug',
        # training
        B=1, # batchsize 
        S=32, # seqlen of the data/model
        N=256, # number of particles to sample from the data
        crop_size=(384,512), # the raw data is 540,960
        use_augs=True, # resizing/jittering/color/blur augs
        dataset_location='/data/flyingthings',
        subset='all', # dataset subset
        shuffle=False, # dataset shuffling
        # optimization
        lr=3e-4,
        grad_acc=1,
        max_iters=10,
        use_scheduler=True,
        # summaries
        log_dir='./logs_export_gmflow_river',
        log_freq=999999,
        fps=4,
        # saving/loading
        ckpt_dir='./checkpoints',
        save_freq=999999999,
        keep_latest=1,
        init_dir='',
        load_optimizer=False,
        load_step=False,
        ignore_load=None,
        # cuda
        device_ids=[0],
):

    
    device = 'cuda:%d' % device_ids[0]
        
    assert(crop_size[0] % 128 == 0)
    assert(crop_size[1] % 128 == 0)
    
    ## autogen a descriptive name
    B_ = B
    model_name = "%d" % B_
    if grad_acc > 1:
        model_name += "x%d" % grad_acc
    model_name += "_%d_%d" % (S, N)
    # model_name += "_I%d" % (I)
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    model_name += "_%s" % lrn
    if use_augs:
        model_name += "_A"
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)
    
    ckpt_dir = '%s/%s' % (ckpt_dir, model_name)

    gmflow_model = GMFlow(
        feature_channels=128,
        num_scales=1,
        upsample_factor=8,
        num_head=1,
        attention_type='swin',
        ffn_dim_expansion=4,
        num_transformer_layers=6,
    ).to(device)

    gmflow_model = torch.nn.DataParallel(gmflow_model, device_ids=device_ids)
    gmflow_model_without_ddp = gmflow_model.module
    checkpoint = torch.load('./pretrained/gmflow_with_refine_sintel-3ed1cf48.pth', map_location=device)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    gmflow_model_without_ddp.load_state_dict(weights, strict=False)
    gmflow_model.eval()
    print('loaded gmflow')

    while global_step < max_iters:
        
        read_start_time = time.time()
        
        global_step += 1
        total_loss = torch.tensor(0.0, requires_grad=True).to(device)

        gotit = (False,False)
        while not all(gotit):
            try:
                sample, gotit = next(the_iterloader)
            except StopIteration:
                the_iterloader = iter(the_dataloader)
                sample, gotit = next(the_iterloader)
            if not all(gotit):
                print('sampling failed')

        read_time = time.time()-read_start_time
        iter_start_time = time.time()

        total_loss, metrics = run_model(gmflow_model, sample, sw=sw_t, device_ids=device_ids)
        
        iter_time = time.time()-iter_start_time
        
        print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
            model_name, global_step, max_iters, read_time, iter_time))

if __name__ == '__main__':
    Fire(main)