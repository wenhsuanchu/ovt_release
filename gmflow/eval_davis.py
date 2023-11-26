import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import os

from data import build_train_dataset
from gmflow.gmflow import GMFlow
from loss import flow_loss_func
from evaluate import (validate_chairs, validate_things, validate_sintel, validate_kitti,
                      create_sintel_submission, create_kitti_submission, inference_on_dir)

from utils.logger import Logger
from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='chairs', type=str,
                        help='training stage')
    parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                        help='image size for training')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding')

    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')
    parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                        help='validation dataset')
    parser.add_argument('--with_speed_metric', action='store_true',
                        help='with speed metric when evaluation')

    # training
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=10000, type=int)
    parser.add_argument('--save_ckpt_freq', default=10000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # GMFlow model
    parser.add_argument('--num_scales', default=1, type=int,
                        help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)

    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')

    # loss
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='loss weight')

    # evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_eval_to_file', action='store_true')
    parser.add_argument('--evaluate_matched_unmatched', action='store_true')

    # inference on a directory
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                        help='can specify the inference size')
    parser.add_argument('--dir_paired_data', action='store_true',
                        help='Paired data in a dir instead of a sequence')
    parser.add_argument('--save_flo_flow', action='store_true')
    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')
    parser.add_argument('--fwd_bwd_consistency_check', action='store_true',
                        help='forward backward consistency check with bidirection flow')

    # predict on sintel and kitti test set for submission
    parser.add_argument('--submission', action='store_true',
                        help='submission to sintel or kitti test sets')
    parser.add_argument('--output_path', default='output', type=str,
                        help='where to save the prediction results')
    parser.add_argument('--save_vis_flow', action='store_true',
                        help='visualize flow prediction as .png image')
    parser.add_argument('--no_save_flo', action='store_true',
                        help='not save flow as .flo')

    # distributed training
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    parser.add_argument('--count_time', action='store_true',
                        help='measure the inference time on sintel')

    return parser

def main(args):
    if not args.eval and not args.submission and args.inference_dir is None:
        if args.local_rank == 0:
            print('pytorch version:', torch.__version__)
            print(args)
            misc.save_args(args)
            misc.check_path(args.checkpoint_dir)
            misc.save_command(args.checkpoint_dir)

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.distributed = True

        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist(args.launcher, **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))

        setup_for_distributed(args.local_rank == 0)

    # model
    model = GMFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device)

    if not args.eval and not args.submission and not args.inference_dir:
        print('Model definition:')
        print(model)

    if torch.cuda.device_count() > 1:
        print('Use %d GPUs' % torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

        model_without_ddp = model.module
    else:
        model_without_ddp = model

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)
    if not args.eval and not args.submission and args.inference_dir is None:
        save_name = '%d_parameters' % num_params
        open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_epoch = 0
    start_step = 0
    # resume checkpoints
    if args.resume:
        print('Load checkpoint: %s' % args.resume)

        loc = 'cuda:{}'.format(args.local_rank)
        checkpoint = torch.load(args.resume, map_location=loc)

        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint

        model_without_ddp.load_state_dict(weights, strict=args.strict_resume)

        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not \
                args.no_resume_optimizer:
            print('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']

        print('start_epoch: %d, start_step: %d' % (start_epoch, start_step))

    # inferece on a dir
    if args.inference_dir is not None:

        vid_list = sorted(os.listdir(args.inference_dir))
        for vid in vid_list:

            print("Working on ", os.path.join(args.inference_dir, vid))

            os.makedirs(os.path.join(args.output_path, vid), exist_ok=True)

            inference_on_dir(model_without_ddp,
                            inference_dir=os.path.join(args.inference_dir, vid),
                            output_path=os.path.join(args.output_path, vid),
                            padding_factor=args.padding_factor,
                            inference_size=args.inference_size,
                            paired_data=args.dir_paired_data,
                            save_flo_flow=args.save_flo_flow,
                            attn_splits_list=args.attn_splits_list,
                            corr_radius_list=args.corr_radius_list,
                            prop_radius_list=args.prop_radius_list,
                            pred_bidir_flow=args.pred_bidir_flow,
                            fwd_bwd_consistency_check=args.fwd_bwd_consistency_check,
                            )

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)