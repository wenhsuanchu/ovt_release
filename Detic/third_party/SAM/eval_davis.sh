python eval_davis.py --config-file ../../configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
                          --vocabulary lvis --confidence-threshold 0.5 \
                          --pred_one_class \
                          --output /projects/katefgroup/datasets/OVT_DAVIS/out_local_nosam \
                          --opts MODEL.WEIGHTS ../../models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth