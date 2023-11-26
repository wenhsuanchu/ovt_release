python eval_burst.py --config-file ../../configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
                      --vocabulary lvis --confidence-threshold 0.5 \
                          --pred_one_class \
                          --output /projects/katefgroup/datasets/BURST/out_burst \
                          --opts MODEL.WEIGHTS ../../models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth