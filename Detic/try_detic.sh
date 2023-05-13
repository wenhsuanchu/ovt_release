python try_detic.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
                          --input /projects/katefgroup/datasets/OVT-DAVIS/ \
                          --vocabulary lvis --confidence-threshold 0.5 \
                          --save-format png \
                          --pred_one_class \
                          --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth