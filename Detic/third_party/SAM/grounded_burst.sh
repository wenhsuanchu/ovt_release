python grounded_burst.py --config-file ../../configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
                         --dataset_path /projects/katefgroup/datasets/OVT-DAVIS/ \
                      --vocabulary lvis --confidence-threshold 0.5 \
                      --caption "person most to the right" \
                          --pred_one_class \
                          --output /projects/katefgroup/datasets/OVT-DAVIS/out_grounded \
                          --opts MODEL.WEIGHTS ../../models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth