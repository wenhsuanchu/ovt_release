python youtube_inference.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
                          --input /projects/katefgroup/datasets/OVT-Youtube/all_frames/valid_all_frames/ \
                          --vocabulary lvis --confidence-threshold 0.5 \
                          --output /projects/katefgroup/datasets/OVT-Youtube/det_vis \
                          --save-format npz \
                          --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth