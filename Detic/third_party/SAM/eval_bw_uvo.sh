python eval_uvo.py --config-file ../../configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
                      --vocabulary lvis --confidence-threshold 0.5 \
                          --pred_one_class \
                          --load_backward \
                          --output /projects/katefgroup/datasets/UVO/out_uvo_iou05_07_05_t75 \
                          --opts MODEL.WEIGHTS ../../models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth