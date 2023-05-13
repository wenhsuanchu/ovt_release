python eval_davis_fb.py --config-file ../../configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
                      --vocabulary lvis --confidence-threshold 0.5 \
                          --pred_one_class \
                          --output ./out_tta_fb \
                          --opts MODEL.WEIGHTS ../../models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth