python eval_ytvis.py --config-file ../../configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
                    --vocabulary custom \
                    --custom_vocabulary person,giant_panda,lizard,parrot,skateboard,sedan,ape,dog,snake,monkey,hand,rabbit,duck,cat,cow,fish,train,horse,turtle,bear,motorbike,giraffe,leopard,fox,deer,owl,surfboard,airplane,truck,zebra,tiger,elephant,snowboard,boat,shark,mouse,frog,eagle,seal,tennis_racket \
                    --confidence-threshold 0.35 \
                    --pred_one_class \
                    --output /projects/katefgroup/datasets/OVT-Youtube/out_debug \
                    --opts MODEL.WEIGHTS ../../models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
