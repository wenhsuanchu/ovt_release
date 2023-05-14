# ovt
```
conda create --name ovt python=3.8 -y
conda activate ovt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```

# install detectron2
```
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
```

# install Detic requirements
```
cd Detic
pip install -r requirements.txt
```

# install (custom) SAM
```
cd segment-anything; pip install -e .
```

# install SAM requirements
```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

# install GMFlow requirements
```
cd Detic/third_party/gmflow
conda env update --file local.yml
```

# download weights
- Download GMFlow weights from https://drive.google.com/file/d/1d5C5cgHIxWGsFR1vYs5XrQbbUiZl9TX2/view and extract to:
  ```
  ovt/Detic/third_party/gmflow/pretrained/
  ```
  Ensure the following checkpoint is available: `ovt/Detic/third_party/gmflow/pretrained/gmflow_with_refine_sintel-3ed1cf48.pth`

- Download DETIC weights:
```
cd ovt/Detic/models/
wget https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```
- Download SAM weights
```
cd ovt/Detic/third_party/SAM/pretrained/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

# run tracker
Please see the eval.sh files under ovt/Detic/third_party/SAM
