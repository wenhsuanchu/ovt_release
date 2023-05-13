import os
import gdown
import zipfile

DL_PATH = "/projects/katefgroup/datasets/UVO"
os.makedirs(DL_PATH, exist_ok=True)

gdown.download('https://drive.google.com/uc?id=1wQHv0IF3oXe4dawPiLoPxWD9JCAa3pro', output=os.path.join(DL_PATH, 'uvo.zip'), quiet=False)