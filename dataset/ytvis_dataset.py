import os

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from ytvostools.ytvos import YTVOS

class YTVISTestDataset(Dataset):
    def __init__(self, data_root, load_backward=False):

        self.image_dir = os.path.join(data_root, 'all_frames', 'valid_all_frames', 'JPEGImages')
        self.annFile = os.path.join(data_root, 'vis_jsons', 'valid.json')

        self.ytvis = YTVOS(self.annFile)
        self.vid_ids = self.ytvis.getVidIds()

        if load_backward:
            print("Loading backwards")
            self.vid_ids = list(reversed(self.vid_ids))

        self.img_transform = transforms.Resize(480, interpolation=InterpolationMode.BILINEAR, antialias=True)

    def __getitem__(self, idx):

        vid = self.ytvis.loadVids(self.vid_ids[idx])[0]

        info = {}
        info['id'] = vid['id']
        info['name'] = vid['file_names'][0].partition('/')[0]
        info['size'] = (vid['height'], vid['width']) # Real sizes
        info['required_frames'] = vid['file_names']

        frames = sorted(os.listdir(os.path.join(self.image_dir, info['name'])))

        images = []
        info['all_frames'] = []
        for i, f in enumerate(frames):

            img = Image.open(os.path.join(self.image_dir, info['name'], f)).convert('RGB')
            img = np.array(self.img_transform(img))
            if i == 0:
                info['size_480p'] = img.shape[:2]
            images.append(np.array(img))
            info['all_frames'].append(os.path.join(info['name'], f))

        images = np.stack(images, 0)

        data = {
            'rgb': images,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.vid_ids)