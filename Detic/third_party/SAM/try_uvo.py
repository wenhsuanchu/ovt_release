from ytvostools.ytvos import YTVOS
from ytvostools.mask import decode as rle_decode
from ytvostools.mask import encode as rle_encode
import numpy as np
import os
from os import path
from PIL import Image
import json

data_root = '/projects/katefgroup/datasets/UVO'
frame_dir = '/projects/katefgroup/datasets/UVO/uvo_videos_dense_frames'
annFile = '/projects/katefgroup/datasets/UVO/VideoDenseSet/UVO_video_val_dense.json'
annFile = '/projects/katefgroup/datasets/UVO/VideoDenseSet/UVO_video_val_dense_with_label.json'

'''uvo = YTVOS(annFile)
vid_ids = uvo.getVidIds()
print(len(vid_ids))

vid = uvo.loadVids(vid_ids[np.random.randint(0,len(vid_ids))])[0]
print(vid)
# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
img = Image.open(os.path.join(frame_dir, vid['file_names'][0]))
print(img.height, img.width)
annIds = uvo.getAnnIds(vidIds=vid['id'])
anns = uvo.loadAnns(annIds)
print(len(anns)) # num objs
print(len(anns[0]['segmentations'])) #num timesteps
#print(np.stack([rle_decode(anns[i]['segmentations']) for i in range(len(anns))]).shape)
#print(np.stack([rle_decode(anns[0]['segmentations'][i]) for i in range(len(anns[0]['segmentations']))], 0).shape)
print(np.unique(rle_decode(anns[0]['segmentations'][0])), rle_decode(anns[0]['segmentations'][0]).dtype)
seg = anns[0]['segmentations'][0]['counts']
reencode_seg = rle_decode(anns[0]['segmentations'][0])
reencode_seg = rle_encode(reencode_seg)
print(reencode_seg['counts'].decode() == seg)'''

image_dir = path.join(data_root, 'uvo_videos_dense_frames')
annFile = path.join(data_root, 'VideoDenseSet', 'UVO_video_val_dense_with_label.json')

uvo = YTVOS(annFile)
vid_ids = uvo.getVidIds()

results = []
for idx in range(10):

    vid = uvo.loadVids(vid_ids[idx])[0]
    if not vid['id'] == 1:
        continue
    annIds = uvo.getAnnIds(vidIds=vid['id'])
    anns = uvo.loadAnns(annIds)

    info = {}
    info['id'] = vid['id']
    info['name'] = vid['ytid']
    info['size'] = (vid['height'], vid['width']) # Real sizes

    for i in range(len(anns)):
        inst_masks = []
        for j in range(len(anns[i]['segmentations'])):
            if anns[i]['segmentations'][j] is not None:
                seg = rle_encode(np.asfortranarray(rle_decode(anns[i]['segmentations'][j]) > 0.5).astype(np.uint8))
                reencode_seg = rle_encode(rle_decode(anns[i]['segmentations'][j]))
                reencode_seg["counts"] = reencode_seg["counts"].decode()
                if not reencode_seg == anns[i]['segmentations'][j]:
                    print("NOPE")
                seg['counts'] = seg['counts'].decode()
                if not seg == anns[i]['segmentations'][j]:
                    print("NOPE")
                inst_masks.append(seg)
            else:
                inst_masks.append(None)
        instance_dict = {
            'video_id': vid['id'],
            'score': 0.9, # We should replace this with detection scores at some point
            'category_id': 1,
            'segmentations': inst_masks
        }
        results.append(instance_dict)

results_json = json.dumps(results)
with open("test.json", 'w') as f:
    f.write(results_json)