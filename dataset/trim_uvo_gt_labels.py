import os
import json

gt_json_path = '/projects/katefgroup/datasets/UVO/VideoDenseSet/UVO_video_val_dense.json'
SPECIFIED_VID_ID = [1, 9, 10, 11, 21, 23]

trimmed_list = []

with open(gt_json_path, 'r') as j:
    gt = json.loads(j.read())

vid_entries = []

for i, vid_entry in enumerate(gt["videos"]):

    start_idx = vid_entry['start_idx']
    video_id = vid_entry['id']
    height = vid_entry['height']
    width = vid_entry['width']
    ytid = vid_entry['ytid']
    label = vid_entry['label']
    vid_len = len(vid_entry['file_names'])
    if SPECIFIED_VID_ID is not None and video_id not in SPECIFIED_VID_ID:
        continue
    entry = {
        'start_idx': start_idx,
        'id': video_id,
        'height': height,
        'width': width,
        'ytid': ytid,
        'label': label,
        'file_names': [vid_entry['file_names']]
    }
    vid_entries.append(entry)

anno_entries = []

for i, tracklet in enumerate(gt["annotations"]):

    height = tracklet["height"]
    width = tracklet["width"]
    length = tracklet["length"]
    cat_id = tracklet["category_id"]
    vid_id = tracklet["video_id"]
    iscrowd = tracklet["iscrowd"]
    start_idx = tracklet["start_idx"]
    idx = tracklet['id']
    vid_len = len(tracklet['segmentations'])
    if SPECIFIED_VID_ID is not None and vid_id not in SPECIFIED_VID_ID:
        continue
    anno_entries.append(tracklet)

trimmed_gt = {
    'info': gt["info"],
    'videos': vid_entries,
    'categories': gt["categories"],
    'annotations': anno_entries
}
with open(os.path.join('/projects/katefgroup/datasets/UVO/VideoDenseSet/UVO_video_val_dense_trimmed.json'), 'w') as out_file:
    out_file.write(json.dumps(trimmed_gt))