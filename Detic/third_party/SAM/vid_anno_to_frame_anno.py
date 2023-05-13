import os
import json

SPECIFIED_VID_ID = None#[1, 9, 10, 11, 21, 23]

# Parse GT
gt_json_path = '/projects/katefgroup/datasets/UVO/VideoDenseSet/UVO_video_val_dense.json'

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
    for ti in range(vid_len):
        pseudo_start_idx = start_idx + ti
        pseudo_video_id = video_id * 1000 + ti
        per_frame_entry = {
            'start_idx': pseudo_start_idx,
            'id': pseudo_video_id,
            'height': height,
            'width': width,
            'ytid': ytid,
            'label': label,
            'file_names': [vid_entry['file_names'][ti]]
        }
        vid_entries.append(per_frame_entry)

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
    for ti in range(vid_len):
        if tracklet['segmentations'][ti] is not None:
            pseudo_video_id = vid_id * 1000 + ti
            pseudo_idx = idx * 1000 + ti
            per_frame_entry = {
                'height': height,
                'width': width,
                'length': 1,
                'category_id': cat_id,
                'video_id': pseudo_video_id,
                'iscrowd': iscrowd,
                'areas': [tracklet['areas'][ti]],
                'bboxes': [tracklet['bboxes'][ti]],
                'segmentations': [tracklet['segmentations'][ti]],
                'start_idx': start_idx,
                'id': pseudo_idx
            }
            anno_entries.append(per_frame_entry)

perframe_gt = {
    'info': gt["info"],
    'videos': vid_entries,
    'categories': gt["categories"],
    'annotations': anno_entries
}
with open(os.path.join('/projects/katefgroup/datasets/UVO/VideoDenseSet/UVO_video_val_dense_perframe.json'), 'w') as out_file:
    out_file.write(json.dumps(perframe_gt))

# Parse predictions
json_dir = '/projects/katefgroup/datasets/UVO/out_uvo/json'
out_json_dir = '/projects/katefgroup/datasets/UVO/out_uvo/json_perframe'

merged_list = []

json_list = sorted(os.listdir(json_dir))
print("Found", len(json_list), "files")
for i, fn in enumerate(json_list):

    if fn == 'merged.json':
        continue

    with open(os.path.join(json_dir, fn), 'r') as json_file:
        pred_json = json.loads(json_file.read())
    for tracklet in pred_json:
        video_id = tracklet['video_id']
        score = tracklet['score']
        cat_id = tracklet['category_id']
        vid_len = len(tracklet['segmentations'])
        if SPECIFIED_VID_ID is not None and video_id not in SPECIFIED_VID_ID:
            continue
        for ti in range(vid_len):
            if tracklet['segmentations'][ti] is not None:
                pseudo_video_id = video_id * 1000 + ti
                per_frame_instance = {
                    'video_id': pseudo_video_id,
                    'score': score, # We should replace this with detection scores at some point
                    'category_id': cat_id,
                    'segmentations': [tracklet['segmentations'][ti]]
                }
                merged_list.append(per_frame_instance)

os.makedirs(out_json_dir, exist_ok=True)
with open(os.path.join(out_json_dir, "merged.json"), 'w') as out_file:
    out_file.write(json.dumps(merged_list))