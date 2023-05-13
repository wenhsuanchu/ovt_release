import os
import json

json_dir = '/projects/katefgroup/datasets/UVO/out_uvo_iou05_07_05/json'

merged_list = []

json_list = sorted(os.listdir(json_dir))
print("Found", len(json_list), "files")
for fn in json_list:
    if fn == 'merged.json':
        continue
    with open(os.path.join(json_dir, fn)) as json_file:
        pred_json = json.loads(json_file.read())
        merged_list.extend(pred_json)

with open(os.path.join(json_dir, "merged.json"), 'w') as out_file:
    out_file.write(json.dumps(merged_list))