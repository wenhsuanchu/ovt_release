import os
import json

json_dir = '/projects/katefgroup/datasets/OVT-Youtube/out_ytvis/json_old'

merged_list = []

json_list = sorted(os.listdir(json_dir))
print("Found", len(json_list), "files")
for fn in json_list:
    if fn == 'merged.json':
        continue
    with open(os.path.join(json_dir, fn)) as json_file:
        pred_json = json.loads(json_file.read())
        # for i, instance in enumerate(pred_json):
        #     for j, seg in enumerate(instance['segmentations']):
        #         if seg is None:
        #             pred_json[i]['segmentations'][j] = seg
        #         else:
        #             pred_json[i]['segmentations'][j] = {"size": [720, 1280],
        #                                                 "counts": seg}
        merged_list.extend(pred_json)

with open(os.path.join(json_dir, "merged.json"), 'w') as out_file:
    out_file.write(json.dumps(merged_list))