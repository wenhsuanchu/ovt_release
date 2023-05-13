import os
from PIL import Image

image_dir = "/projects/katefgroup/datasets/OVT-DAVIS/det_vis"

vid_list = sorted(os.listdir(image_dir))
for vid_name in vid_list:
    vid_dir = os.path.join(image_dir, vid_name)
    frames_list = sorted(os.listdir(vid_dir))

    gif_frames = []
    for frame in frames_list:
        frame_path = os.path.join(vid_dir, frame)
        img = Image.open(frame_path)
        img = img.resize((720, 480))

        gif_frames.append(img)

    os.makedirs(os.path.join(image_dir, "gifs"), exist_ok=True)
    gif_save_path = os.path.join(image_dir, "gifs", vid_name+".gif")
    gif_frames[0].save(gif_save_path, format="GIF", append_images=gif_frames, save_all=True, interlace=False, duration=100, loop=0)
        