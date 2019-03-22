import os
import sys
import glob
import math
import random
import tqdm
import pickle
import multiprocessing as mp
import subprocess
import json

import cv2
import numpy as np

base_dir = "/mnt/data2/"
workers = 16

def main():

	dataset = sys.argv[1]
	if dataset == "kinetics":
		videos = get_videos_kinetics()
	elif dataset == "ava":
		videos = get_videos_ava()
	else:
		raise ValueError("Invalid dataset selected.")

	pool = mp.Pool(workers)
	data = list(tqdm.tqdm(pool.imap(get_vid_info, videos), total=len(videos)))

	data = dict(data)
	out_fn = os.path.join(base_dir, dataset, "meta", "vid_info.pkl")
	with open(out_fn, "wb") as f:
		pickle.dump(data, f)

def get_videos_kinetics():
	videos = glob.glob(os.path.join(base_dir, "kinetics/vid/", "*/*.mp4"))
	videos = sorted(videos)
	return videos

def get_videos_ava():
	videos = glob.glob(os.path.join(base_dir, "ava/", "vid/", "*"))
	videos = [fn for fn in videos if fn.split('.')[-1] in ["mp4", "mkv", "webm"]]
	videos = sorted(videos)
	return videos

def get_vid_info(vid_fn):
	vid_id = vid_fn.split('/')[-1][:-4]
	cmd = ['/usr/local/bin/ffprobe', vid_fn, '-v', 'error', '-count_frames', '-select_streams', 'v', '-print_format', 'json', '-show_streams']
	result = subprocess.run(cmd, stdout=subprocess.PIPE)
	data = json.loads(result.stdout)
	data = data["streams"]
	if not data: return (vid_id, {"n_frames": 0, "n_frames_alt": 0, "error": True})
	data = data[0]
	info = {
		"n_frames": int(data["nb_read_frames"]),
		"n_frames_alt": int(data["nb_frames"]),
		"duration": float(data["duration"]),
		"height": int(data["height"]),
		"width": int(data["width"]),
		"aspect_ratio": str(data["display_aspect_ratio"]) if 'display_aspect_ratio' in data else "",
		"fps": float(eval(data["avg_frame_rate"])),
		"fps_alt": float(eval(data["r_frame_rate"])),
		"error": False
	}
	return (vid_id, info)

if __name__ == "__main__":
	main()