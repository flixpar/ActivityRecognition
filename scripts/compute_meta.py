import os
import sys
import glob
import math
import random
import tqdm
import pickle
import multiprocessing as mp

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
	vid = cv2.VideoCapture(vid_fn)
	if not vid.isOpened(): return (vid_id, {"n_frames": None, "fps": None, "width": None, "height": None, "error": True})
	n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = int(vid.get(cv2.CAP_PROP_FPS))
	height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
	width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
	info = {
		"n_frames": n_frames,
		"fps": fps,
		"width": width,
		"height": height,
		"error": False
	}
	return (vid_id, info)

if __name__ == "__main__":
	main()