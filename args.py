import torch
import transforms.spatial as tfms

class Args:

	debug = False                   # DEFAULT False (bool)

	# dataset
	dataset     = "kinetics"        # DEFAULT kinetics (kinetics | charades | ava | diva | icu)
	train_split = "train"           # DEFAULT train (train | val | trainval)
	val_split   = "val"             # DEFAULT val (train | val | trainval)
	input_types = [                 # DEFAULT [rgb] (rgb, flow, objects, pose)
		"rgb"
	]

	# data
	frame_size  = 256               # DEFAULT 256 (None | int)
	clip_length = 50                # DEFAULT 50 (None | int)
	frame_selection = "random"      # DEFAULT random (random | random_replace | uniform | first | center)

	# model
	model = "resnet-lstm"           # DEFAULT resnet_lstm (resnet-lstm | resnet-tcn | 3dresnet | i3d)
	model_config = {
	}

	# training
	epochs = 10                     # DEFAULT 30 (int 1-99)
	batch_size = 2                  # DEFAULT 16 (int)
	weight_decay = 1e-4             # DEFAULT 0 (float)
	optimizer = "adam"              # DEFAULT adam (adam | sgd)
	loss = "crossentropy"           # DEFAULT crossentropy (crossentropy | focal | mse)

	# learning rate
	initial_lr = 1e-3               # DEFAULT 1e-3 (float)
	lr_schedule = None              # DEFAULT None (None | poly | exp | step | multistep | cosine)
	lr_schedule_params = {          # DEFAULT {} (dict)
	}

	# weighting
	weight_mode = ["inverse"]       # DEFAULT [inverse] ({inverse, sqrt} | None)
	weight_method = "sampling"      # DEFAULT loss (loss | sampling | None)

	# sampling
	n_train_samples = 10000         # DEFAULT 10000 (int | None)
	n_vval_samples = None           # DEFAULT None  (int | None)
	n_tval_samples = 1024           # DEFAULT 1024  (int | None)

	# data loading
	workers = 8                     # DEFAULT 4 (int 0-32)
	loader = "default"              # DEFAULT default (default | lintel | pyav | opencv | pil)
	device_ids = [0,1]              # DEFAULT [0,1] (list int 0-8)

	# logging
	log_freq = 10                   # DEFAULT 10 (int)

	# augmentation
	train_augmentation = tfms.Compose([
		tfms.RandomFlip(),
	])
	test_augmentation = [           # DEFAULT [] (list)
	]
	postprocessing    = [           # DEFAULT [] (list)
	]

	# root data path
	data_base_path  = ""
