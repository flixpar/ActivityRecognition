import torch
import utils.transforms as tfms

class Args:

	##############################
	###### Hyperparameters #######
	##############################

	dataset = "kinetics"            # DEFAULT kinetics (kinetics | charades | ava | diva | icu)

	epochs = 30                     # DEFAULT 30 (int 1-99)
	batch_size = 16                 # DEFAULT 16 (int)
	weight_decay = 1e-4             # DEFAULT 0 (float)

	model = "resnet-lstm"           # DEFAULT resnet_lstm (resnet-lstm | resnet-tcn | 3dresnet | i3d)
	model_config = {
	}

	initial_lr = 1e-5               # DEFAULT 1e-5 (float)
	lr_schedule = None              # DEFAULT None (None | poly | exp | step | multistep | cosine)
	lr_schedule_params = {          # DEFAULT {} (dict)
	}

	frame_size  = 256               # DEFAULT 256 (None | int)
	clip_length = 50                # DEFAULT 50 (None | int)
	frame_selection = "pad"         # DEFAULT pad (none | pad | loop | interpolate | center)

	features = [                    # DEFAULT [rgb] (rgb, flow, objects, pose)
		"rgb"
	]

	optimizer = "adam"              # DEFAULT adam (adam | sgd)
	loss = "crossentropy"           # DEFAULT crossentropy (crossentropy | focal | mse)

	weight_mode = ["inverse"]       # DEFAULT [inverse] ({inverse, sqrt} | None)
	weight_method = "sampling"      # DEFAULT loss (loss | sampling | None)

	device_ids = [0,1]              # DEFAULT [0,] (list int 0-8)
	workers = 4                     # DEFAULT 4 (int 0-32)

	log_freq = 10                   # DEFAULT 10 (int)
	n_vval_samples = None           # DEFAULT None (int | None)
	n_tval_samples = 1024           # DEFAULT 1024 (int | None)

	train_split = "train"           # DEFAULT train (train | val | trainval)
	val_split   = "val"             # DEFAULT val (train | val | trainval)

	train_augmentation = tfms.Compose([
	])

	##############################
	########### Test #############
	##############################

	test_augmentation = [         # DEFAULT [] (list)
	]
	postprocessing    = [         # DEFAULT [] (list)
	]

	##############################
	########## Paths #############
	##############################

	data_base_path  = ""
