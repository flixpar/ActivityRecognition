import torch
from torch import nn
import torch.optim.lr_scheduler
from torch.utils.data import WeightedRandomSampler
from utils.lr_schedule import ConstantLR, PolynomialLR

def get_dataset(split, args):

	# train_dataset = ProteinImageDataset(split=args.train_split, args=args,
	# 	transforms=args.train_augmentation, channels=args.img_channels, debug=False)

	# train_static_dataset = ProteinImageDataset(split=args.train_split, args=args,
	# 	test_transforms=args.test_augmentation, channels=args.img_channels, debug=False,
	# 	n_samples=args.n_train_eval_samples)

	# val_dataset  = ProteinImageDataset(split=args.val_split, args=args,
	# 	test_transforms=args.test_augmentation, channels=args.img_channels, debug=False,
	# 	n_samples=args.n_val_samples)

	raise NotImplementedError()

def get_model(args):
	raise NotImplementedError()

def get_loss(args, weights):

	if args.weight_method == "loss":
		if args.weight_mode is not None:
			class_weights = weights
		else:
			class_weights = None
	else:
		class_weights = None

	if args.loss == "crossentropy":
		loss_func = nn.CrossEntropyLoss(weight=class_weights)
	else:
		raise ValueError("Invalid loss function specifier: {}".format(args.loss))

	return loss_func

def get_train_sampler(args, dataset):
	if args.weight_method == "sampling":
		return WeightedRandomSampler(weights=dataset.example_weights, num_samples=len(dataset))
	else:
		return None

def get_scheduler(args, optimizer):
	params = args.lr_schedule_params
	if args.lr_schedule == "poly":
		gamma = params["gamma"] if "gamma" in params else 0.9
		max_iter = args.epochs
		decay_iter = 1
		return PolynomialLR(optimizer, max_iter, decay_iter, gamma)
	elif args.lr_schedule == "exp":
		gamma = params["gamma"] if "gamma" in params else 0.9
		return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
	elif args.lr_schedule == "step":
		step_size = params["step_size"] if "step_size" in params else 5
		gamma = params["gamma"] if "gamma" in params else 0.5
		return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
	elif args.lr_schedule == "multistep":
		milestones = params["milestones"] if "milestones" in params else list(range(10, args.epochs, 10))
		gamma = params["gamma"] if "gamma" in params else 0.2
		return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma)
	elif args.lr_schedule == "cosine":
		T_max = params["period"] // 2 if "period" in params else 10
		max_decay = params["max_decay"] if "max_decay" in params else 50
		eta_min = args.initial_lr / max_decay
		return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
	else:
		return ConstantLR(optimizer)

