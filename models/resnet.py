import torch
from torch import nn
import torchvision

class Resnet(nn.Module):

	def __init__(self, layers=152, n_classes=600, n_input_channels=3):
		super(Resnet, self).__init__()

		if layers == 152:
			base_model = torchvision.models.resnet152(pretrained=True)
		elif layers == 101:
			base_model = torchvision.models.resnet101(pretrained=True)
		elif layers == 50:
			base_model = torchvision.models.resnet50(pretrained=True)
		elif layers == 34:
			base_model = torchvision.models.resnet34(pretrained=True)
		else:
			raise ValueError("Unsupported ResNet.")

		if layers > 34: features_size = 2048
		else: features_size = 512

		conv1 = self.inflate_conv(base_model.conv1, n_input_channels)
		self.resnet = nn.Sequential(
			conv1,
			base_model.bn1,
			base_model.relu,
			base_model.maxpool,
			base_model.layer1,
			base_model.layer2,
			base_model.layer3,
			base_model.layer4
		)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Linear(features_size, n_classes)

		nn.init.kaiming_normal_(self.classifier.weight, mode='fan_out', nonlinearity='relu')

	def forward(self, x):
		x = self.resnet(x)
		x = self.pool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

	def inflate_conv(self, layer, n_channels):

		if n_channels == 3:
			return layer

		original_state_dict = layer.state_dict()
		original_weights = original_state_dict["weight"]
		s = original_weights.shape

		mean_weights = original_weights.mean(dim=1).unsqueeze(dim=1)

		if n_channels == 1:
			weights = mean_weights
		else:
			weights = mean_weights.repeat(1, n_channels, 1, 1)

		out_state_dict = original_state_dict
		out_state_dict["weight"] = weights

		out_layer = nn.Conv2d(n_channels, s[0], kernel_size=(s[2],s[3]),
			stride=layer.stride, padding=layer.padding, bias=layer.bias)
		out_layer.load_state_dict(out_state_dict)

		return out_layer


class ResNetLSTM(nn.Module):

	def __init__(self):
		pass

	def forward(self, x):
		pass
