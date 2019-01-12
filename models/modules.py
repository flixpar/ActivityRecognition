import torch
from torch import nn
import torchvision

class Resnet(nn.Module):

	def __init__(self, layers=152, output_dims=512, n_input_channels=3):
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
		self.fc = nn.Linear(features_size, output_dims)

		nn.init.kaiming_normal_(self.fc.weight, mode="fan_out", nonlinearity='relu')

	def forward(self, x):
		x = self.resnet(x)
		x = self.pool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
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


class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()

class TCN_TemporalBlock(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
		super(TCN_TemporalBlock, self).__init__()
		self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
									stride=stride, padding=padding, dilation=dilation))
		self.chomp1 = Chomp1d(padding)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
									stride=stride, padding=padding, dilation=dilation))
		self.chomp2 = Chomp1d(padding)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)

		self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
								self.conv2, self.chomp2, self.relu2, self.dropout2)
		self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
		self.relu = nn.ReLU()
		self.init_weights()

	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.01)
		self.conv2.weight.data.normal_(0, 0.01)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)
		return self.relu(out + res)


class TemporalConvNet(nn.Module):
	def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
		super(TemporalConvNet, self).__init__()

		layers = []
		num_levels = len(num_channels)

		for i in range(num_levels):
			dilation_size = 2 ** i
			in_channels = num_inputs if i == 0 else num_channels[i-1]
			out_channels = num_channels[i]
			layers += [TCN_TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
									dilation=dilation_size, padding=(kernel_size-1) * dilation_size,
									dropout=dropout)]

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)

