import torch
from torch.nn import functional as F

class SpatialAttention(torch.nn.Module):
	def __init__(self, in_channels, kernel_size=9):
		super(SpatialAttention, self).__init__()
		pad = (kernel_size - 1) // 2

		self.conv1k_1 = torch.nn.Conv2d(in_channels, in_channels // 2, (1, kernel_size), padding=(0, pad))
		self.bn1_1 = torch.nn.BatchNorm2d(in_channels // 2)
		self.convk1_1 = torch.nn.Conv2d(in_channels // 2, 1, (kernel_size, 1), padding=(pad, 0))
		self.bn2_1 = torch.nn.BatchNorm2d(1)

		self.convk1_2 = torch.nn.Conv2d(in_channels, in_channels // 2, (kernel_size, 1), padding=(pad, 0))
		self.bn1_2 = torch.nn.BatchNorm2d(in_channels // 2)
		self.conv1k_2 = torch.nn.Conv2d(in_channels // 2, 1, (1, kernel_size), padding=(0, pad))
		self.bn2_2 = torch.nn.BatchNorm2d(1)

	def forward(self, x):
		#Group1
		features_1 = self.conv1k_1(x)
		features_1 = self.bn1_1(features_1)
		features_1 = F.relu(features_1)
		features_1 = self.convk1_1(features_1)
		features_1 = self.bn2_1(features_1)
		features_1 = F.relu(features_1)

		#Group2
		features_2 = self.convk1_2(x)
		features_2 = self.bn1_2(features_2)
		features_2 = F.relu(features_2)
		features_2 = self.conv1k_2(features_2)
		features_2 = self.bn2_2(features_2)
		features_2 = F.relu(features_2)

		add_feature = torch.sigmoid(torch.add(features_1, features_2))
		add_feature = add_feature.expand_as(x).clone()

		return add_feature

class ChannelWiseAttention(torch.nn.Module):
	def __init__(self, in_channels):
		super(ChannelWiseAttention, self).__init__()
		self.in_channels = in_channels

		self.linear_1 = torch.nn.Linear(self.in_channels, self.in_channels // 4)
		self.linear_2 = torch.nn.Linear(self.in_channels // 4, self.in_channels)

	def forward(self, x):
		batch, channels, h, w = x.size()

		features = F.adaptive_avg_pool2d(x, (1, 1)).view((batch, channels))
		features = self.linear_1(features)
		features = F.relu(features)

		features = self.linear_2(features)
		features = F.relu(features)
		features = torch.sigmoid(features)

		features = features.view((batch, channels, 1, 1))
		features = features.expand_as(x).clone()

		return features