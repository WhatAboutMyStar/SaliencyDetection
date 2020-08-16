import torch
import torchvision
from  torch.nn import functional as F
from torchvision import transforms
from attention import ChannelWiseAttention, SpatialAttention
import cv2

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

cfg = {
	'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

class MFE(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super(MFE, self).__init__()
		self.conv11 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
		self.conv33 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
		self.conv55 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)

	def forward(self, x):
		x1 = self.conv11(x)
		x1 = F.relu(x1)

		x2 = self.conv33(x)
		x2 = F.relu(x2)

		x3 = self.conv55(x)
		x3 = F.relu(x3)

		x = x1 + x2 + x3
		x = F.relu(x)

		return x

class VGG16(torch.nn.Module):
	def __init__(self):
		super(VGG16, self).__init__()
		#Block1
		self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
		self.conv1_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.maxpooling1 = torch.nn.MaxPool2d(kernel_size=2, stride=(2, 2))

		#Block2
		self.conv2_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
		self.conv2_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
		self.maxpooling2 = torch.nn.MaxPool2d(kernel_size=2, stride=(2, 2))

		#Block3
		self.conv3_1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
		self.conv3_2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.conv3_3 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.maxpooling3 = torch.nn.MaxPool2d(kernel_size=2, stride=(2, 2))

		#Block4
		self.conv4_1 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
		self.conv4_2 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
		self.conv4_3 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
		self.maxpooling4 = torch.nn.MaxPool2d(kernel_size=2, stride=(2, 2))

		#Block5
		self.conv5_1 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
		self.conv5_2 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
		self.conv5_3 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
		self.maxpooling5 = torch.nn.MaxPool2d(kernel_size=2, stride=(2, 2))

		#MFE_1
		self.mfe1 = MFE(in_channels=512, out_channels=512)
		self.mfe2 = MFE(in_channels=1024, out_channels=256)
		self.mfe3 = MFE(in_channels=512, out_channels=128)
		self.mfe4 = MFE(in_channels=256, out_channels=64)
		self.mfe5 = MFE(in_channels=128, out_channels=64)

		#MFE_2
		self.mfe6 = MFE(in_channels=128, out_channels=64)
		self.mfe7 = MFE(in_channels=256, out_channels=128)
		self.mfe8 = MFE(in_channels=512, out_channels=256)
		#MFE_3
		self.mfe11 = MFE(in_channels=192, out_channels=64)
		self.mfe12 = MFE(in_channels=768, out_channels=256)

		#CA SA
		self.CA = ChannelWiseAttention(in_channels=512)
		self.SA = SpatialAttention(in_channels=64)

		self.conv11 = torch.nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, padding=0)

		#BN
		self.bn1 = torch.nn.BatchNorm2d(64)
		self.bn2 = torch.nn.BatchNorm2d(128)
		self.bn3 = torch.nn.BatchNorm2d(64)
		self.bn4 = torch.nn.BatchNorm2d(64)

		self.conv = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0)



	def forward(self, x):
		#Block1
		x = self.conv1_1(x)
		x = F.relu(x)
		x = self.conv1_2(x)
		x = F.relu(x)
		c1 = x
		x = self.maxpooling1(x)
		x = torch.nn.Dropout(0.5)(x)

		#Block2
		x = self.conv2_1(x)
		x = F.relu(x)
		x = self.conv2_2(x)
		x = F.relu(x)
		c2 = x
		x = self.maxpooling2(x)
		x = torch.nn.Dropout(0.5)(x)

		#Block3
		x = self.conv3_1(x)
		x = F.relu(x)
		x = self.conv3_2(x)
		x = F.relu(x)
		x = self.conv3_3(x)
		x = F.relu(x)
		c3 = x
		x = self.maxpooling3(x)
		x = torch.nn.Dropout(0.5)(x)

		#Block4
		x = self.conv4_1(x)
		x = F.relu(x)
		x = self.conv4_2(x)
		x = F.relu(x)
		x = self.conv4_3(x)
		x = F.relu(x)
		c4 = x
		x = self.maxpooling4(x)
		x = torch.nn.Dropout(0.5)(x)

		#Block5
		x = self.conv5_1(x)
		x = F.relu(x)
		x = self.conv5_2(x)
		x = F.relu(x)
		x = self.conv5_3(x)
		x = F.relu(x)
		c5 = x
		x = torch.nn.Dropout(0.5)(x)

		#MFE_1
		x = self.mfe1(x)
		x = torch.nn.Dropout(0.5)(x)
		c6_up = torch.nn.UpsamplingBilinear2d(scale_factor=2)(x)

		x = torch.cat((c4, c6_up), 1)
		x = self.mfe2(x)
		x = torch.nn.Dropout(0.5)(x)
		c7 = x

		x = torch.nn.UpsamplingBilinear2d(scale_factor=2)(x)
		x = torch.cat((c3, x), 1)
		x = self.mfe3(x)
		x = torch.nn.Dropout(0.5)(x)
		c8 = x

		x = torch.nn.UpsamplingBilinear2d(scale_factor=2)(x)
		x = torch.cat((c2, x), 1)
		x = self.mfe4(x)
		x = torch.nn.Dropout(0.5)(x)
		c9 = x

		x = torch.nn.UpsamplingBilinear2d(scale_factor=2)(x)
		x = torch.cat((c1, x), 1)
		x = self.mfe5(x)
		x = torch.nn.Dropout(0.5)(x)
		c10 = x

		#MFE_2
		x = torch.cat((c1, x), 1)
		x = self.mfe6(x)
		x = torch.nn.Dropout(0.5)(x)
		c11 = x

		x = torch.nn.MaxPool2d(kernel_size=2, stride=(2, 2))(x)
		x = torch.cat((x, c9, c2), 1)
		x = self.mfe7(x)
		x = torch.nn.Dropout(0.5)(x)
		c12 = x

		x = torch.nn.MaxPool2d(kernel_size=2, stride=(2, 2))(x)
		x = torch.cat((x, c8, c3), 1)
		x = self.mfe8(x)
		x = torch.nn.Dropout(0.5)(x)
		c13 = x

		#cat
		c17 = torch.cat((c6_up, c7), 1)
		x = self.mfe12(c17)
		x = torch.nn.UpsamplingBilinear2d(scale_factor=2)(x)
		c17 = x

		# C A
		x = torch.cat((c13, c17), 1)
		ca = self.CA(x)
		x = torch.mul(x, ca)
		x = self.conv11(x)
		x = torch.nn.UpsamplingBilinear2d(scale_factor=4)(x)

		# C11 C12 bn123
		c11 = self.bn1(c11)
		c12 = self.bn2(c12)
		c12 = torch.nn.UpsamplingBilinear2d(scale_factor=2)(c12)
		c11_12 = torch.cat((c11, c12), 1)
		c16 = self.mfe11(c11_12)
		c16 = torch.nn.Dropout(0.5)(c16)
		c16 = self.bn3(c16)

		# S A
		sa = self.SA(x)
		sa = torch.mul(c16, sa)
		x = torch.cat((x, sa), 1)

		x = self.conv(x)
		x = torch.sigmoid(x)

		return x
if __name__ == '__main__':
	img = torch.randn(1, 3, 256, 256)
	model = VGG16()
	print(model(img).shape)