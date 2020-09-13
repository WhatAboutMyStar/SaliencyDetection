import torch
import torchvision
from  torch.nn import functional as F
from torchvision import transforms
from attention import ChannelWiseAttention, SpatialAttention
import cv2
import numpy as np
from data import pad_resize_image

# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
#     'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
#     'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
#     'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
#     'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
# }
#
# cfg = {
# 	'VGG16' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
# }

class CFEV3(torch.nn.Module):
	def __init__(self, in_channels, out_channels=32):
		super(CFEV3, self).__init__()

		self.conv_1_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
										kernel_size=1, padding=0)
		self.conv_d_3 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
										kernel_size=3, padding=3, dilation=3, stride=1)
		self.conv_d_5 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
										kernel_size=3, padding=5, dilation=5, stride=1)
		self.conv_d_7 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
										kernel_size=3, padding=7, dilation=7, stride=1)
		self.bn1 = torch.nn.BatchNorm2d(out_channels * 4)


		self.conv_d2_3 = torch.nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels,
										kernel_size=3, padding=3, dilation=3, stride=1)
		self.conv_d2_5 = torch.nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels,
										kernel_size=3, padding=5, dilation=5, stride=1)
		self.conv_d2_7 = torch.nn.Conv2d(in_channels=out_channels * 4, out_channels=out_channels,
										kernel_size=3, padding=7, dilation=7, stride=1)
		self.bn2 = torch.nn.BatchNorm2d(out_channels * 3)
		self.conv_1_1_2 = torch.nn.Conv2d(in_channels=out_channels * 3, out_channels=out_channels,
										  kernel_size=1, padding=0)

	def forward(self, x):
		cfe1_0 = self.conv_1_1(x)
		cfe1_1 = self.conv_d_3(x)
		cfe1_2 = self.conv_d_5(x)
		cfe1_3 = self.conv_d_7(x)

		cat0 = torch.cat((cfe1_0, cfe1_1, cfe1_2, cfe1_3), dim=1)
		x = self.bn1(cat0)
		x = F.relu(x)

		cfe2_1 = self.conv_d2_3(x)
		cfe2_2 = self.conv_d2_5(x)
		cfe2_3 = self.conv_d2_7(x)

		cat1 = torch.cat((cfe2_1, cfe2_2, cfe2_3), dim=1)
		x = self.bn2(cat1)
		x = F.relu(x)
		x = self.conv_1_1_2(x)
		x = x + cfe1_0
		x = F.relu(x)
		return x


class VGG16(torch.nn.Module):
	def __init__(self, isDropout=True):
		super(VGG16, self).__init__()
		self.isDropout = isDropout
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

		#C12
		self.c1_conv = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.bn1 = torch.nn.BatchNorm2d(64)
		self.c2_conv = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
		self.bn2 = torch.nn.BatchNorm2d(64)


		#CPFE
		self.cfe1 = CFEV3(in_channels=512, out_channels=32)
		self.cfe2 = CFEV3(in_channels=512, out_channels=32)
		self.cfe3 = CFEV3(in_channels=256, out_channels=32)

		#CA
		self.CA = ChannelWiseAttention(in_channels=96)
		self.conv_11 = torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1, padding=0)
		self.bn3 = torch.nn.BatchNorm2d(64)

		#SA
		self.SA = SpatialAttention(in_channels=64)
		self.c12_conv = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
		self.bn4 = torch.nn.BatchNorm2d(64)

		self.out_conv = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0)

	def forward(self, x):
		#Block1
		x = self.conv1_1(x)
		x = F.relu(x)
		x = self.conv1_2(x)
		x = F.relu(x)
		c1 = x
		x = self.maxpooling1(x)
		if self.isDropout:
			x = torch.nn.Dropout(0.5)(x)

		#Block2
		x = self.conv2_1(x)
		x = F.relu(x)
		x = self.conv2_2(x)
		x = F.relu(x)
		c2 = x
		x = self.maxpooling2(x)
		c2_p = x
		if self.isDropout:
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
		if self.isDropout:
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
		if self.isDropout:
			x = torch.nn.Dropout(0.5)(x)

		#Block5
		x = self.conv5_1(x)
		x = F.relu(x)
		x = self.conv5_2(x)
		x = F.relu(x)
		x = self.conv5_3(x)
		x = F.relu(x)
		c5 = x
		if self.isDropout:
			c5 = torch.nn.Dropout(0.5)(c5)

		c1 = self.c1_conv(c1)
		c1 = self.bn1(c1)
		c1 = F.relu(c1)

		c2 = self.c2_conv(c2)
		c2 = self.bn2(c2)
		c2 = F.relu(c2)


		# C P F E
		c3_cfe = self.cfe3(c3)
		c4_cfe = self.cfe2(c4)
		c5_cfe = self.cfe1(c5)
		c5_cfe = torch.nn.UpsamplingBilinear2d(scale_factor=4)(c5_cfe)
		c4_cfe = torch.nn.UpsamplingBilinear2d(scale_factor=2)(c4_cfe)
		c345 = torch.cat((c3_cfe, c4_cfe, c5_cfe), dim=1)

		# C A
		ca = self.CA(c345)
		x = torch.mul(c345, ca)
		x = self.conv_11(x)
		x = self.bn3(x)
		x = F.relu(x)
		x = torch.nn.UpsamplingBilinear2d(scale_factor=4)(x)

		# S A
		sa = self.SA(x)
		c2 = torch.nn.UpsamplingBilinear2d(scale_factor=2)(c2)
		c12 = torch.cat((c1, c2), dim=1)
		c12 = self.c12_conv(c12)
		c12 = self.bn4(c12)
		c12 = torch.mul(sa, c12)

		x = torch.cat((c12, x), dim=1)
		x = self.out_conv(x)
		x = torch.sigmoid(x)

		return x

class FeatureVisualization:
	def __init__(self, layer):
		self.layer = layer
		self.model = VGG16()
		self.model.eval()
		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
											  std=[0.229, 0.224, 0.225])


	def forward(self, img_path):
		image = cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype('float32')
		image = pad_resize_image(image, target_size=256)
		image = np.transpose(image, axes=(2, 0, 1))
		image = torch.from_numpy(image).float()
		image = self.normalize(image)
		image = image.reshape(1, 3, 256, 256)

		x = image
		for name, module in self.model._modules.items():
			x = module(x)
			if name == 'conv1_2':
				break
		return x



if __name__ == '__main__':
	import matplotlib.pyplot as plt
	img = torch.randn(1, 3, 256, 256)
	model = VGG16()
	print(model(img).shape)
	num_parameters = 0
	for layer in model.parameters():
		num_parameters += layer.numel()
	print(num_parameters)

	fea = FeatureVisualization(model.conv1_1)
	img = fea.forward("./image/1.jpg")
	F.relu(img)
	o = cv2.imread("./image/1.jpg")
	o = cv2.resize(o, (256, 256))
	# cv2.imshow("img1", o)
	img = img.detach().numpy()
	img = img.reshape(64, 256, 256)
	img = img[0]
	img = img.reshape(256, 256, 1)
	# img = cv2.resize(img, (256, 256))
	# cv2.imshow("img", img)
	plt.imshow(img, cmap='gray')
	plt.show()
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# for name, module in model._modules.items():
	# 	print(name)
