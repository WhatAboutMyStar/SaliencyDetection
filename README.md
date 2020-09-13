# SaliencyDetection
我的毕业设计，图像显著性检测

## 目录结构
```
│  attention.py
│  data.py
│  loss.py
│  model.py
│  PR曲线.ipynb
│  README.md
│  utils.py
│  学习率衰减曲线.ipynb
│  测试.ipynb
│  训练模型样例.ipynb
│
├─DUTS
│  ├─DUTS-TE
│  │  ├─DUTS-TE-Image
│  │  │      ILSVRC2012_test_00000003.jpg
│  │  │      ILSVRC2012_test_00000023.jpg
│  │  │      ILSVRC2012_test_00000025.jpg
│  │  │      ...
│  │  └─DUTS-TE-Mask
│  │          ILSVRC2012_test_00000003.png
│  │          ILSVRC2012_test_00000023.png
│  │          ILSVRC2012_test_00000025.png
│  │          ...
│  └─DUTS-TR
│      ├─DUTS-TR-Image
│      │      ILSVRC2012_test_00000004.jpg
│      │      ILSVRC2012_test_00000018.jpg
│      │      ILSVRC2012_test_00000019.jpg
│      │      ...
│      └─DUTS-TR-Mask
│              ILSVRC2012_test_00000004.png
│              ILSVRC2012_test_00000018.png
│              ILSVRC2012_test_00000019.png
│              ...
├─image
│      1.jpg
│      2.jpg
│      3.jpg
│      CPFEV3.png
│      金字塔特征注意网络.png
│
├─model
│      vgg16_no_top.pth
│
└─notebook
      (略)
```
- model.py实现的是整个模型
- attention.py实现的是空间注意和通道注意力机制
- loss.py实现的是边界保持损失函数，但是最终没用上
- data.py实现的是数据增强和数据载入
- utils.py实现的是评价指标，F-Measure，recall，precision，accuracy
- DUTS文件夹下存放DUTS数据集
- image文件夹下存放有网络结构图
- model文件夹下存放有一个去掉全连接层的VGG16模型，不过state_dict状态字典的命名是我在model.py中的命名
- notebook文件夹下存放着大量notebook，记录有我的调参记录，别问为什么这么做，问就是电脑垃圾长时间训练会很烫，慢慢来最好。
