# YOLACT: Real-time Instance Segmentation

[YOLACT: Real-time Instance Segmentation](https://arxiv.org/pdf/1904.02689.pdf)论文复现代码，

paddle文件夹中的是paddle的代码，pytorch文件夹中的是[torch官方原代码](https://github.com/dbolya/yolact)，

torch2paddle.py是torch模型转paddle模型的代码

## PaddlePaddle

切换到paddle目录下，`cd paddle`

### 训练

若训练可使用`python train.py --config=yolact_base_config`

注意，该训练使用的是val2017数据集，本人只训练了4轮，并修改了验证轮次以及模型保存轮次为2，仅仅为了调试训练是否正常，

但，目前训练配置中设置的训练最大次数为80000，验证轮次等都与本人训练参数不同，可按照自己的喜欢进行更改

如果想用train2017或其他数据集，请更改`data\config.py`文件中的对应属性

### 预测

预测使用的模型为官方模型转化后的模型，并未进行修改或者重新训练，

预测可使用`python eval.py --trained_model=weights/yolact_base_54_800000.pdparams`

## PyTorch

切换到pytorch目录下，

若训练可使用`python train.py --config=yolact_base_config`

预测可使用`python eval.py --trained_model=weights/yolact_base_54_800000.pth`

## 精度验证

官方给出的mAP为29.8，然而使用官方给的模型验证后mask精度为29.73，box精度为32.07，至于为何，请问pytorch官方

![](https://github.com/Shelly111111/yolact/pytorch_mAP.png)
paddle转化后的模型mask精度为29.73，box精度为32.44

![](https://github.com/Shelly111111/yolact/paddle_mAP.png)

请首先检查一下`paddle/data`目录下是否存在`coco/annotations`以及`coco/images`两个文件夹和文件夹下的文件，

若没有，则需要下载对应的[val2017数据集](http://images.cocodataset.org/zips/val2017.zip)，并解压缩到`paddle/data`目录下，

由于git限制，模型、源代码以及数据集更新到了百度网盘中，

链接：https://pan.baidu.com/s/1ROYSRN94DFP-KvKjPH1YEw 

提取码：8gl6

