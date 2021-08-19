# YOLACT: Real-time Instance Segmentation

[YOLACT: Real-time Instance Segmentation](https://arxiv.org/pdf/1904.02689.pdf)论文复现代码，

paddle文件夹中的是paddle的代码，pytorch文件夹中的是[torch官方原代码](https://github.com/dbolya/yolact)，

torch2paddle.py是torch模型转paddle模型的代码

## PaddlePaddle

切换到paddle目录下，`cd paddle`

### 训练

若训练可使用`python train.py --config=yolact_base_config`

注意，该训练使用的是train2017数据集，在官方给出的模型上继续训练后，精度可到

如果想指定数据集路径，请更改`data\config.py`文件中的对应属性

### 预测

预测使用的模型为官方模型转化后的模型，并未进行修改或者重新训练，

预测可使用`python eval.py --trained_model=weights/yolact_base_0_1.pdparams`

## PyTorch

切换到pytorch目录下，

若训练可使用`python train.py --config=yolact_base_config`

预测可使用`python eval.py --trained_model=weights/yolact_base_54_800000.pth`

## 精度验证

官方给出的mAP为29.8，然而使用官方给的模型验证后mask精度为29.73，box精度为32.07，至于为何，请问pytorch官方

![](https://github.com/Shelly111111/yolact/pytorch_mAP.png)
paddle转化后的模型mask精度为29.73，box精度为32.44

![](https://github.com/Shelly111111/yolact/paddle_mAP.png)

请首先检查一下`paddle/data`目录下是否存在`coco/annotations`、`coco/train`以及`coco\val`这三个文件夹和文件夹下的文件，

`coco/annotations`保存coco2017数据集的label，`coco/train`以及`coco\val`分别是coco2017的训练集和验证集，

若没有，则需要下载对应的[val2017数据集](http://images.cocodataset.org/zips/val2017.zip)、[train2017数据集](http://images.cocodataset.org/zips/train2017.zip)，并解压缩到`paddle/data`目录下，



由于git限制，模型、源代码以及数据集更新到了百度网盘中，

链接：https://pan.baidu.com/s/1ROYSRN94DFP-KvKjPH1YEw 

提取码：8gl6

