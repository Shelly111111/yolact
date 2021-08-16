# YOLACT: Real-time Instance Segmentation

[YOLACT: Real-time Instance Segmentation](https://arxiv.org/pdf/1904.02689.pdf)论文复现代码，

paddle文件夹中的是paddle的代码，pytorch文件夹中的是[torch官方原代码](https://github.com/dbolya/yolact)，

torch2paddle.py是torch模型转paddle模型的代码

## PaddlePaddle

切换到paddle目录下，

若训练可使用`python train.py --config=yolact_base_config`

预测可使用`python eval.py --trained_model=weights/yolact_base_54_800000.pdparams`

## PyTorch

切换到pytorch目录下，

若训练可使用`python train.py --config=yolact_base_config`

预测可使用`python eval.py --trained_model=weights/yolact_base_54_800000.pth`

## 精度验证

官方给出的mAP为29.8，然而使用官方给的模型验证后mask精度为29.73，box精度为32.07

![](https://github.com/Shelly111111/yolact/pytorch_mAP.png)
paddle转化后的模型mask精度为29.73，box精度为32.44

![](https://github.com/Shelly111111/yolact/paddle_mAP.png)

由于git限制，模型、源代码以及数据集更新到了百度网盘中，

