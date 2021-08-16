# YOLACT: Real-time Instance Segmentation

[YOLACT: Real-time Instance Segmentation](https://arxiv.org/pdf/1904.02689.pdf)���ĸ��ִ��룬

paddle�ļ����е���paddle�Ĵ��룬pytorch�ļ����е���[torch�ٷ�ԭ����](https://github.com/dbolya/yolact)��

torch2paddle.py��torchģ��תpaddleģ�͵Ĵ���

## PaddlePaddle

�л���paddleĿ¼�£�

��ѵ����ʹ��`python train.py --config=yolact_base_config`

Ԥ���ʹ��`python eval.py --trained_model=weights/yolact_base_54_800000.pdparams`

## PyTorch

�л���pytorchĿ¼�£�

��ѵ����ʹ��`python train.py --config=yolact_base_config`

Ԥ���ʹ��`python eval.py --trained_model=weights/yolact_base_54_800000.pth`

## ������֤

�ٷ�������mAPΪ29.8��Ȼ��ʹ�ùٷ�����ģ����֤��mask����Ϊ29.73��box����Ϊ32.07

![](https://github.com/Shelly111111/yolact/pytorch_mAP.png)
paddleת�����ģ��mask����Ϊ29.73��box����Ϊ32.44

![](https://github.com/Shelly111111/yolact/paddle_mAP.png)

����git���ƣ�ģ�͡�Դ�����Լ����ݼ����µ��˰ٶ������У�

