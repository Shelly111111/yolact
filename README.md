# YOLACT: Real-time Instance Segmentation

[YOLACT: Real-time Instance Segmentation](https://arxiv.org/pdf/1904.02689.pdf)���ĸ��ִ��룬

paddle�ļ����е���paddle�Ĵ��룬pytorch�ļ����е���[torch�ٷ�ԭ����](https://github.com/dbolya/yolact)��

torch2paddle.py��torchģ��תpaddleģ�͵Ĵ���

## PaddlePaddle

�л���paddleĿ¼�£�`cd paddle`

### ѵ��

��ѵ����ʹ��`python train.py --config=yolact_base_config`

ע�⣬��ѵ��ʹ�õ���val2017���ݼ�������ֻѵ����4�֣����޸�����֤�ִ��Լ�ģ�ͱ����ִ�Ϊ2������Ϊ�˵���ѵ���Ƿ�������

����Ŀǰѵ�����������õ�ѵ��������Ϊ80000����֤�ִεȶ��뱾��ѵ��������ͬ���ɰ����Լ���ϲ�����и���

�������train2017���������ݼ��������`data\config.py`�ļ��еĶ�Ӧ����

### Ԥ��

Ԥ��ʹ�õ�ģ��Ϊ�ٷ�ģ��ת�����ģ�ͣ���δ�����޸Ļ�������ѵ����

Ԥ���ʹ��`python eval.py --trained_model=weights/yolact_base_54_800000.pdparams`

## PyTorch

�л���pytorchĿ¼�£�

��ѵ����ʹ��`python train.py --config=yolact_base_config`

Ԥ���ʹ��`python eval.py --trained_model=weights/yolact_base_54_800000.pth`

## ������֤

�ٷ�������mAPΪ29.8��Ȼ��ʹ�ùٷ�����ģ����֤��mask����Ϊ29.73��box����Ϊ32.07������Ϊ�Σ�����pytorch�ٷ�

![](https://github.com/Shelly111111/yolact/pytorch_mAP.png)
paddleת�����ģ��mask����Ϊ29.73��box����Ϊ32.44

![](https://github.com/Shelly111111/yolact/paddle_mAP.png)

�����ȼ��һ��`paddle/data`Ŀ¼���Ƿ����`coco/annotations`�Լ�`coco/images`�����ļ��к��ļ����µ��ļ���

��û�У�����Ҫ���ض�Ӧ��[val2017���ݼ�](http://images.cocodataset.org/zips/val2017.zip)������ѹ����`paddle/data`Ŀ¼�£�

����git���ƣ�ģ�͡�Դ�����Լ����ݼ����µ��˰ٶ������У�

���ӣ�https://pan.baidu.com/s/1ROYSRN94DFP-KvKjPH1YEw 

��ȡ�룺8gl6

