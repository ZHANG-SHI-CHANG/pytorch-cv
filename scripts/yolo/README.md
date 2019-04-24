# YOLOv3: An Incremental Improvement

## Performance

**PASCAL VOC2007 Test**

|              Model               | Paper | Gluon-CV | Here (Convert) | Here (train) |
| :------------------------------: | :---: | :------: | :------------: | :----------: |
|  yolo3_darknet53_voc (320x320)   |   /   |  79.3 %  |      []()      |              |
|  yolo3_darknet53_voc (416x416)   |   /   |  81.5 %  |      []()      |              |
| yolo3_mobilenet1.0_voc (320x320) |   /   |  73.9 %  |      []()      |              |
| yolo3_mobilenet1.0_voc (416x416) |   /   |  75.8 %  |      []()      |              |

Scores are mean Average Precision (mAP) with PASCAL VOC2007 metric.

**COCO Test**

|               Model               |     Paper      |    Gluon-CV    | Here (Convert) | Here (train) |
| :-------------------------------: | :------------: | :------------: | :------------: | :----------: |
|  yolo3_darknet53_coco (320x320)   |   no/51.5/no   | 33.6/54.1/35.8 |      []()      |              |
|  yolo3_darknet53_coco (416x416)   |   no/55.3/no   | 36.0/57.2/38.7 |      []()      |              |
|  yolo3_darknet53_coco (608x608)   | 33.0/57.9/34.4 | 37.0/58.2/40.1 |      []()      |              |
| yolo3_mobilenet1.0_coco (320x320) |       /        | 26.7/46.1/27.5 |      []()      |              |
| yolo3_mobilenet1.0_coco (416x416) |       /        | 28.6/48.9/29.9 |      []()      |              |
| yolo3_mobilenet1.0_coco (608x608) |       /        | 28.0/49.8/27.8 |      []()      |              |

## Demo

Detect objects in an given image. (Please download pre-trained model to `~/.torch/models` first. --- If you put pre-trained model to other folder, please change the `--root`)

```shell
$ python demo_ssd_cv.py [--network ssd_300_vgg16_astrous_voc] [--images <image>.jpg] [--cuda] 
```

> Note：please choose one of the model listed in performance as network. There are several images in `../png`, you can choose one as demo

## Evaluation

The default data root is `~/.torch/datasets` (You can build a soft-link to it)

```shell
$ python [--network ssd_300_vgg16_astrous_voc] [--data-shape 300|512] [--batch-size 8] [--dataset voc|coco] [--cuda] [--root pretrained-model folder]
```

> Note：
>
> 1. please make sure the network and data-shape is consistent. 
> 2. the default root is `~/.torch/models` (And make sure the pre-trained model is named as  `<--network>.pth`)

## Train

Download pre-trained backbone and put it on `~/.torch/models`

Recommend to using distributed training.

```shell
$ export NGPUS=4
$ python -m torch.distributed.launch --nproc_per_node=$NGPUS train_ssd_cv.py [--network vgg16_atrous] [--data-shape 300] [--dataset voc|coco] [--batch-size 32] [--test-batch-size 16] [--lr 1e-3] [--lr-decay-epoch 160,200] [--lr-decay 0.1] [--epochs 240] [-j 16] [--lr-mode step|cos] [--warmup-factor 0.01] [--log-step 10]
```

> Note：
>
> 1. the batch-size is per-batch-size-in-one-gpu

## Appendix

**Pre-trained Backbone**

|           darknet53           |         mobilenet1.0          |
| :---------------------------: | :---------------------------: |
| [BaiduYun]()/[Google Drive]() | [BaiduYun]()/[Google Drive]() |

