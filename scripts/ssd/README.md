# Single Shot Multibox Detector (SSD)

## Performance

**PASCAL VOC2007 Test**

|          Model           | Paper  | Gluon-CV |                        Here (Convert)                        |
| :----------------------: | :----: | :------: | :----------------------------------------------------------: |
| ssd_300_vgg16_atrous_voc | 77.2 % |  77.6 %  | [77.4 %](https://drive.google.com/open?id=1ymUQEXvLxskudRHR3Kod3GSZ58hK98xP) |
| ssd_512_vgg16_atrous_voc | 79.8 % |  79.2 %  | [79.0 %](https://drive.google.com/open?id=1fq4GQln4eEKTl0weL-sURGRLxsa_P-j2) |
| ssd_512_resnet50_v1_voc  |   /    |  80.1 %  | [80.3 %](https://drive.google.com/open?id=1hWAb_VtfsLfXwmGkx0ehwrO0fHmJ_61Q) |
| ssd_512_mobilenet1.0_voc |   /    |  75.4 %  | [75.5 %](https://drive.google.com/open?id=12eK3Wfbef2NRwY8YGuxpbY-z8r_OMCUX) |

Scores are mean Average Precision (mAP) with PASCAL VOC2007 metric.

**COCO Test**

|           Model           |     Paper      |    Gluon-CV    |                        Here (Convert)                        |
| :-----------------------: | :------------: | :------------: | :----------------------------------------------------------: |
| ssd_300_vgg16_atrous_coco | 25.1/43.1/25.8 | 25.1/42.9/25.8 | [25.0/42.6/25.8](https://drive.google.com/open?id=10_LyISCBNIHpitYVb2qB1ESU16srxOGj) |
| ssd_512_vgg16_atrous_coco | 28.8/48.5/30.3 | 28.9/47.9/30.6 | [29.0/48.0/30.6](https://drive.google.com/open?id=10MY6wLuT21d3MF0xLY6OGngwdUoDeF-x) |
| ssd_512_resnet50_v1_coco  |       /        | 30.6/50.0/32.2 | [29.7/49.1/31.1](https://drive.google.com/open?id=1irK_mEZ9d1M44BchejKEI3Bdi8FGq2vz) |
| ssd_512_mobilenet1.0_coco |       /        | 21.7/39.2/21.3 | [20.8/38.3/20.3](https://drive.google.com/open?id=150Z-dxEyOsgdEooeI48IlWth-4TipuBO) |

### Results from training code

| Model | size |   Backbone   | Dataset |                         mAP (train)                          |
| :---: | :--: | :----------: | :-----: | :----------------------------------------------------------: |
|  ssd  | 512  | resnet50_v1s |   voc   | [80.45 %](https://drive.google.com/open?id=1_s-2t8DFhy4tGu_0A-UizESHC1zKH_6g) |

## Demo

Detect objects in an given image. (Please download pre-trained model to `~/.torch/models` first. --- If you put pre-trained model to other folder, please change the `--root`)

```shell
$ python demo_ssd_cv.py [--network ssd_300_vgg16_astrous_voc] [--images <image>.jpg] [--cuda] 
```

> Note：please choose one of the model listed in performance as network. There are several images in `../png`, you can choose one as demo

## Evaluation

The default data root is `~/.torch/datasets` (You can download dataset and build a soft-link to it)

```shell
$ python eval_ssd_cv.py [--network ssd_300_vgg16_astrous_voc] [--data-shape 300|512] [--batch-size 8] [--dataset voc|coco] [--cuda] [--root pretrained-model folder]
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

|                        vgg16_astrous                         |                           resnet50                           |                         resnet50_v1s                         |                         mobilenet1.0                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [Google Drive](https://drive.google.com/open?id=1vykA0_ANTAAcepKZIiByapK4aNxG68dm) | [Google Drive](https://drive.google.com/open?id=1G0QNgVplfNoeFeQoER4gQITzva7bAvdW) | [Google Drive](https://drive.google.com/open?id=1Mx_SIv1o1qjRz1tqEc-ggQ_MtZKQT3ET) | [Google Drive](https://drive.google.com/open?id=1F_AzbcO8VSga_2D4ER8pqTGbC5h5sWg3) |