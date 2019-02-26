# pytorch-cv

Convert the [gluon-cv](https://github.com/dmlc/gluon-cv/) to pytorch. 

## Usage

1. using [gluon2torch](./utils/gluon2torch.py) to convert pretrained gluon model to pytorch

   > - `base=False` in classification and `True` in detection
   > - `reorder=True` in ssd (with resnet or mobilenet), others is `False`

2. run demo 

## Demo

### Classification

- [x] CIFAR-10
- [x] ImageNet

### Detection

- [x] SSD
- [x] YOLO

### Segmentation

- [x] FCN
- [x] PSPNet
- [x] DeepLab

## Evaluation

> Note: 
>
> 1. pytorch-cv using the pretrained model from gluon-cv
> 2. the difference is mainly caused by **image process**: in gluon-cv, use mxnet.image (which is based on OpenCV), and pytorch-cv use PIL.Image. 

### Classification

#### CIFAR10

| Model                  | gluon-cv | pytorch-cv |
| ---------------------- | -------- | ---------- |
| CIFAR_ResNet20_v1      | 92.1     | 92.9       |
| CIFAR_ResNet56_v1      | 93.6     | 94.2       |
| CIFAR_ResNet110_v1     | 93.0     | 95.2       |
| CIFAR_ResNet20_v2      | 92.1     | 92.7       |
| CIFAR_ResNet56_v2      | 93.7     | 94.6       |
| CIFAR_ResNet110_v2     | 94.3     | 95.5       |
| CIFAR_WideResNet16_10  | 95.1     | 96.7       |
| CIFAR_WideResNet28_10  | 95.6     |            |
| CIFAR_WideResNet40_8   | 95.9     |            |
| CIFAR_ResNeXt29_16x64d | 96.3     |            |

#### ImageNet

:cry: Do not have large enough device to save it.

### Detection

#### VOC

| SSD Model                | gluon-cv | pytorch-cv |
| ------------------------ | -------- | ---------- |
| ssd_300_vgg16_atrous_voc | 77.6     | 74.6       |
| ssd_512_vgg16_atrous_voc | 79.2     | 76.2       |
| ssd_512_resnet50_v1_voc  | 80.1     | 78.0       |
| ssd_512_mobilenet1.0_voc | 75.4     | 72.9       |

| YOLO Model                      | gluon-cv | pytorch-cv |
| ------------------------------- | -------- | ---------- |
| yolo3_darknet53_voc (320x320)   | 79.3     | 78.5       |
| yolo3_darknet53_voc (416x416)   | 81.5     | 80.9       |
| yolo3_mobilenet1.0_voc(320x320) | 73.9     | 72.1       |
| yolo3_mobilenet1.0_voc(416x416) | 75.8     | 74.0       |

#### COCO

| SSD Model                 | gluon-cv       | pytorch-cv     |
| ------------------------- | -------------- | -------------- |
| ssd_300_vgg16_atrous_coco | 25.1/42.9/25.8 | 23.8/40.2/24.5 |
| ssd_512_vgg16_atrous_coco | 28.9/47.9/30.6 | 27.7/45.5/29.4 |
| ssd_512_resnet50_v1_coco  | 30.6/50.0/32.2 | 28.4/46.7/29.7 |
| ssd_512_mobilenet1.0_coco | 21.7/39.2/21.3 | 19.9/36.6/19.7 |

| YOLO Model                        | gluon-cv       | pytorch-cv |
| --------------------------------- | -------------- | ---------- |
| yolo3_darknet53_coco (320x320)    | 33.6/54.1/35.8 |            |
| yolo3_darknet53_coco (416x416)    | 36.0/57.2/38.7 |            |
| yolo3_darknet53_coco (608x608)    | 37.0/58.2/40.1 |            |
| yolo3_mobilenet1.0_coco (320x320) | 26.7/46.1/27.5 |            |
| yolo3_mobilenet1.0_coco (416x416) | 28.6/48.9/29.9 |            |
| yolo3_mobilenet1.0_coco (608x608) | 28.0/49.8/27.8 |            |

### Segmentation



## TODO

- [x] add GPU to demo
- [ ] add evaluation
- [ ] add opencv version

