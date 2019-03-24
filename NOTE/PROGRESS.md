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

### Pose Estimation

- [x] simple pose

## Evaluation

> Note: 
>
> 1. pytorch-cv using the pretrained model from gluon-cv
> 2. the difference is mainly caused by **image process**: in gluon-cv, use mxnet.image (which is based on OpenCV --- not the CV2 Python library.), and pytorch-cv use PIL.Image or CV2 Python library. 

### Classification

#### CIFAR10

| Model                  | gluon-cv | pytorch-cv(PIL) |
| ---------------------- | -------- | --------------- |
| CIFAR_ResNet20_v1      | 92.1     | 92.9            |
| CIFAR_ResNet56_v1      | 93.6     | 94.2            |
| CIFAR_ResNet110_v1     | 93.0     | 95.2            |
| CIFAR_ResNet20_v2      | 92.1     | 92.7            |
| CIFAR_ResNet56_v2      | 93.7     | 94.6            |
| CIFAR_ResNet110_v2     | 94.3     | 95.5            |
| CIFAR_WideResNet16_10  | 95.1     | 96.7            |
| CIFAR_WideResNet28_10  | 95.6     | 97.1            |
| CIFAR_WideResNet40_8   | 95.9     | 97.3            |
| CIFAR_ResNeXt29_16x64d | 96.3     | 97.3            |

#### ImageNet

😢 Do not have large enough device to save it.

### Detection

#### VOC

| SSD Model                | gluon-cv | pytorch-cv(PIL) |
| ------------------------ | -------- | --------------- |
| ssd_300_vgg16_atrous_voc | 77.6     | 74.6            |
| ssd_512_vgg16_atrous_voc | 79.2     | 76.2            |
| ssd_512_resnet50_v1_voc  | 80.1     | 78.0            |
| ssd_512_mobilenet1.0_voc | 75.4     | 72.9            |

| YOLO Model                      | gluon-cv | pytorch-cv(PIL) |
| ------------------------------- | -------- | --------------- |
| yolo3_darknet53_voc (320x320)   | 79.3     | 78.5            |
| yolo3_darknet53_voc (416x416)   | 81.5     | 80.9            |
| yolo3_mobilenet1.0_voc(320x320) | 73.9     | 72.1            |
| yolo3_mobilenet1.0_voc(416x416) | 75.8     | 74.0            |

#### COCO

| SSD Model                 | gluon-cv       | pytorch-cv(PIL) |
| ------------------------- | -------------- | --------------- |
| ssd_300_vgg16_atrous_coco | 25.1/42.9/25.8 | 23.8/40.2/24.5  |
| ssd_512_vgg16_atrous_coco | 28.9/47.9/30.6 | 27.7/45.5/29.4  |
| ssd_512_resnet50_v1_coco  | 30.6/50.0/32.2 | 28.4/46.7/29.7  |
| ssd_512_mobilenet1.0_coco | 21.7/39.2/21.3 | 19.9/36.6/19.7  |

| YOLO Model                        | gluon-cv       | pytorch-cv(PIL) |
| --------------------------------- | -------------- | --------------- |
| yolo3_darknet53_coco (320x320)    | 33.6/54.1/35.8 | 32.3/51.8/34.5  |
| yolo3_darknet53_coco (416x416)    | 36.0/57.2/38.7 | 34.9/55.2/37.7  |
| yolo3_darknet53_coco (608x608)    | 37.0/58.2/40.1 | 35.9/56.2/39.0  |
| yolo3_mobilenet1.0_coco (320x320) | 26.7/46.1/27.5 | 25.4/43.6/26.1  |
| yolo3_mobilenet1.0_coco (416x416) | 28.6/48.9/29.9 | 27.6/46.8/28.9  |
| yolo3_mobilenet1.0_coco (608x608) | 28.0/49.8/27.8 | 27.1/48.1/27.1  |

### Segmentation

Note：

1. value in () means after using base_size=520, crop_size=480.  
2. pytorch-cv using PIL.Image

| ADE20K Dataset        | pixAcc(gluon-cv) | mIoU(gluon-cv) | pixAcc(pytorch-cv) | mIoU(pytorch-cv) |
| --------------------- | ---------------- | -------------- | ------------------ | ---------------- |
| fcn_resnet50_ade      | 79.0             | 39.5           | 79.0               | 39.5             |
| fcn_resnet101_ade     | 80.6             | 41.6           | 80.6               | 41.6             |
| psp_resnet50_ade      | 80.1             | 41.5           | 80.1               | 41.5             |
| psp_resnet101_ade     | 80.8             | 43.3           | 80.8               | 43.3             |
| deeplab_resnet50_ade  | 80.5             | 42.5           | 80.5               | 42.5             |
| deeplab_resnet101_ade | 81.1             | 44.1           | 81.1               | 44.1             |

| COCO Dataset           | pixAcc(gluon-cv) | mIoU(gluon-cv) | pixAcc(pytorch-cv) | mIoU(pytorch-cv) |
| ---------------------- | ---------------- | -------------- | ------------------ | ---------------- |
| fcn_resnet101_coco     | 92.2 (91.1)      | 66.2 (60.3)    | 92.2 (90.9)        | 66.2 (59.8)      |
| psp_resnet101_coco     | 92.4 (91.8)      | 70.4 (68.5)    | 92.4 (91.7)        | 70.4 (68.8)      |
| deeplab_resnet101_coco | 92.5 (91.7)      | 70.4 (68.7)    | 92.5 (91.6)        | 70.4 (68.3)      |

| VOC Dataset           | pixAcc(gluon-cv) | mIoU(gluon-cv) | pixAcc(pytorch-cv) | mIoU(pytorch-cv) |
| --------------------- | ---------------- | -------------- | ------------------ | ---------------- |
| fcn_resnet101_voc     |                  | 83.6           |                    |                  |
| psp_resnet101_voc     |                  | 85.1           |                    |                  |
| deeplab_resnet101_voc |                  | 86.2           |                    |                  |
| deeplab_resnet152_voc |                  | 86.7           |                    |                  |
| psp_resnet101_citys   |                  | 77.1           |                    |                  |

### Pose Estimation

#### Simple Pose with ResNet

| COCO Dataset                       | AP(gluon-cv)   | AP with flip(gluon-cv) | AP(pytorch-cv) | AP with flip(pytorch-cv) |
| ---------------------------------- | -------------- | ---------------------- | -------------- | ------------------------ |
| simple_pose_resnet18_v1b           | 66.3/89.2/73.4 | 68.4/90.3/75.7         | 66.3/89.2/73.4 | 68.4/90.3/75.7           |
| simple_pose_resnet18_v1b(128x96)   | 52.8/83.6/57.9 | 54.5/84.8/60.3         | 52.8/83.6/57.9 | 54.5/84.8/60.3           |
| simple_pose_resnet50_v1b           | 71.0/91.2/78.6 | 72.2/92.2/79.9         | 71.0/91.2/78.6 | 72.2/92.2/79.9           |
| simple_pose_resnet50_v1d           | 71.6/91.3/78.7 | 73.3/92.4/80.8         | 71.6/91.3/78.7 | 73.3/92.4/80.8           |
| simple_pose_resnet101_v1b          | 72.4/92.2/79.8 | 73.7/92.3/81.1         | 72.4/92.2/79.8 | 73.7/92.3/81.1           |
| simple_pose_resnet101_v1d          | 73.0/92.2/80.8 | 74.2/92.4/82.0         | 73.0/92.2/80.8 | 74.2/92.4/82.0           |
| simple_pose_resnet152_v1b          | 72.4/92.1/79.6 | 74.2/92.3/82.1         | 72.4/92.1/79.6 | 74.2/92.3/82.1           |
| simple_pose_resnet152_v1d          | 73.4/92.3/80.7 | 74.6/93.4/82.1         | 73.4/92.3/80.7 | 74.6/93.4/82.1           |
| simple_pose_resnet152_v1d(384x288) | 74.8/92.3/82.0 | 76.1/92.4/83.2         | 74.8/92.3/82.0 | 76.1/92.4/83.2           |

> Note：default input size is `256x192`

## Training

### Classification

#### CIFAR10

| Model1                 | 1-GPU | 4-GPU |
| ---------------------- | ----- | ----- |
| CIFAR_ResNet20_v1      | 92.0  | 91.1  |
| CIFAR_ResNet56_v1      |       |       |
| CIFAR_ResNet110_v1     |       |       |
| CIFAR_ResNet20_v2      |       |       |
| CIFAR_ResNet56_v2      |       |       |
| CIFAR_ResNet110_v2     |       |       |
| CIFAR_WideResNet16_10  |       |       |
| CIFAR_WideResNet28_10  |       |       |
| CIFAR_WideResNet40_8   |       |       |
| CIFAR_ResNeXt29_16x64d |       |       |

> Note：The difference between 1-GPU and 4-GPU is mainly caused by arguments