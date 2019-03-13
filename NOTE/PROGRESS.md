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

| ADE20K Dataset        | pixAcc(gluon-cv) | mIoU(gluon-cv) | pixAcc(pytorch-cv) | mIoU(pytorch-cv) |
| --------------------- | ---------------- | -------------- | ------------------ | ---------------- |
| fcn_resnet50_ade      | 79.0 (75.8)      | 39.5 (36.4)    | (75.7)             | (36.2)           |
| fcn_resnet101_ade     | 80.6 (78.0)      | 41.6 (39.3)    | (77.9)             | (39.0)           |
| psp_resnet50_ade      | 80.1 (78.2)      | 41.5 (40.6)    | (78.0)             | (40.1)           |
| psp_resnet101_ade     | 80.8 (78.9)      | 43.3 (40.9)    | (78.8)             | (40.9)           |
| deeplab_resnet50_ade  | 80.5 (78.6)      | 42.5 (41.4)    | (78.6)             | (41.2)           |
| deeplab_resnet101_ade | 81.1 (79.2)      | 44.1 (42.5)    | (79.1)             | (42.3)           |

> 1. value in () means after using base_size=520, crop_size=480.  
> 2. pytorch-cv using PIL.Image

| VOC Dataset           | pixAcc(gluon-cv) | mIoU(gluon-cv) | pixAcc(pytorch-cv) | mIoU(pytorch-cv) |
| --------------------- | ---------------- | -------------- | ------------------ | ---------------- |
| fcn_resnet101_voc     |                  | 83.6           |                    |                  |
| psp_resnet101_voc     |                  | 85.1           |                    |                  |
| deeplab_resnet101_voc |                  | 86.2           |                    |                  |
| deeplab_resnet152_voc |                  | 86.7           |                    |                  |
| psp_resnet101_citys   |                  | 77.1           |                    |                  |
