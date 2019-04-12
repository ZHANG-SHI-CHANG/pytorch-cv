# Evaluation

> Note: 
>
> 1. pytorch-cv using the pretrained model from gluon-cv
> 2. the difference is mainly caused by **image process**: in gluon-cv, use mxnet.image (which is based on OpenCV --- not the CV2 Python library.), and pytorch-cv use PIL.Image or CV2 Python library. 

## Classification

### CIFAR10

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

### ImageNet

#### ResNet

| Model           | Top-1 (gluon-cv) | Top-5 (gluon-cv) | Top-1 (here-PIL) | Top-5 (here-PIL) |
| --------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| ResNet18_v1     | 70.93            | 89.92            | 69.53            | 89.11            |
| ResNet34_v1     | 74.37            | 91.87            | 73.20            | 91.40            |
| ResNet50_v1     | 77.36            | 93.57            | 76.31            | 93.00            |
| ResNet101_v1    | 78.34            | 94.01            | 77.49            | 93.58            |
| ResNet152_v1    | 79.22            | 94.64            | 78.42            | 94.21            |
| ResNet18_v1b    | 70.94            | 89.83            | 69.57            | 89.11            |
| ResNet34_v1b    | 74.65            | 92.08            | 73.52            | 91.49            |
| ResNet50_v1b    | 77.67            | 93.82            | 76.87            | 93.20            |
| ResNet50_v1b_gn | 77.36            | 93.59            | 76.62            | 93.05            |
| ResNet101_v1b   | 79.20            | 94.61            | 78.49            | 94.14            |
| ResNet152_v1b   | 79.69            | 94.74            | 78.87            | 94.41            |
| ResNet50_v1c    | 78.03            | 94.09            | 77.18            | 93.72            |
| ResNet101_v1c   | 79.60            | 94.75            | 78.84            | 94.20            |
| ResNet152_v1c   | 80.01            | 94.96            | 79.53            | 94.57            |
| ResNet50_v1d    | 79.15            | 94.58            | 78.55            | 94.17            |
| ResNet101_v1d   | 80.51            | 95.12            | 79.79            | 94.88            |
| ResNet152_v1d   | 80.61            | 95.34            | 80.20            | 95.00            |
| ResNet18_v2     | 71.00            | 89.92            | 69.49            | 89.09            |
| ResNet34_v2     | 74.40            | 92.08            | 73.51            | 91.42            |
| ResNet50_v2     | 77.11            | 93.43            | 76.27            | 92.89            |
| ResNet101_v2    | 78.53            | 94.17            | 77.89            | 93.76            |
| ResNet152_v2    | 79.21            | 94.31            | 78.45            | 94.07            |

#### ResNext

| Model               | Top-1 (gluon-cv) | Top-5 (gluon-cv) | Top-1 (here-PIL) | Top-5 (here-PIL) |
| ------------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| ResNext50_32x4d     | 79.32            | 94.53            | 78.79            | 94.15            |
| ResNext101_32x4d    | 80.37            | 95.06            | 79.74            | 94.61            |
| ResNext101_64x4d    | 80.69            | 95.17            | 80.03            | 94.71            |
| SE_ResNext50_32x4d  | 79.95            | 94.93            | 79.56            | 94.68            |
| SE_ResNext101_32x4d | 80.91            | 95.39            | 80.67            | 95.01            |
| SE_ResNext101_64x4d | 81.01            | 95.32            | 80.60            | 95.14            |

#### MobileNet

| Model            | Top-1 (gluon-cv) | Top-5 (gluon-cv) | Top-1 (here-PIL) | Top-5 (here-PIL) |
| ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| MobileNet1.0     | 73.28            | 91.30            | 72.31            | 90.64            |
| MobileNet0.75    | 70.25            | 89.49            | 69.00            | 88.76            |
| MobileNet0.5     | 65.20            | 86.34            | 63.52            | 85.12            |
| MobileNet0.25    | 52.91            | 76.94            | 50.81            | 75.03            |
| MobileNetV2_1.0  | 71.92            | 90.56            | 70.14            | 89.38            |
| MobileNetV2_0.75 | 69.61            | 88.95            | 67.84            | 87.84            |
| MobileNetV2_0.5  | 64.49            | 85.47            | 62.76            | 84.15            |
| MobileNetV2_0.25 | 50.74            | 74.56            | 49.35            | 73.28            |

#### VGG

| Model    | Top-1 (gluon-cv) | Top-5 (gluon-cv) | Top-1 (here-PIL) | Top-5 (here-PIL) |
| -------- | ---------------- | ---------------- | ---------------- | ---------------- |
| VGG11    | 66.62            | 87.34            | 50.93            | 75.21            |
| VGG13    | 67.74            | 88.11            | 51.40            | 75.51            |
| VGG16    | 73.23            | 91.31            | 59.43            | 82.24            |
| VGG19    | 74.11            | 91.35            | 62.56            | 84.23            |
| VGG11_bn | 68.59            | 88.72            | 61.88            | 84.10            |
| VGG13_bn | 68.84            | 88.82            | 63.54            | 85.57            |
| VGG16_bn | 73.10            | 91.76            | 62.51            | 84.64            |
| VGG19_bn | 74.33            | 91.85            | 65.10            | 86.51            |

> ! so much drop !!! :cry:

#### SqueezeNet

| Model         | Top-1 (gluon-cv) | Top-5 (gluon-cv) | Top-1 (here-PIL) | Top-5 (here-PIL) |
| ------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| SqueezeNet1.0 | 56.11            | 79.09            | 56.84            | 79.31            |
| SqueezeNet1.1 | 54.96            | 78.17            | 57.01            | 79.54            |

#### DenseNet

| Model       | Top-1 (gluon-cv) | Top-5 (gluon-cv) | Top-1 (here-PIL) | Top-5 (here-PIL) |
| ----------- | ---------------- | ---------------- | ---------------- | ---------------- |
| DenseNet121 | 74.97            | 92.25            | 73.67            | 91.55            |
| DenseNet161 | 77.70            | 93.80            | 76.71            | 93.32            |
| DenseNet169 | 76.17            | 93.17            | 75.19            | 92.46            |
| DenseNet201 | 77.32            | 93.62            | 76.15            | 92.98            |

#### Pruned ResNet

| Model              | Top-1 (gluon-cv) | Top-5 (gluon-cv) | Top-1 (here-PIL) | Top-5 (here-PIL) |
| ------------------ | ---------------- | ---------------- | ---------------- | ---------------- |
| resnet18_v1b_0.89  | 67.2             | 87.45            | 65.71            | 86.68            |
| resnet50_v1d_0.86  | 78.02            | 93.82            | 76.84            | 93.41            |
| resnet50_v1d_0.48  | 74.66            | 92.34            | 73.72            | 91.64            |
| resnet50_v1d_0.37  | 70.71            | 89.74            | 69.33            | 88.82            |
| resnet50_v1d_0.11  | 63.22            | 84.79            | 61.26            | 83.31            |
| resnet101_v1d_0.76 | 79.46            | 94.69            | 78.96            | 94.29            |
| resnet101_v1d_0.73 | 78.89            | 94.48            | 78.10            | 93.97            |

#### Others

| Model       | Top-1 (gluon-cv) | Top-5 (gluon-cv) | Top-1 (here-PIL) | Top-5 (here-PIL) |
| ----------- | ---------------- | ---------------- | ---------------- | ---------------- |
| AlexNet     | 54.92            | 78.03            | 54.85            | 78.04            |
| darknet53   | 78.56            | 94.43            | 78.05            | 94.12            |
| InceptionV3 | 78.77            | 94.39            | 72.56            | 90.76            |
| SENet_154   | 81.26            | 95.51            | 81.00            | 95.38            |



## Detection

> Note：there are two version nms in this our achievement --- one (box_nms_py) without  valid_thresh and another (box_nms) is more likely [torchvison nms](https://github.com/pytorch/vision/pull/826)
>
> - v1：PIL+box_nms_py
> - v2：PIL+box_nms
> - v3：opencv(python version)+box_nms
>

### VOC

| SSD Model                | gluon-cv | pytorch-cv (v1) | pytorch-cv (v3) |
| ------------------------ | -------- | --------------- | --------------- |
| ssd_300_vgg16_atrous_voc | 77.6     | 74.6            | 77.5            |
| ssd_512_vgg16_atrous_voc | 79.2     | 76.2            | 79.1            |
| ssd_512_resnet50_v1_voc  | 80.1     | 78.0            | 80.4            |
| ssd_512_mobilenet1.0_voc | 75.4     | 72.9            | 75.5            |

| YOLO Model                      | gluon-cv | pytorch-cv (v1) | pytorch-cv (v3) |
| ------------------------------- | -------- | --------------- | --------------- |
| yolo3_darknet53_voc (320x320)   | 79.3     | 78.5            | 79.3            |
| yolo3_darknet53_voc (416x416)   | 81.5     | 80.9            | 81.4            |
| yolo3_mobilenet1.0_voc(320x320) | 73.9     | 72.1            | 74.0            |
| yolo3_mobilenet1.0_voc(416x416) | 75.8     | 74.0            | 76.1            |

| Faster-RCNN Model            | gluon-cv | pytorch-cv (v3) |
| ---------------------------- | -------- | --------------- |
| faster_rcnn_resnet50_v1b_voc | 78.3     |                 |

### COCO

| SSD Model                 | gluon-cv       | pytorch-cv(PIL) | pytorch-cv (*) |
| ------------------------- | -------------- | --------------- | -------------- |
| ssd_300_vgg16_atrous_coco | 25.1/42.9/25.8 | 23.8/40.2/24.5  | 25.0/42.6/25.8 |
| ssd_512_vgg16_atrous_coco | 28.9/47.9/30.6 | 27.7/45.5/29.4  | 29.0/48.0/30.6 |
| ssd_512_resnet50_v1_coco  | 30.6/50.0/32.2 | 28.4/46.7/29.7  | 29.7/49.1/31.1 |
| ssd_512_mobilenet1.0_coco | 21.7/39.2/21.3 | 19.9/36.6/19.7  | 20.8/38.4/20.3 |

| YOLO Model                        | gluon-cv       | pytorch-cv (v1) | pytorch-cv (v3) |
| --------------------------------- | -------------- | --------------- | --------------- |
| yolo3_darknet53_coco (320x320)    | 33.6/54.1/35.8 | 32.3/51.8/34.5  | 33.6/54.1/35.8  |
| yolo3_darknet53_coco (416x416)    | 36.0/57.2/38.7 | 34.9/55.2/37.7  | 36.0/57.2/38.7  |
| yolo3_darknet53_coco (608x608)    | 37.0/58.2/40.1 | 35.9/56.2/39.0  | 37.0/58.2/40.1  |
| yolo3_mobilenet1.0_coco (320x320) | 26.7/46.1/27.5 | 25.4/43.6/26.1  | 26.7/46.1/27.5  |
| yolo3_mobilenet1.0_coco (416x416) | 28.6/48.9/29.9 | 27.6/46.8/28.9  | 28.6/48.9/29.9  |
| yolo3_mobilenet1.0_coco (608x608) | 28.0/49.8/27.8 | 27.1/48.1/27.1  | 28.0/49.8/27.8  |

## Segmentation

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
| fcn_resnet101_coco     | 92.2             | 66.2           | 92.2               | 66.2             |
| psp_resnet101_coco     | 92.4             | 70.4           | 92.4               | 70.4             |
| deeplab_resnet101_coco | 92.5             | 70.4           | 92.5               | 70.4             |

| VOC Dataset           | pixAcc(gluon-cv) | mIoU(gluon-cv) | pixAcc(pytorch-cv) | mIoU(pytorch-cv) |
| --------------------- | ---------------- | -------------- | ------------------ | ---------------- |
| fcn_resnet101_voc     |                  | 83.6           |                    |                  |
| psp_resnet101_voc     |                  | 85.1           |                    |                  |
| deeplab_resnet101_voc |                  | 86.2           |                    |                  |
| deeplab_resnet152_voc |                  | 86.7           |                    |                  |
| psp_resnet101_citys   |                  | 77.1           |                    |                  |

## Pose Estimation

### Simple Pose with ResNet

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

