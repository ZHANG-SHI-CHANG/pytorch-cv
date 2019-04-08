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
| ResNet50_v1b_gn | 77.36            | 93.59            |                  |                  |
| ResNet101_v1b   | 79.20            | 94.61            | 78.49            | 94.14            |
| ResNet152_v1b   | 79.69            | 94.74            | 78.87            | 94.41            |
| ResNet50_v1c    | 78.03            | 94.09            | 77.18            | 93.72            |
| ResNet101_v1c   | 79.60            | 94.75            | 78.84            | 94.20            |
| ResNet152_v1c   | 80.01            | 94.96            | 79.53            | 94.57            |
| ResNet50_v1d    | 79.15            | 94.58            | 78.55            | 94.17            |
| ResNet101_v1d   | 80.51            | 95.12            |                  |                  |
| ResNet152_v1d   | 80.61            | 95.34            |                  |                  |
| ResNet18_v2     | 71.00            | 89.92            |                  |                  |
| ResNet34_v2     | 74.40            | 92.08            |                  |                  |
| ResNet50_v2     | 77.11            | 93.43            |                  |                  |
| ResNet101_v2    | 78.53            | 94.17            |                  |                  |
| ResNet152_v2    | 79.21            | 94.31            |                  |                  |

> GN has bugs. 

#### ResNext

| Model | Top-1 (gluon-cv) | Top-5 (gluon-cv) | Top-1 (here-PIL) | Top-5 (here-PIL) |
| ----- | ---------------- | ---------------- | ---------------- | ---------------- |
|       |                  |                  |                  |                  |



## Detection

### VOC

| SSD Model                | gluon-cv | pytorch-cv (PIL) |
| ------------------------ | -------- | ---------------- |
| ssd_300_vgg16_atrous_voc | 77.6     | 74.6             |
| ssd_512_vgg16_atrous_voc | 79.2     | 76.2             |
| ssd_512_resnet50_v1_voc  | 80.1     | 78.0             |
| ssd_512_mobilenet1.0_voc | 75.4     | 72.9             |

| YOLO Model                      | gluon-cv | pytorch-cv (PIL) |
| ------------------------------- | -------- | ---------------- |
| yolo3_darknet53_voc (320x320)   | 79.3     | 78.5             |
| yolo3_darknet53_voc (416x416)   | 81.5     | 80.9             |
| yolo3_mobilenet1.0_voc(320x320) | 73.9     | 72.1             |
| yolo3_mobilenet1.0_voc(416x416) | 75.8     | 74.0             |

### COCO

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

