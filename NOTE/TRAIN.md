# Training Results

## Classification

### CIFAR10

| Model                  | here | gluon-cv |
| ---------------------- | ---- | -------- |
| CIFAR_ResNet20_v1      | 92.0 | 92.1     |
| CIFAR_ResNet56_v1      | 93.4 | 93.6     |
| CIFAR_ResNet110_v1     | 93.9 | 93.0     |
| CIFAR_ResNet20_v2      | 92.2 | 92.1     |
| CIFAR_ResNet56_v2      | 93.5 | 93.7     |
| CIFAR_ResNet110_v2     | 94.1 | 94.3     |
| CIFAR_WideResNet16_10  | 95.2 | 95.1     |
| CIFAR_WideResNet28_10  |      | 95.6     |
| CIFAR_WideResNet40_8   |      | 95.9     |
| CIFAR_ResNeXt29_16x64d |      | 96.3     |

## Segmentation

### ADE20K

| Model            | pixAcc/mIoU（pytorch-cv） | pixAcc/mIoU（gluon-cv） |
| ---------------- | ------------------------- | ----------------------- |
| fcn_resnet50_ade | 78.40/38.58               | 79.0/39.5               |

> Note: `lr=0.01/16*batch_size(per GPU)*NGPU` 

### COCO

| Model              | pixAcc/mIoU | pixAcc/mIoU（gluon-cv） |
| ------------------ | ----------- | ----------------------- |
| fcn_resnet101_coco | 91.11/61.81 | 92.2/66.2               |

## Detection

| Model                    | pytorch-cv     | gluon-cv |
| ------------------------ | -------------- | -------- |
| ssd_300_vgg16_atrous_voc | 72.95  (75.44) | 77.6     |

> 实验对比分析：
>
> 1. 全部都采用`xavier_uniform`初始化：
> 2. 按照和gluon-cv类似的初始化策略：75.20（采用07metric）
