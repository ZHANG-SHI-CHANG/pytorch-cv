# Training Results

## Classification

### CIFAR10

| Model1                 | 1-GPU | 4-GPU | gluon-cv |
| ---------------------- | ----- | ----- | -------- |
| CIFAR_ResNet20_v1      | 92.0  | 91.1  | 92.1     |
| CIFAR_ResNet56_v1      |       |       | 93.6     |
| CIFAR_ResNet110_v1     |       |       | 93.0     |
| CIFAR_ResNet20_v2      |       |       | 92.1     |
| CIFAR_ResNet56_v2      |       |       | 93.7     |
| CIFAR_ResNet110_v2     |       |       | 94.3     |
| CIFAR_WideResNet16_10  |       |       | 95.1     |
| CIFAR_WideResNet28_10  |       |       | 95.6     |
| CIFAR_WideResNet40_8   |       |       | 95.9     |
| CIFAR_ResNeXt29_16x64d |       |       | 96.3     |

> using: lr = nGPU * lr，other arguments is same.

## Segmentation

### ADE20K

