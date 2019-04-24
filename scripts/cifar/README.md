# CIFAR10

## Performance

The metric is accuracy：`acc1/acc2` means `Vanilla/Mixup` （more detail please see [cifar10](https://gluon-cv.mxnet.io/model_zoo/classification.html#cifar10)）

|         Model          | Gluon-CV (Vanilla/Mixup) | Here (Convert Mixup) | Here (train) |
| :--------------------: | :----------------------: | :------------------: | :----------: |
|   CIFAR_ResNet20_v1    |       92.1 / 92.9        |       [92.9]()       |     92.0     |
|   CIFAR_ResNet56_v1    |       93.6 / 94.2        |       [94.2]()       |     93.4     |
|   CIFAR_ResNet110_v1   |       93.0 / 95.2        |       [95.2]()       |     93.9     |
|   CIFAR_ResNet20_v2    |       92.1 / 92.7        |       [92.7]()       |     92.2     |
|   CIFAR_ResNet56_v2    |       93.7 / 94.6        |       [94.6]()       |     93.5     |
|   CIFAR_ResNet110_v2   |       94.3 / 95.5        |       [95.5]()       |     94.1     |
| CIFAR_WideResNet16_10  |       95.1 / 96.7        |       [96.7]()       |     95.2     |
| CIFAR_WideResNet28_10  |       95.6 / 97.2        |       [97.1]()       |              |
|  CIFAR_WideResNet40_8  |       95.9 / 97.3        |       [97.3]()       |              |
| CIFAR_ResNeXt29_16x64d |       96.3 / 97.3        |       [97.3]()       |              |

> Note：The training here is Vanilla version

## Demo

Recognize give image. (Please download pre-trained model to `~/.torch/models` first. --- If you put pre-trained model to other folder, please change the `--root`)

```shell
$ python demo_cifar10_pil.py [--model CIFAR_ResNet20_v1] [--input-pic <image>.jpg] [--cuda]
```

## Evaluation

The default data root is `~/.torch/datasets` (You can build a soft-link to it or change `--data-root`)

```shell
$ python eval_cifar10_pil.py [--network CIFAR_ResNet20_v1] [--batch-size 8] [--cuda]
```

## Train

```shell
$ python train_cifar10_pil.py [--model CIFAR_ResNet20_v1] [--batch-size 128] [--cuda]
```

> Note：more parameters can change in `train_cifar10_pil.py`

