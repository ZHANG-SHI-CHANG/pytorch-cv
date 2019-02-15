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

## TODO

- [x] add GPU to demo
- [ ] add evaluation

