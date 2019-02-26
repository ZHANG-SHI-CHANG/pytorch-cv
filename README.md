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



### Segmentation



## TODO

- [x] add GPU to demo
- [ ] add evaluation
- [ ] add auto-select interp（in `data.transforms.utils`）

