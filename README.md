# pytorch-cv

Convert the [gluon-cv](https://github.com/dmlc/gluon-cv/) to pytorch. (And add more models from others) 

> Note: use pytorch-nightly （due to `nn.SyncBatchNorm`）

## Enviroment

```shell
# 1. create new enviroment
conda env create -f environment.yml
source activate ptcv
# 2. install opencv 
conda install -c menpo opencv3
# 3. install pytorch (https://pytorch.org/get-started/locally/)
conda install pytorch-nightly cudatoolkit=9.0 -c pytorch  # choose your cuda version
# # 4. install mxnet and gluoncv (options---for convert pre-trained gluon model)
# pip install mxnet-cu92   # (https://beta.mxnet.io/install.html, choose your cuda version)
# pip install gluoncv --pre --upgrade
```

### build cpp-extension (for object detection)

> Note: it will delete after torchvision merge this [pull-826](https://github.com/pytorch/vision/pull/826)

```shell
cd model
python setup.py build_ext develop
```

## Supported Applications

|                         Application                          |              Illustration               |                       Available Models                       |
| :----------------------------------------------------------: | :-------------------------------------: | :----------------------------------------------------------: |
| **Image Classification:** <br>recognize an object in an image. | ![](NOTE/png/image-classification.png)  |    [CIFAR10](scripts/cifar), [IMAGENET](scripts/imagenet)    |
| **Object Detection:** <br>detect multiple objects with their <br>bounding boxes in an image. |   ![](NOTE/png/object-detection.png)    | [SSD](scripts/ssd), [YOLO3](scripts/yolo), [Faster-RCNN](scripts/faster_rcnn), [CenterNet](scripts/centernet/README.md) |
| **Semantic Segmentation:** <br>associate each pixel of an image <br>with a categorical label. | ![](NOTE/png/semantic-segmentation.png) | [FCN, DeepLabv3, PSPNet, DANet, OCNet, BiSeNet, CCNet](scripts/segmentation) |
| **Instance Segmentation:** <br>detect objects and associate <br>each pixel inside object area with an <br>instance label. | ![](NOTE/png/instance-segmentation.png) |   [Mask RCNN](scripts/mask_rcnn), [YOLACT](scripts/yolact)   |
| **Pose Estimation:** <br>detect human pose <br>from images.  |    ![](NOTE/png/pose-estimation.svg)    |              [Simple-Pose](scripts/simple_pose)              |

## Schedule

Classification: 

|         | demo | eval | train |          | demo | eval | train |
| :-----: | :--: | :--: | :---: | :------: | :--: | :--: | :---: |
| CIFAR10 |  ✓   |  ✓   |   ✓   | IMAGENET |  ✓   |  ✓   |   ✗   |

Semantic Segmentation：training performance nearly same as paper

|              | demo | eval | train |
| :----------: | :--: | :--: | :---: |
| Segmentation |  ✓   |  ✓   |   ✓   |

Object Detection：the training performance nearly same as gluon-cv

|             | demo | eval | train |           | demo | eval | train |
| :---------: | :--: | :--: | :---: | :-------: | :--: | :--: | :---: |
|     SSD     |  ✓   |  ✓   |   ✓   |   YOLO    |  ✓   |  ✓   |   ✓   |
| Faster-RCNN |  ✓   |  ✓   |   ✗   | CenterNet |  ✓   |  ✗   |   ✗   |

Instance Segmentation

|           | demo | eval | train |        | demo | eval | train |
| :-------: | :--: | :--: | :---: | :----: | :--: | :--: | :---: |
| Mask-RCNN |  ✓   |  ✓   |   ✗   | YOLACT |  ✓   |  ✗   |   ✗   |

Pose Estimation

|             | demo | eval | train |      | demo | eval | train |
| :---------: | :--: | :--: | :---: | :--: | :--: | :--: | :---: |
| Simple-Pose |  ✓   |  ✓   |   ✗   |      |      |      |       |

## TODO

- [ ] Provide converted models (BaiduYun and GoogleDrive)
- [ ] Compare the backbone performance with [torchvison](https://github.com/pytorch/vision) and [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) , and replace with better one.
- [ ] Improve performance
- [x] Move sync BN through use `convert_sync_batchnorm`
- [ ] Add more training code and models
- [ ] Move convert code from master to branch