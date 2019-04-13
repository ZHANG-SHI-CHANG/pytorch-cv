# pytorch-cv

Convert the [gluon-cv](https://github.com/dmlc/gluon-cv/) to pytorch. 

**You can see more detail results in [EVAL](./NOTE/EVAL.md) and [TRAIN](./NOTE/TRAIN.md)** 

> Note: use pytorch-nightly （due to nn.SyncBatchNorm）

## Schedule

> Note：the ✓ of train means the training code is available. However, I did not enough machine to run all the results.

#### classification

|          | demo | train | eval |
| -------- | :--: | :---: | :--: |
| cifar10  |  ✓   |   ✓   |  ✓   |
| imagenet |  ✓   |   ✗   |  ✓   |

#### Segmentation

|         | demo | train | eval |
| ------- | :--: | :---: | :--: |
| fcn     |  ✓   |   ✓   |  ✓   |
| pspnet  |  ✓   |   ✓   |  ✓   |
| deeplab |  ✓   |   ✓   |  ✓   |

#### Object Detection

|             | demo | train | eval |
| ----------- | :--: | :---: | :--: |
| YOLOv3      |  ✓   |   ✓   |  ✓   |
| SSD         |  ✓   |   ✓   |  ✓   |
| Faster-RCNN |  ✓   |   ✗   |  ✓   |

#### Pose Estimation

|             | demo | train | eval |
| ----------- | :--: | :---: | :--: |
| Simple Pose |  ✓   |   ✗   |  ✓   |

#### Instance Segmentation

|           | demo | train | eval |
| --------- | :--: | :---: | :--: |
| Mask RCNN |  ✗   |   ✗   |  ✗   |

## Enviroment

```shell
# 1. create new enviroment
conda env create -f environment.yml
source activate ptcv
# 2. install opencv 
conda install -c menpo opencv3
# 3. install pytorch (https://pytorch.org/get-started/locally/)
conda install pytorch-nightly cudatoolkit=9.0 -c pytorch  # choose your cuda version
# 4. install mxnet and gluoncv (options---for convert pre-trained gluon model)
pip install mxnet-cu92   # (https://beta.mxnet.io/install.html, choose your cuda version)
pip install gluoncv --pre --upgrade
```

### build cpp-extension (for object detection)

> Note: it will delete after torchvision merge this [pull-826](https://github.com/pytorch/vision/pull/826)

```shell
cd model
python setup.py build_ext develop
```

## Usage

### Using script (in scripts files)

- using [sh_convert.sh](./scripts/sh_convert.sh) to convert model（choose the model you want）
- using [sh_demo.sh](./scripts/sh_convert.sh) to run demo（choose the model you want）
- using [sh_eval.sh](./scripts/sh_eval.sh) to run evaluation（choose the model you want）
- using [sh_eval_distributed.sh](./scripts/sh_eval_distributed.sh) to run evaluation with multi-gpu (choose the model you want)
- using [sh_train.sh](./scripts/sh_train.sh) to run training （choose the model you want）
- using [sh_train_distributed.sh](./scripts/sh_train_distributed.sh) to run training （choose the model you want）

Another way is follow **Demo** and **Evaluation** and **Training**

### Demo

> using gluon-cv pretrained model

1. using [gluon2torch](./utils/gluon2torch.py) to convert pretrained gluon model to pytorch

   ```shell
   cd utils
   # convert cifar model for example 
   python gluon2torch.py --name CIFAR_ResNeXt29_16x64d
   cd ..
   ```

   > - `base=False` in classification and `True` in detection
   > - `reorder=True` in ssd (with resnet or mobilenet), others is `False`

2. run demo 

   ```shell
   cd scripts/demo
   # cifar as example
   python demo_cifar10.py --model CIFAR_ResNeXt29_16x64d
   ```

### Evaluation

> using gluon-cv pretrained model

1. using [gluon2torch](./utils/gluon2torch.py) to convert pretrained gluon model to pytorch（If you have done it in demo. ignore this step）

   ```shell
   cd utils
   # convert cifar model for example 
   python gluon2torch.py --name CIFAR_ResNeXt29_16x64d
   cd ..
   ```

2. run evaluate

   ```shell
   cd scritps/eval
   # cifar as example
   python eval_cifar.py --network CIFAR_ResNeXt29_16x64d
   
   ## another option: distributed version (especially recommended for segmentation)
   export NGPUS=4
   python -m torch.distributed.launch --nproc_per_node=$NGPUS eval/eval_cifar10.py --network CIFAR_ResNet20_v1 --batch-size 8 --cuda
   ```

You can see the performance (compare with gluon-cv) in [EVAL](./NOTE/EVAL.md).

### Training

1. using [gluon2torch](./utils/gluon2torch.py) to convert backbone gluon model to pytorch (exclude classification)

2. run training

   ```shell
   cd scripts/train
   # cifar as example
   python train/train_cifar10.py --num-epochs 200 -j 4 --batch-size 128 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --cuda
   
   # another option: distributed version
   export NGPUS=4
   python -m torch.distributed.launch --nproc_per_node=$NGPUS train/train_cifar10.py --model CIFAR_ResNet20_v1 --num-epochs 200 -j 4 --batch-size 128 --wd 0.0001 --lr 0.4 --lr-decay 0.1 --lr-decay-epoch 100,150 --cuda
   ```

   > Note：please change lr to NGPU*one_gpu_lr

## TODO

- [x] add GPU to demo
- [x] add evaluation (in progress)
- [x] add multi-gpu support (in progress)
- [x] rewrite metric（more efficient for distributed）
- [ ] add training
- [ ] add mixup
- [ ] add tutorial
- [ ] modify doc
- [ ] delete duplicated code
- [ ] provide better pre-trained model
- [ ] make code more efficient and use less memory 

### BUG

- [x] ~~evaluation:  distributed version is slow~~（rewrite metric in pytorch）