# pytorch-cv

Convert the [gluon-cv](https://github.com/dmlc/gluon-cv/) to pytorch. 

**You can see more detail results in [EVAL](./NOTE/EVAL.md) and [TRAIN](./NOTE/TRAIN.md)** 

> Note: use pytorch-nightly （due to nn.SyncBatchNorm）

## Schedule

- [x] CIFAR10：Demo+Eval+Train


- [x] ImageNet：Demo+Eval
- [x] Segmentation：Demo+Eval+Train
- [x] SSD：Demo+Eval+Train
- [x] YOLOv3：Demo+Eval+Train
- [x] Simple Pose Estimation：Demo+Eval

## Usage

### Using script (in scripts files)

- using [sh_convert.sh](./scripts/sh_convert.sh) to convert model（choose the model you want）
- using [sh_demo.sh](./scripts/sh_convert.sh) to run demo（choose the model you want）
- using [sh_eval.sh](./scripts/sh_eval.sh) to run evaluation（choose the model you want）
- using [sh_eval_distributed.sh](./scripts/sh_eval_distributed.sh) to run evaluation with multi-gpu (choose the model you want)
- using [sh_train.sh](./scripts/sh_train.sh) to run training （choose the model you want）
- using [sh_train_distributed.sh](./scripts/sh_train_distributed.sh) to run training （choose the model you want）

Another way is follow **Demo** and **Evaluation**

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
   ```

You can see the performance (compare with gluon-cv) in [EVAL](./NOTE/EVAL.md).

### Training

Recommend use [sh_train_distributed.sh](./scripts/sh_train_distributed.sh)

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

### BUG

- [x] ~~evaluation:  distributed version is slow~~（rewrite metric in pytorch）
- [ ] Training with validate in new metric have bug (number of GPU=8 will out of memory, nGPU=4 without this problem, guess caused by move between gpu) 