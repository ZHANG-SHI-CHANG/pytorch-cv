# pytorch-cv

Convert the [gluon-cv](https://github.com/dmlc/gluon-cv/) to pytorch. 

**You can see more detail results in [PROGRESS](./NOTE/PROGRESS.md)**

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

You can see the performance (compare with gluon-cv) in [PROGRESS](./NOTE/PROGRESS.md).

## TODO

- [x] add GPU to demo
- [x] add evaluation (in progress)
- [ ] add multi-gpu support (in progress)
- [x] rewrite metric（more efficient for distributed）
- [ ] add training
- [ ] add tutorial
- [ ] modify doc
- [ ] add python opencv version (check the "difference" --- note, it's still different with mxnet.image)

### BUG

- [ ] evaluation:  distributed version is slow
- [ ] segmentation: evaluation is slow（gluon-cv is also slow）