# pytorch-cv

Convert the [gluon-cv](https://github.com/dmlc/gluon-cv/) to pytorch. 

**You can see more detail results in [PROGRESS](./NOTE/PROGRESS.md)**

## Usage

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



## TODO

- [x] add GPU to demo
- [ ] add evaluation (in progress)
- [ ] add training
- [ ] add tutorial
- [ ] add python opencv version (check the "difference" --- note, it's still different with mxnet.image)

