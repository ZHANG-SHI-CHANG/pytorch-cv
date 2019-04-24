# Semantic Segmentation

**Support Models：**

- [x] FCN
- [x] PSPNet
- [x] DeepLabv3

## Performance

#### Pascal VOC 2012

Here, we using train (10582), val (1449), test (1456) as most paper used. (More detail can reference [DeepLabv3](https://github.com/chenxi116/DeepLabv3.pytorch)) . And the performance is evaluated with single scale

- Base LR 0.001, Base Size 540, Crop Size 480

|        Model        | Paper (val) | Epoch |  val (crop)   |      val      |
| :-----------------: | :---------: | :---: | :-----------: | :-----------: |
|  FCN-ResNet50 (*)   |      /      |  50   | 90.97 / 65.25 | 92.17 / 66.90 |
|  PSPNet-ResNet101   |      /      |  50   | 93.37 / 74.94 | 93.93 / 75.20 |
| DeepLabv3-ResNet101 | no / 77.02  |  50   | 92.57 / 72.04 |               |
|                     |      /      |       |               |               |

> 1. `*` means without aux loss
> 2. crop means crop to 480
> 3. `val` means in validate dataset, `test` means in test dataset
> 4. the metric is `pixAcc/mIoU`
> 5. the `test` label should download from official website

#### Cityscapes





## TODO

- [ ] Improve performance, like using better pre-trained backbone (like [torchvison](https://github.com/pytorch/vision) or [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch))
- [ ] Add more Segmentation methods
- [ ] delete warm-up (set to 0)
- [ ] remove convert code from master to branch