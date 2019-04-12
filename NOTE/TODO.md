# Training

## Segmentation

- [x] 检查MixCrossEntroyLoss是否和mxnet里的一致
- [ ] 查看kvstore的影响
- [ ] 学习率是否应该比gluon-cv里面大GPU的数目倍数
- [ ] 

#### Bugs

- [ ] 当GPU=8的时候，训练同时进行validation存在内存溢出（GPU=4不存在这个问题---单纯训练或者验证也不存在该问题，采用原始Metric也不存在该问题）



### SSD

- [x] ~~训练时测试阶段速度明显太慢了！！！比单纯测试代码慢了一大截！~~（这主要是前期bbox中重叠的框框很少，所以非极大值抑制迭代的次数很多，并不是代码bug） --- 所以训练一段时间后再引入验证比较合适
- [ ] 增大lr会出现nan，可以采用小的lr先预训练再调大？

