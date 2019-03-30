# Training

## Segmentation

- [x] 检查MixCrossEntroyLoss是否和mxnet里的一致
- [ ] 查看kvstore的影响
- [ ] 学习率是否应该比gluon-cv里面大GPU的数目倍数
- [ ] 

#### Bugs

- [ ] 当GPU=8的时候，训练同时进行validation存在内存溢出（GPU=4不存在这个问题---单纯训练或者验证也不存在该问题，采用原始Metric也不存在该问题）
- [ ] 存在“同步”慢的问题（超过了5分钟，导致的问题 --- [distributed problem](https://github.com/pytorch/pytorch/issues/16225)）
  - debug：
    1. 问题不是出在`nn.SyncBatchNorm`上面
    2. 