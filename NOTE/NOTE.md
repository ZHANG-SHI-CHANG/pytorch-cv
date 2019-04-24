## 相关操作替换

> 下述前者指mxnet中的操作，后者指pytorch中的操作


1. 取"子集"（比如`a:2x3x4`希望提取出前`2x3x2`）
   - `a.slice_axis(axis=-1, begin=0, end=2)`
   - `a.narrow(dimension=-1, start=0, length=2)`

2. 特定维度裁剪为指定变量的大小（比如`a:2x10x10x4, b:2x3x3x4`，希望提取`a`的前面`2x3x3x4`）
   - `nd.slice_like(a, b*0, axes(1, 2))`
   - `a.narrow(1, 0, b.shape[1]).narrow(2, 0, b.shape[2])`（采用多个`narrow`来完成）

3. 重复多次（比如`x:2x2`，行列分别重复2次和3次，那么得到的就是`x: 4x6`）
   - `nd.tile(a, reps=(2, 3))`
   - `a.repeat(2, 3)`

4. 重新"排列维度"（比如`a:2x3x4`，我们希望调整为`a:4x2x3`）
   - `nd.transpose(a, axes=(2, 0, 1))`
   - `a.permute(2, 0, 1)`

5. 扩展"维度"（比如`a:2x3`，我们希望调整为`a:2x1x3`）
   - `a.expand_dims(1)`
   - `a.unsqueeze(1)`

6. 合并多个tensor（比如`a:2x2, b:3x2, c:4x2`希望合并为`d:9x2`）
   - `nd.concat(a, b, c, dim=0)`
   - `torch.cat([a,b,c], 0)`

     > 对于0dim的情况（比如 `a=torch.Tensr([1.0]), b=torch.Tensor([2.0])`），无法使用`torch.cat`，可以利用`torch.Tensor([a, b])`来合并

7. 合并tensor到新的维度上面（比如`a:2x2, b=2x2, c:2x2`，希望合并为`d:3x2x2`）
   - `nd.stack([a,b,c], 0)`
   - `torch.stack([a,b,c], 0)`

8. 翻转某一维度：（比如`a: 2x3x4`）

   - `a[:,:,::-1]`
   - `a.flip(2)`（注：暂时不支持逆下标）

9. 复制一个张量：

   - `b=a.copy()`
   - `b=a.clone()`

10. 寻找指定维度下，某些下标对应的值：
  - `nd.pick(x, index, axis=)`（注意：对于越界情况默认采用截断！！！）
  - `torch.gather(x, dim, index)`

11. "histogram"：

    - `np.histogram`
    - `torch.histc`

12. 两者取大

    - `np.maximum(a, b)`
    - `torch.max(a, b)`

13. 类型改变

    - `nd.cast(a, 某种类型)`

    - pytorch：`a.type(b.dtype)`

14. 获得"形状"张量：

    - `nd.shape_array(a)`
    - `a.shape`

15. 拆分向量

    - `nd.split(x, axis, num_outputs, squeeze_axis)`：按照某个维度拆分成`num_outputs`个，如果`squeeze_axis`为true的话，该维度拆分后每个为1维则将该维去掉
    - `torch.split(tensor, split_size_or_sections, dim=0)`：按照dim维拆分成每一个该维大小为split_size_or_sections大小

16. 按下标提取元素

    - `nd.gather_nd(data, indices)`
    - `torch.gather(data, dim, index)`

      > 其实也不对，这个话题可以参考：[6445](https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445)，[27402](https://discuss.pytorch.org/t/batched-index-select-tf-gather-nd/27402/8)

17. 获取维度数目
    - `numpy`对象：`a.ndim`
    - `torch`对象：`a.ndimensions()`

### torchvision

1. 左右翻转：

   - `transforms.RandomFlipLeftRight()`

   - `transforms.RandoHorizontalFlip()`



### pytorch分布式

- `data.DistributedSampler`：使得Dataset每次`__getitem__`获得的是多个data（和指定的gpu数目相关---以iter形式）
- 

### 注意事项

1. 对于每次输入大小改变的情况，将`torch.backends.cudnn`设置为`False`，否则会严重影响速度（比如在Faster-RCNN测试等阶段）

## python

- 善用`getattr`：利用属性名从某个类返回其中的某个"属性"（比如某一层网络结构）

- `setattr(name, 模块)`：将模块属性命名为name，方便`getattr`"寻找"

  > 类中对应`self.__setattr__`，`self.__getattr__`

- opencv的图片对应的：`img.shape ---> h, w`； 而PIL对应的则为`img.size--->w, h`