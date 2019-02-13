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
   - `torch.repeat(2, 3)`
4. 重新"排列维度"（比如`a:2x3x4`，我们希望调整为`a:4x2x3`）
   - `nd.transpose(a, axes=(2, 0, 1))`
   - `a.permute(2, 0, 1)`
5. 扩展"维度"（比如`a:2x3`，我们希望调整为`a:2x1x3`）
   - `a.expand_dims(1)`
   - `a.unsqueeze(1)`
6. 合并多个tensor（比如`a:2x2, b:3x2, c:4x2`希望合并为`d:9x2`）
   - `nd.concat(a, b, c, dim=0)`
   - `torch.cat([a,b,c], 0)`

