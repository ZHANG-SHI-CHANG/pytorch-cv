"""Feature extraction blocks.
Feature or Multi-Feature extraction is a key component in object detection.
Class predictor/Box predictor are usually applied on feature layer(s).
A good feature extraction mechanism is critical to performance.
"""
from __future__ import absolute_import

from torch import nn
import torch.nn.functional as F
from utils.init import xavier_uniform_init, mxnet_xavier_normal_init


def _parse_network(network, outputs, pretrained, **kwargs):
    """Parse network with specified outputs and other arguments.

    Parameters
    ----------
    network : str or nn.Module
        Logic chain: load from gluoncv.model_zoo if network is string.
        Convert to Symbol if network is HybridBlock
    outputs : str or iterable of str
        The name of layers to be extracted as features.
    pretrained : bool
        Use pretrained parameters as in model_zoo

    Returns
    -------
    results: list of nn.Module (the same size as len(outputs))

    """
    l, n = len(outputs), len(outputs[0])
    results = [[] for _ in range(l)]
    if isinstance(network, str):
        from model.model_zoo import get_model
        network = get_model(network, pretrained=pretrained, **kwargs).features

    # helper func
    def recursive(pos, block, arr, j):
        if j == n:
            results[pos].append([block])
            return
        child = list(block.children())
        results[pos].append(child[:arr[j]])
        if pos + 1 < l: results[pos + 1].append(child[arr[j] + 1:])
        recursive(pos, child[arr[j]], arr, j + 1)

    block = list(network.children())

    for i in range(l):
        pos = outputs[i][0]
        if i == 0:
            results[i].append(block[:pos])
        elif i < l:
            results[i].append(block[outputs[i - 1][0] + 1: pos])
        recursive(i, block[pos], outputs[i], 1)

    for i in range(l):
        results[i] = nn.Sequential(*[item for sub in results[i] for item in sub if sub])
    return results


class FeatureExpander(nn.Module):
    """Feature extractor with additional layers to append.
    This is very common in vision networks where extra branches are attached to
    backbone network.

    Parameters
    ----------
    network : str or nn.Module
        Logic chain: load from model_zoo if network is string.
    outputs : list or list of list
        The position of layers to be extracted as features

    num_filters : list of int
        Number of filters to be appended. (Note: )
    use_1x1_transition : bool
        Whether to use 1x1 convolution between attached layers. It is effective
        reducing network size.
    use_bn : bool
        Whether to use BatchNorm between attached layers.
    reduce_ratio : float
        Channel reduction ratio of the transition layers.
    min_depth : int
        Minimum channel number of transition layers.
    global_pool : bool
        Whether to use global pooling as the last layer.
    pretrained : bool
        Use pretrained parameters as in model_zoo if `True`.

    """

    def __init__(self, network, outputs, num_filters, channels=[1024, 2048], use_1x1_transition=True,
                 use_bn=True, reduce_ratio=1.0, min_depth=128, global_pool=False, pretrained=False, **kwargs):
        super(FeatureExpander, self).__init__()
        self.features = nn.ModuleList(_parse_network(network, outputs, pretrained, **kwargs))
        self.channel = channels + num_filters[1:]
        self.extras = list()
        for i, f in enumerate(num_filters[1:]):
            extra = list()
            if use_1x1_transition:
                num_trans = max(min_depth, int(round(f * reduce_ratio)))
                extra.append(nn.Conv2d(num_filters[i], num_trans, 1, bias=not use_bn))
                if use_bn:
                    extra.append(nn.BatchNorm2d(num_trans))
                extra.append(nn.ReLU(inplace=True))
            extra.append(nn.Conv2d(num_trans, f, 3, stride=2, padding=1, bias=not use_bn))
            if use_bn:
                extra.append(nn.BatchNorm2d(f))
            extra.append(nn.ReLU(inplace=True))
            self.extras.append(nn.Sequential(*extra))
        self.extras = nn.ModuleList(self.extras)
        self.global_pool = global_pool

    def forward(self, x):
        output = list()
        for feat in self.features:
            x = feat(x)
            output.append(x)
        for extra in self.extras:
            x = extra(x)
            output.append(x)
        if self.global_pool:
            x = F.avg_pool2d(x, x.shape[2])
            output.append(x)
        return output

    def _weight_init(self):
        # self.extras.apply(xavier_uniform_init)
        self.extras.apply(mxnet_xavier_normal_init)


class FPNFeatureExpander(nn.Module):
    """Feature extractor with additional layers to append.
    This is specified for ``Feature Pyramid Network for Object Detection``
    which implement ``Top-down pathway and lateral connections``.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluon.model_zoo.vision if network is string.
        Convert to Symbol if network is HybridBlock.
    outputs : str or list of str
        The name of layers to be extracted as features
    num_filters : list of int e.g. [256, 256, 256, 256]
        Number of filters to be appended.
    use_1x1 : bool
        Whether to use 1x1 convolution
    use_upsample : bool
        Whether to use upsample
    use_elewadd : float
        Whether to use element-wise add operation
    use_p6 : bool
        Whther use P6 stage, this is used for RPN experiments in ori paper
    no_bias : bool
        Whether use bias for Convolution operation.
    norm_layer : HybridBlock or SymbolBlock
        Type of normalization layer.
    norm_kwargs : dict
        Arguments for normalization layer.
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo if `True`.
    ctx : Context
        The context, e.g. mxnet.cpu(), mxnet.gpu(0).
    inputs : list of str
        Name of input variables to the network.

    """

    # TODO: add weight init --- here version is due to "converted model"
    def __init__(self, network, outputs, channels, num_filters, use_1x1=True, use_upsample=True,
                 use_elewadd=True, use_p6=False, use_bias=False, use_relu=False, version='v1',
                 pretrained=False, norm_layer=None, norm_kwargs=None):
        super(FPNFeatureExpander, self).__init__()
        self.features = nn.ModuleList(_parse_network(network, outputs, pretrained))
        extras1 = [[] for _ in range(len(self.features))]
        extras2 = [[] for _ in range(len(self.features))]

        if norm_kwargs is None:
            norm_kwargs = {}
        # e.g. For ResNet50, the feature is :
        # outputs = ['stage1_activation2', 'stage2_activation3',
        #            'stage3_activation5', 'stage4_activation2']
        # with regard to [conv2, conv3, conv4, conv5] -> [C2, C3, C4, C5]
        # append more layers with reversed order : [P5, P4, P3, P2]

        # num_filter is 256 in ori paper
        for i, (extra1, extra2, c, f) in enumerate(zip(extras1, extras2, channels, num_filters)):
            if use_1x1:
                extra1.append(nn.Conv2d(c, f, kernel_size=(1, 1), padding=(0, 0),
                                        stride=1, bias=use_bias))
                if norm_layer is not None:
                    extra1.append(norm_layer(f, **norm_kwargs))
            # Reduce the aliasing effect of upsampling described in ori paper
            extra2.append(nn.Conv2d(f, f, kernel_size=(3, 3), padding=(1, 1), stride=1,
                                    bias=use_bias))
            if norm_layer is not None:
                extra2.append(norm_layer(f, **norm_kwargs))
            if use_relu:
                extra2.append(nn.ReLU(inplace=True))
        self.extras1 = nn.ModuleList([nn.Sequential(*ext) for ext in extras1])
        self.extras2 = nn.ModuleList([nn.Sequential(*ext) for ext in extras2])
        if use_p6:
            if norm_layer is not None:
                self.extra = nn.Sequential(nn.Conv2d(f, f, kernel_size=(3, 3), padding=(1, 1),
                                                     stride=2, bias=use_bias),
                                           norm_layer(f, **norm_kwargs))
            else:
                self.extra = nn.Conv2d(f, f, kernel_size=(3, 3), padding=(1, 1),
                                       stride=2, bias=use_bias)
        self.use_upsample, self.use_elewadd = use_upsample, use_elewadd
        self.use_p6, self.version = use_p6, version

    def forward(self, x):
        feat_list = list()
        for feat in self.features:
            x = feat(x)
            feat_list.append(x)

        outputs, num = list(), len(feat_list)
        for i in range(num - 1, -1, -1):
            if i == num - 1:
                y = self.extras1[i](feat_list[i])
                if self.use_p6:
                    outputs.append(self.extra(y))
            else:
                bf = self.extras1[i](feat_list[i])
                if self.version == 'v1':
                    if self.use_upsample:
                        y = F.interpolate(y, scale_factor=2, mode='nearest')
                    if self.use_elewadd:
                        y = bf + y[..., :bf.shape[2], : bf.shape[3]]
                else:
                    y = F.interpolate(y, size=(bf.shape[2], bf.shape[3]), mode='bilinear', align_corners=False)
                    y = bf + y
            outputs.append(self.extras2[i](y))

        return outputs[::-1]


if __name__ == '__main__':
    net = 'resnet50_v1b'
    # outputs = [[6, 5, 0, 7], [7, 2, 0, 7]]
    # outputs = [[68], [80]]
    outputs = [[4, 2, 5], [5, 3, 5], [6, 5, 5], [7, 2, 5]]
    res = _parse_network(net, outputs, pretrained=False)
    print(res[2])
    print(res[3])
    # cnt = 0
    # for key in res[0].state_dict().keys():
    #     if not (key.endswith('num_batches_tracked') or key.endswith('running_mean') or
    #             key.endswith('running_var')):
    #         cnt += 1
    # print(cnt)
