"""Feature extraction blocks.
Feature or Multi-Feature extraction is a key component in object detection.
Class predictor/Box predictor are usually applied on feature layer(s).
A good feature extraction mechanism is critical to performance.
"""
from __future__ import absolute_import

from torch import nn
import torch.nn.functional as F


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
                 use_bn=True, reduce_ratio=1.0, min_depth=128, global_pool=False,
                 pretrained=False, **kwargs):
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


if __name__ == '__main__':
    net = 'mobilenet1.0'
    # outputs = [[6, 5, 0, 7], [7, 2, 0, 7]]
    outputs = [[68], [80]]
    res = _parse_network(net, outputs, pretrained=False)
    print(res[0])
    cnt = 0
    for key in res[0].state_dict().keys():
        if not (key.endswith('num_batches_tracked') or key.endswith('running_mean') or
                key.endswith('running_var')):
            cnt += 1
    print(cnt)
