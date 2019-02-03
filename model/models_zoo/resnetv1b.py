"""ResNetV1bs, implemented in PyTorch."""
from __future__ import division

import os
from torch import nn
import torch.nn.functional as F

__all__ = ['ResNetV1b', 'resnet18_v1b', 'resnet34_v1b',
           'resnet50_v1b', 'resnet101_v1b',
           'resnet152_v1b', 'BasicBlockV1b', 'BottleneckV1b',
           'resnet50_v1c', 'resnet101_v1c', 'resnet152_v1c',
           'resnet50_v1d', 'resnet101_v1d', 'resnet152_v1d',
           'resnet50_v1e', 'resnet101_v1e', 'resnet152_v1e',
           'resnet50_v1s', 'resnet101_v1s', 'resnet152_v1s']


# -----------------------------------------------------------------------------
# BLOCKS & BOTTLENECK
# -----------------------------------------------------------------------------
class BasicBlockV1b(nn.Module):
    """ResNetV1b BasicBlockV1b"""
    expansion = 1

    def __init__(self, in_channel, planes, strides=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=None, norm_kwargs=None, **kwargs):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, planes, kernel_size=3, stride=strides,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes, **({} if norm_kwargs is None else norm_kwargs))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=previous_dilation,
                               dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes, **({} if norm_kwargs is None else norm_kwargs))
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.strides = strides

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu2(out)

        return out


class BottleneckV1b(nn.Module):
    """ResNetV1b BottleneckV1b"""
    expansion = 4

    def __init__(self, in_channel, planes, strides=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None,
                 norm_kwargs=None, last_gamma=False, **kwargs):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, **({} if norm_kwargs is None else norm_kwargs))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=strides,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes, **({} if norm_kwargs is None else norm_kwargs))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        if not last_gamma:
            self.bn3 = norm_layer(planes * 4, **({} if norm_kwargs is None else norm_kwargs))
        else:
            self.bn3 = norm_layer(planes * 4, **({} if norm_kwargs is None else norm_kwargs))
            nn.init.zeros_(self.bn3.weight)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.strides = strides

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu3(out)

        return out


# -----------------------------------------------------------------------------
# NETS
# -----------------------------------------------------------------------------
class ResNetV1b(nn.Module):
    """ Pre-trained ResNetV1b Model, which produces the strides of 8
    feature maps at conv5.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.


    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    def __init__(self, block, layers, classes=1000, dilated=False, norm_layer=nn.BatchNorm2d,
                 norm_kwargs=None, last_gamma=False, deep_stem=False, stem_width=32,
                 avg_down=False, final_drop=0.0, use_global_stats=False, **kwargs):
        channel = [64, 64, 128, 256] if block is BasicBlockV1b else [64, 256, 512, 1024]
        self.basic = block is BasicBlockV1b
        self.inplanes = stem_width * 2 if deep_stem else 64
        super(ResNetV1b, self).__init__(**kwargs)
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            norm_kwargs['track_running_stats'] = True
        self.norm_kwargs = norm_kwargs
        if not deep_stem:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = list()
            self.conv1.append(nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False))
            self.conv1.append(norm_layer(stem_width, **({} if norm_kwargs is None else norm_kwargs)))
            self.conv1.append(nn.ReLU(inplace=True))
            self.conv1.append(nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False))
            self.conv1.append(norm_layer(stem_width, **({} if norm_kwargs is None else norm_kwargs)))
            self.conv1.append(nn.ReLU(inplace=True))
            self.conv1.append(nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1,
                                        padding=1, bias=False))
            self.conv1 = nn.Sequential(*self.conv1)
        self.bn1 = norm_layer(stem_width * 2, **({} if norm_kwargs is None else norm_kwargs))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channel[0], 64, layers[0], avg_down=avg_down,
                                       norm_layer=norm_layer, last_gamma=last_gamma)
        self.layer2 = self._make_layer(block, channel[1], 128, layers[1], strides=2, avg_down=avg_down,
                                       norm_layer=norm_layer, last_gamma=last_gamma)
        if dilated:
            self.layer3 = self._make_layer(block, channel[2], 256, layers[2], strides=1, dilation=2,
                                           avg_down=avg_down, norm_layer=norm_layer,
                                           last_gamma=last_gamma)
            self.layer4 = self._make_layer(block, channel[3], 512, layers[3], strides=1, dilation=4,
                                           avg_down=avg_down, norm_layer=norm_layer,
                                           last_gamma=last_gamma)
        else:
            self.layer3 = self._make_layer(block, channel[2], 256, layers[2], strides=2,
                                           avg_down=avg_down, norm_layer=norm_layer,
                                           last_gamma=last_gamma)
            self.layer4 = self._make_layer(block, channel[3], 512, layers[3], strides=2,
                                           avg_down=avg_down, norm_layer=norm_layer,
                                           last_gamma=last_gamma)

        self.drop = None
        if final_drop > 0.0:
            self.drop = nn.Dropout(final_drop)
        self.fc = nn.Linear(512 * block.expansion, classes)

    def _make_layer(self, block, in_channel, planes, blocks, strides=1, dilation=1,
                    avg_down=False, norm_layer=None, last_gamma=False):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = list()
            if avg_down:
                if dilation == 1:
                    downsample.append(nn.AvgPool2d(kernel_size=strides, stride=strides,
                                                   ceil_mode=True, count_include_pad=False))
                else:
                    downsample.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                   ceil_mode=True, count_include_pad=False))
                downsample.append(nn.Conv2d(in_channel, planes * block.expansion, kernel_size=1,
                                            stride=1, bias=False))
                downsample.append(norm_layer(planes * block.expansion, **self.norm_kwargs))
            else:
                downsample.append(nn.Conv2d(in_channel, planes * block.expansion,
                                            kernel_size=1, stride=strides, bias=False))
                downsample.append(norm_layer(planes * block.expansion, **self.norm_kwargs))
            downsample = nn.Sequential(*downsample)
        layers = list()
        if dilation in (1, 2):
            layers.append(block(in_channel, planes, strides, dilation=1,
                                downsample=downsample, previous_dilation=dilation,
                                norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                                last_gamma=last_gamma))
        elif dilation == 4:
            layers.append(block(in_channel, planes, strides, dilation=2,
                                downsample=downsample, previous_dilation=dilation,
                                norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                                last_gamma=last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes if self.basic else planes * 4, planes, dilation=dilation,
                                previous_dilation=dilation, norm_layer=norm_layer,
                                norm_kwargs=self.norm_kwargs, last_gamma=last_gamma))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, x.shape[2]).squeeze(3).squeeze(2)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)

        return x


# -----------------------------------------------------------------------------
# Constructor
# -----------------------------------------------------------------------------
def resnet18_v1b(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1b-18 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.
    """
    model = ResNetV1b(BasicBlockV1b, [2, 2, 2, 2], **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%db' % (18, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet34_v1b(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1b-34 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.
    """
    model = ResNetV1b(BasicBlockV1b, [3, 4, 6, 3], **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%db' % (34, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1b(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1b-50 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%db' % (50, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet101_v1b(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1b-101 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%db' % (101, 1),
                                                        root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet152_v1b(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1b-152 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.
    """
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%db' % (152, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1c(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1c-50 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`).
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%dc' % (50, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


if __name__ == '__main__':
    net = resnet50_v1c()
    print(net)
    # print(net.block)


def resnet101_v1c(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1c-101 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`).
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%dc' % (101, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet152_v1c(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1b-152 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`).
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], deep_stem=True, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%dc' % (152, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1d(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1d-50 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`).
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, avg_down=True, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%dd' % (50, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet101_v1d(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1d-101 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`).
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, avg_down=True, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%dd' % (101, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet152_v1d(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1d-152 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`).
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], deep_stem=True, avg_down=True, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%dd' % (152, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1e(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1e-50 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`).
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3],
                      deep_stem=True, avg_down=True, stem_width=64, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%de' % (50, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet101_v1e(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1e-101 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`).
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3],
                      deep_stem=True, avg_down=True, stem_width=64, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%de' % (101, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet152_v1e(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1e-152 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`).
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3],
                      deep_stem=True, avg_down=True, stem_width=64, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%de' % (152, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1s(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1s-50 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`).
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, stem_width=64, **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%ds' % (50, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet101_v1s(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1s-101 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`).
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, stem_width=64,
                      **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%ds' % (101, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet152_v1s(pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch/models'), **kwargs):
    """Constructs a ResNetV1s-152 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNetV1b, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`).
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    """
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], deep_stem=True, stem_width=64,
                      **kwargs)
    if pretrained:
        import torch
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet%d_v%ds' % (152, 1), root=root)))
        from data.imagenet import ImageNetAttr
        attrib = ImageNetAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model
