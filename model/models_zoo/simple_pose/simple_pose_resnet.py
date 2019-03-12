from __future__ import division

import os
import torch
from torch import nn

__all__ = ['get_simple_pose_resnet', 'SimplePoseResNet',
           'simple_pose_resnet18_v1b', 'simple_pose_resnet50_v1b',
           'simple_pose_resnet101_v1b', 'simple_pose_resnet152_v1b',
           'simple_pose_resnet50_v1d', 'simple_pose_resnet101_v1d',
           'simple_pose_resnet152_v1d']


class SimplePoseResNet(nn.Module):
    def __init__(self, base_name='resnet50_v1b',
                 pretrained_base=False,
                 num_joints=17,
                 num_deconv_layers=3,
                 num_deconv_filters=(2048, 256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 final_conv_kernel=1, deconv_with_bias=False, **kwargs):
        super(SimplePoseResNet, self).__init__(**kwargs)

        from model.model_zoo import get_model
        base_network = get_model(base_name, pretrained=pretrained_base,
                                 norm_layer=nn.BatchNorm2d)

        self.resnet = list()
        if base_name.endswith('v1'):
            for layer in ['features']:
                self.resnet.append(getattr(base_network, layer))
        else:
            for layer in ['conv1', 'bn1', 'relu', 'maxpool',
                          'layer1', 'layer2', 'layer3', 'layer4']:
                self.resnet.append(getattr(base_network, layer))
        self.resnet = nn.Sequential(*self.resnet)

        self.deconv_with_bias = deconv_with_bias

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(num_deconv_layers, num_deconv_filters, num_deconv_kernels, )

        self.final_layer = nn.Conv2d(
            num_deconv_filters[-1], num_joints,
            kernel_size=final_conv_kernel, stride=1,
            padding=1 if final_conv_kernel == 3 else 0,
        )

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters) - 1, \
            'ERROR: num_deconv_layers is different from len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different from len(num_deconv_filters)'

        layer = list()
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i + 1]
            layer.append(nn.ConvTranspose2d(
                num_filters[i], planes,
                kernel_size=kernel, stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias))
            layer.append(nn.BatchNorm2d(planes))
            layer.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.resnet(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x


def get_simple_pose_resnet(base_name, pretrained=False,
                           root=os.path.expanduser('~/.torch/models'), **kwargs):
    net = SimplePoseResNet(base_name, **kwargs)

    if pretrained:
        from model.model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file('simple_pose_%s' % base_name, root=root)))

    return net


def simple_pose_resnet18_v1b(**kwargs):
    r"""ResNet-18 backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '$TORCH_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet18_v1b', **kwargs)


def simple_pose_resnet50_v1b(**kwargs):
    r"""ResNet-50 backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '$TORCH_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet50_v1b', **kwargs)


def simple_pose_resnet101_v1b(**kwargs):
    r"""ResNet-101 backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '$TORCH_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet101_v1b', **kwargs)


def simple_pose_resnet152_v1b(**kwargs):
    r"""ResNet-152 backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '$TORCH_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet152_v1b', **kwargs)


def simple_pose_resnet50_v1d(**kwargs):
    r"""ResNet-50-d backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '$TORCH_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet50_v1d', **kwargs)


def simple_pose_resnet101_v1d(**kwargs):
    r"""ResNet-101-d backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '$TORCH_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet101_v1d', **kwargs)


def simple_pose_resnet152_v1d(**kwargs):
    r"""ResNet-152-d backbone model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '$TORCH_HOME/models'
        Location for keeping the model parameters.
    """
    return get_simple_pose_resnet('resnet152_v1d', **kwargs)
