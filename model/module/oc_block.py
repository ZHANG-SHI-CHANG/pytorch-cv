# copy from https://github.com/PkuRainBow/OCNet.pytorch
import torch
from torch import nn
import torch.nn.functional as F


class _SelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels), nn.ReLU(inplace=True)
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(self.in_channels, self.value_channels, 1, stride=1, padding=0)
        self.W = nn.Conv2d(self.value_channels, self.out_channels, 1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        b, _, h, w = x.size()
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(b, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(b, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(b, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(b, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)
        return context


class BaseOCModule(nn.Module):
    """
    Implementation of the BaseOC module
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
        super(BaseOCModule, self).__init__()
        self.stages = nn.ModuleList(
            [_SelfAttentionBlock(in_channels, key_channels, value_channels, out_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class BaseOCContextModule(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, sizes=([1])):
        super(BaseOCContextModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [_SelfAttentionBlock(in_channels, key_channels, value_channels, out_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(context)
        return output


# ASPP+OC
class ASPOCModule(nn.Module):
    def __init__(self, features, out_features=512, dilations=(12, 24, 36)):
        super(ASPOCModule, self).__init__()
        self.context = nn.Sequential(nn.Conv2d(features, out_features, kernel_size=3, padding=1, dilation=1, bias=True),
                                     nn.BatchNorm2d(out_features), nn.ReLU(inplace=True),
                                     BaseOCContextModule(in_channels=out_features, out_channels=out_features,
                                                         key_channels=out_features // 2, value_channels=out_features,
                                                         sizes=([2])))
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, out_features, 1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_features), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(out_features), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_features), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, out_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_features), nn.ReLU(inplace=True))

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_features), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    @staticmethod
    def _cat_each(feat1, feat2, feat3, feat4, feat5):
        assert (len(feat1) == len(feat2))
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i], feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')

        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)

        if isinstance(x, torch.Tensor):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')

        output = self.conv_bn_dropout(out)
        return output


# Pyramid+OC
class _PyramidSelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_PyramidSelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels), nn.ReLU(inplace=True)
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(self.in_channels, self.value_channels, 1, stride=1, padding=0)
        self.W = nn.Conv2d(self.value_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()

        local_x = []
        local_y = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == (self.scale - 1):
                    end_x = h
                if j == (self.scale - 1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        local_list = []
        local_block_cnt = 2 * self.scale * self.scale
        for i in range(0, local_block_cnt, 2):
            value_local = value[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            query_local = query[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]
            key_local = key[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]]

            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(b, self.value_channels, -1)
            value_local = value_local.permute(0, 2, 1)

            query_local = query_local.contiguous().view(b, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(b, self.key_channels, -1)

            sim_map = torch.matmul(query_local, key_local)
            sim_map = (self.key_channels ** -.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.matmul(sim_map, value_local)
            context_local = context_local.permute(0, 2, 1).contiguous()
            context_local = context_local.view(b, self.value_channels, h_local, w_local)
            local_list.append(context_local)

        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                row_tmp.append(local_list[j + i * self.scale])
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        context = self.W(context)

        return context


class PyramidOCModule(nn.Module):
    """
    Output the combination of the context features and the original features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, dropout, sizes=([1])):
        super(PyramidOCModule, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.stages = nn.ModuleList(
            [_PyramidSelfAttentionBlock(in_channels, out_channels, in_channels // 2, in_channels, size)
             for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels * self.group, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        self.up_dr = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * self.group, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels * self.group), nn.ReLU(inplace=True)
        )

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = [self.up_dr(feats)]
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn_dropout(torch.cat(context, 1))
        return output
