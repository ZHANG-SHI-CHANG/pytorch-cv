"""Base Model for Semantic Segmentation"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from model.models_zoo.resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s
from utils.parallel import tuple_map


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : nn.Module
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`).
    """

    def __init__(self, nclass, aux, backbone='resnet50', height=None, width=None,
                 base_size=520, crop_size=480, pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        self.aux = aux
        self.nclass = nclass
        if backbone == 'resnet50':
            pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=True, **kwargs)
        elif backbone == 'resnet101':
            pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=True, **kwargs)
        elif backbone == 'resnet152':
            pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=True, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        self.conv1 = pretrained.conv1
        self.bn1 = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.layer1 = pretrained.layer1
        self.layer2 = pretrained.layer2
        self.layer3 = pretrained.layer3
        self.layer4 = pretrained.layer4
        height = height if height is not None else crop_size
        width = width if width is not None else crop_size
        self._up_kwargs = (height, width)
        self.base_size = base_size
        self.crop_size = crop_size

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        return c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        h, w = x.shape[2:]
        self._up_kwargs = (h, w)
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


class SegEvalModel(object):
    """Segmentation Eval Module"""

    def __init__(self, module, device):
        self.module = module
        self.module = self.module.to(device)
        self.module.eval()

    def __call__(self, *inputs, **kwargs):
        with torch.no_grad():
            return self.module.evaluate(*inputs, **kwargs)

    def collect_params(self):
        return self.module.state_dict()


class MultiEvalModel(object):
    """Multi-size Segmentation Evaluator"""

    def __init__(self, module, nclass, device, flip=True,
                 scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75]):
        self.flip = flip
        self.device = device
        self.base_size = module.base_size
        self.crop_size = module.crop_size
        self.nclass = nclass
        self.scales = scales
        self.evalmodule = SegEvalModel(module, device)

    # TODO: support multiple gpu
    def parallel_forward(self, inputs, device):
        inputs = tuple([tuple([x.to(device)]) for x in inputs])
        return tuple_map(self(*inputs[0]))

    def forward(self, inputs):
        return self(inputs)

    def __call__(self, image):
        # only single image is supported for evaluation
        batch, _, h, w = image.shape
        assert (batch == 1)
        base_size = self.base_size
        crop_size = self.crop_size
        stride_rate = 2.0 / 3.0
        stride = int(crop_size * stride_rate)
        scores = torch.zeros((batch, self.nclass, h, w), device=image.device)
        for scale in self.scales:
            long_size = int(math.ceil(base_size * scale))
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height
            # resize image to current size
            cur_img = _resize_image(image, height, width)
            if long_size <= crop_size:
                pad_img = _pad_image(cur_img, crop_size)
                outputs = self.flip_inference(pad_img)
                outputs = _crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < crop_size:
                    # pad if needed
                    pad_img = _pad_image(cur_img, crop_size)
                else:
                    pad_img = cur_img
                _, _, ph, pw = pad_img.shape
                assert (ph >= height and pw >= width)
                # grid forward and normalize
                h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
                outputs = torch.zeros((batch, self.nclass, ph, pw), device=image.device)
                count_norm = torch.zeros((batch, 1, ph, pw), device=image.device)
                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + crop_size, ph)
                        w1 = min(w0 + crop_size, pw)
                        crop_img = _crop_image(pad_img, h0, h1, w0, w1)
                        # pad if needed
                        pad_crop_img = _pad_image(crop_img, crop_size)
                        output = self.flip_inference(pad_crop_img)
                        outputs[:, :, h0:h1, w0:w1] += _crop_image(
                            output, 0, h1 - h0, 0, w1 - w0)
                        count_norm[:, :, h0:h1, w0:w1] += 1
                assert ((count_norm == 0).sum() == 0)
                outputs = outputs / count_norm
                outputs = outputs[:, :, :height, :width]

            score = _resize_image(outputs, h, w)
            scores += score

        return scores

    def flip_inference(self, image):
        assert (isinstance(image, torch.Tensor))
        output = self.evalmodule(image)
        if self.flip:
            fimg = _flip_image(image)
            foutput = self.evalmodule(fimg)
            output += _flip_image(foutput)
        return output.exp()

    def collect_params(self):
        return self.evalmodule.collect_params()


def _resize_image(img, h, w):
    return F.interpolate(img, (h, w), mode='bilinear')


def _crop_image(img, h0, h1, w0, w1):
    return img[:, :, h0:h1, w0:w1]


def _flip_image(img):
    assert (img.ndimension() == 4)
    return img.flip(3)


def _pad_image(img, crop_size=480):
    b, c, h, w = img.shape
    assert (c == 3)
    padh = crop_size - h if h < crop_size else 0
    padw = crop_size - w if w < crop_size else 0
    mean = [.485, .456, .406]
    std = [.229, .224, .225]
    pad_values = -np.array(mean) / np.array(std)
    img_pad = torch.zeros((b, c, h + padh, w + padw), device=img.device)
    for i in range(c):
        img_pad[:, i, :, :] = torch.squeeze(
            F.pad(img[:, i, :, :].unsqueeze(1), (0, padw, 0, padh),
                  mode='constant', value=pad_values[i]))
    assert (img_pad.shape[2] >= crop_size and img_pad.shape[3] >= crop_size)
    return img_pad


if __name__ == '__main__':
    img = torch.ones(1, 3, 400, 400)
    out = _pad_image(img)
    print(out)
