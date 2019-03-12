import warnings
import numpy as np

import torch
from torch import nn
from model.module.basic import _conv2d


def _upsample(x, stride=2):
    """Simple upsampling layer by stack pixel alongside horizontal and vertical directions.
    Parameters
    ----------
    x : mxnet.nd.NDArray or mxnet.symbol.Symbol
        The input array.
    stride : int, default is 2
        Upsampling stride
    """
    d, dims = x.dim(), [t for t in x.shape]
    dims_expand = dims.copy()
    dims_expand.insert(d - 1, stride)
    dims_expand.insert(d + 1, stride)
    dims[d - 1] *= stride
    dims[d - 2] *= stride
    x.unsqueeze_(d - 1).unsqueeze_(d + 1)
    return x.expand(*dims_expand).contiguous().view(*dims)


if __name__ == '__main__':
    a = torch.arange(12).view(1, 2, 3, 2)
    out = _upsample(a)
    print(out)


class YOLODetectionBlockV3(nn.Module):
    """YOLO V3 Detection Block which does the following:
    - add a few conv layers
    - return the output
    - have a branch that do yolo detection.
    Parameters
    ----------
    channel : int
        Number of input channels
    channel : int
        Number of channels for 1x1 conv. 3x3 Conv will have 2*channel.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments.
    """

    def __init__(self, in_channel, channel, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(YOLODetectionBlockV3, self).__init__(**kwargs)
        assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
        self.body = list()
        for _ in range(2):
            # 1x1 reduce
            self.body.append(_conv2d(in_channel, channel, 1, 0, 1,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            # 3x3 expand
            self.body.append(_conv2d(channel, channel * 2, 3, 1, 1,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            in_channel = channel * 2
        self.body.append(_conv2d(in_channel, channel, 1, 0, 1,
                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs))
        self.body = nn.Sequential(*self.body)
        self.tip = _conv2d(channel, channel * 2, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def forward(self, x):
        route = self.body(x)
        tip = self.tip(route)
        return route, tip


class YOLOOutputV3(nn.Module):
    """YOLO output layer V3.
    Parameters
    ----------
    num_class : int
        Number of foreground objects.
    anchors : iterable
        The anchor setting. Reference: https://arxiv.org/pdf/1804.02767.pdf.
    stride : int
        Stride of feature map.
    alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, Scalar, etc.
    """

    def __init__(self, in_channel, num_class, anchors, stride, alloc_size=(128, 128), **kwargs):
        super(YOLOOutputV3, self).__init__(**kwargs)
        anchors = np.array(anchors).astype('float32')
        self._classes = num_class
        self._num_pred = 1 + 4 + num_class  # 1 objness + 4 box + num_class
        self._num_anchors = anchors.size // 2
        self._stride = stride
        all_pred = self._num_pred * self._num_anchors
        self.prediction = nn.Conv2d(in_channel, all_pred, kernel_size=1, padding=0, stride=1)
        # anchors will be multiplied to predictions
        anchors = anchors.reshape(1, 1, -1, 2)
        self.anchors = nn.Parameter(torch.from_numpy(anchors), requires_grad=False)
        # offsets will be added to predictions
        grid_x = np.arange(alloc_size[1]).astype('float32')
        grid_y = np.arange(alloc_size[0]).astype('float32')
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        # stack to (n, n, 2)
        offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
        # expand dims to (1, 1, n, n, 2) so it's easier for broadcasting
        offsets = np.expand_dims(np.expand_dims(offsets, axis=0), axis=0)
        self.offsets = nn.Parameter(torch.from_numpy(offsets), requires_grad=False)

    def forward(self, x):
        """Forward of YOLOV3Output layer.
        Parameters
        ----------
        x : tensor
            Input feature map.
        Returns
        -------
        (tuple of) tensor
            During training, return (bbox, raw_box_centers, raw_box_scales, objness,
            class_pred, anchors, offsets).
            During inference, return detections.
        """
        b = x.shape[0]
        # prediction flat to (batch, pred per pixel, height * width)
        pred = self.prediction(x).view((b, self._num_anchors * self._num_pred, -1))
        # transpose to (batch, height * width, num_anchor, num_pred)
        pred = pred.transpose(2, 1).view((b, -1, self._num_anchors, self._num_pred))
        # components
        raw_box_centers = pred.narrow(-1, 0, 2)
        raw_box_scales = pred.narrow(-1, 2, 2)
        objness = pred.narrow(-1, 4, 1)
        class_pred = pred.narrow(-1, 5, self._classes)

        # valid offsets, (1, 1, height, width, 2)
        offsets = self.offsets.narrow(2, 0, x.shape[2]).narrow(3, 0, x.shape[3])
        # reshape to (1, height*width, 1, 2)
        offsets = offsets.contiguous().view(1, -1, 1, 2)

        box_centers = (torch.sigmoid(raw_box_centers) + offsets) * self._stride
        box_scales = torch.exp(raw_box_scales) * self.anchors
        confidence = torch.sigmoid(objness)
        class_score = torch.sigmoid(class_pred) * confidence
        wh = box_scales / 2.0
        bbox = torch.cat([box_centers - wh, box_centers + wh], dim=-1)

        if self.training:
            # during training, we don't need to convert whole bunch of info to detection results
            return (bbox.reshape((0, -1, 4)), raw_box_centers, raw_box_scales,
                    objness, class_pred, self.anchors, offsets)

        # prediction per class
        bboxes = bbox.repeat(self._classes, 1, 1, 1, 1)
        scores = class_score.permute(3, 0, 1, 2).unsqueeze(4)
        ids = scores * 0 + torch.arange(0, self._classes, dtype=torch.float, device=x.device).view(-1, 1, 1, 1, 1)
        detections = torch.cat([ids, scores, bboxes], dim=-1)
        # reshape to (B, xx, 6)
        detections = detections.permute(1, 0, 2, 3, 4).contiguous().view(b, -1, 6)
        return detections

    def reset_class(self, classes, reuse_weights=None):
        """Reset class prediction.
        Parameters
        ----------
        classes : type
            Description of parameter `classes`.
        reuse_weights : dict
            A {new_integer : old_integer} mapping dict that allows the new predictor to reuse the
            previously trained weights specified by the integer index.
        Returns
        -------
        type
            Description of returned object.
        """
        # keep old records
        old_classes = self._classes
        old_pred = self.prediction
        old_num_pred = self._num_pred
        self._classes = len(classes)
        self._num_pred = 1 + 4 + len(classes)
        all_pred = self._num_pred * self._num_anchors
        # to avoid deferred init, number of in_channels must be defined
        in_channels = old_pred.weight.shape[1]
        device = old_pred.weight.device
        self.prediction = nn.Conv2d(in_channels, all_pred, kernel_size=1, padding=0, stride=1,).to(device)
        if reuse_weights:
            new_pred = self.prediction
            assert isinstance(reuse_weights, dict)
            for old_params, new_params in zip(old_pred.parameters(), new_pred.parameters()):
                old_data = old_params.data
                new_data = new_params.data
                for k, v in reuse_weights.items():
                    if k >= self._classes or v >= old_classes:
                        warnings.warn("reuse mapping {}/{} -> {}/{} out of range".format(
                            k, self._classes, v, old_classes))
                        continue
                    for i in range(self._num_anchors):
                        off_new = i * self._num_pred
                        off_old = i * old_num_pred
                        # copy along the first dimension
                        new_data[1 + 4 + k + off_new] = old_data[1 + 4 + v + off_old]
                        # copy non-class weights as well
                        new_data[off_new : 1 + 4 + off_new] = old_data[off_old : 1 + 4 + off_old]
                # set data to new conv layers
                new_params.data = new_data

# if __name__ == '__main__':
#     num_class = 20
#     anchors = [116, 90, 156, 198, 373, 326]
#     stride = 32
#     alloc_size = (128, 128)
#     op = YOLOOutputV3(1024, num_class, anchors, stride, alloc_size)
#     op.eval()
#
#     x = torch.randn(1, 1024, 16, 16)
#     out = op(x)
#     print(out.shape)
