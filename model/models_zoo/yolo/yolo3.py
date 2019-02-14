"""You Only Look Once Object Detection v3"""
from __future__ import absolute_import
from __future__ import division

import os

import torch
from torch import nn

from model.module.nms import box_nms
from model.module.basic import _conv2d
from model.models_zoo.yolo.darknet import darknet53
from model.models_zoo.mobilenet import get_mobilenet
from model.models_zoo.yolo.yolo_module import _upsample, YOLOOutputV3, YOLODetectionBlockV3

__all__ = ['YOLOV3', 'get_yolov3',
           'yolo3_darknet53_voc',
           'yolo3_mobilenet1_0_voc',
           'yolo3_darknet53_coco',
           'yolo3_mobilenet1_0_coco',
           ]


class YOLOV3(nn.Module):
    """YOLO V3 detection network.
    Reference: https://arxiv.org/pdf/1804.02767.pdf.
    Parameters
    ----------
    stages : nn.Module
        Staged feature extraction blocks.
        For example, 3 stages and 3 YOLO output layers are used original paper.
    channels : iterable
        Number of conv channels for each appended stage.
        `len(channels)` should match `len(stages)`.
    num_class : int
        Number of foreground objects.
    anchors : iterable
        The anchor setting. `len(anchors)` should match `len(stages)`.
    strides : iterable
        Strides of feature map. `len(strides)` should match `len(stages)`.
    alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, Scalar, etc.
    nms_thresh : float, default is 0.45.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    pos_iou_thresh : float, default is 1.0
        IOU threshold for true anchors that match real objects.
        'pos_iou_thresh < 1' is not implemented.
    ignore_iou_thresh : float
        Anchors that has IOU in `range(ignore_iou_thresh, pos_iou_thresh)` don't get
        penalized of objectness score.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, stages, out_channels, block_channels, channels, anchors, strides, classes,
                 alloc_size=(128, 128), nms_thresh=0.45, nms_topk=400, post_nms=100, pos_iou_thresh=1.0,
                 ignore_iou_thresh=0.7, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(YOLOV3, self).__init__(**kwargs)
        self._classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self._pos_iou_thresh = pos_iou_thresh
        self._ignore_iou_thresh = ignore_iou_thresh
        # TODO: add training phase
        # if pos_iou_thresh >= 1:
        #     self._target_generator = YOLOV3TargetMerger(len(classes), ignore_iou_thresh)
        # else:
        #     raise NotImplementedError(
        #         "pos_iou_thresh({}) < 1.0 is not implemented!".format(pos_iou_thresh))
        # self._loss = YOLOV3Loss()
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.yolo_blocks = nn.ModuleList()
        self.yolo_outputs = nn.ModuleList()
        # note that anchors and strides should be used in reverse order
        for i, stage, out_channel, block_channel, channel, anchor, stride in zip(
                range(len(stages)), stages, out_channels, block_channels, channels, anchors[::-1], strides[::-1]):
            self.stages.append(stage)
            block = YOLODetectionBlockV3(block_channel, channel, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.yolo_blocks.append(block)
            output = YOLOOutputV3(out_channel, len(classes), anchor, stride, alloc_size=alloc_size)
            self.yolo_outputs.append(output)
            if i > 0:
                self.transitions.append(_conv2d(out_channel, channel, 1, 0, 1, norm_layer=norm_layer,
                                                norm_kwargs=norm_kwargs))

    def forward(self, x, *args):
        """YOLOV3 network forward.
        Parameters
        ----------
        x : tensor
            Input data.
        *args : optional, tensor
            During training, extra inputs are required:
            (gt_boxes, obj_t, centers_t, scales_t, weights_t, class_t)
            These are generated by YOLOV3PrefetchTargetGenerator in dataloader transform function.
        Returns
        -------
        (tuple of) tensor
            During inference, return detections in shape (B, N, 6)
            with format (cid, score, xmin, ymin, xmax, ymax)
            During training, return losses only: (obj_loss, center_loss, scale_loss, cls_loss).
        """
        all_box_centers = []
        all_box_scales = []
        all_objectness = []
        all_class_pred = []
        all_anchors = []
        all_offsets = []
        all_feat_maps = []
        all_detections = []
        routes = []
        for stage in self.stages:
            x = stage(x)
            routes.append(x)

        # the YOLO output layers are used in reverse order, i.e., from very deep layers to shallow
        for i, block, output in zip(range(len(routes)), self.yolo_blocks, self.yolo_outputs):
            x, tip = block(x)
            if self.training:
                pass  # TODO
                # dets, box_centers, box_scales, objness, class_pred, anchors, offsets = output(tip)
                # all_box_centers.append(box_centers.reshape((0, -3, -1)))
                # all_box_scales.append(box_scales.reshape((0, -3, -1)))
                # all_objectness.append(objness.reshape((0, -3, -1)))
                # all_class_pred.append(class_pred.reshape((0, -3, -1)))
                # all_anchors.append(anchors)
                # all_offsets.append(offsets)
                # # here we use fake featmap to reduce memory consuption, only shape[2, 3] is used
                # fake_featmap = F.zeros_like(tip.slice_axis(
                #     axis=0, begin=0, end=1).slice_axis(axis=1, begin=0, end=1))
                # all_feat_maps.append(fake_featmap)
            else:
                dets = output(tip)
            all_detections.append(dets)
            if i >= len(routes) - 1:
                break
            # add transition layers
            x = self.transitions[i](x)
            # upsample feature map reverse to shallow layers
            upsample = _upsample(x, stride=2)
            route_now = routes[::-1][i + 1]
            x = torch.cat([upsample.narrow(2, 0, route_now.shape[2]).narrow(3, 0, route_now.shape[3]),
                           route_now], dim=1)

        # if autograd.is_training():
        #     # during training, the network behaves differently since we don't need detection results
        #     if autograd.is_recording():
        #         # generate losses and return them directly
        #         box_preds = F.concat(*all_detections, dim=1)
        #         all_preds = [F.concat(*p, dim=1) for p in [
        #             all_objectness, all_box_centers, all_box_scales, all_class_pred]]
        #         all_targets = self._target_generator(box_preds, *args)
        #         return self._loss(*(all_preds + all_targets))
        #
        #     # return raw predictions, this is only used in DataLoader transform function.
        #     return (F.concat(*all_detections, dim=1), all_anchors, all_offsets, all_feat_maps,
        #             F.concat(*all_box_centers, dim=1), F.concat(*all_box_scales, dim=1),
        #             F.concat(*all_objectness, dim=1), F.concat(*all_class_pred, dim=1))

        # concat all detection results from different stages
        result = torch.cat(all_detections, dim=1)
        # apply nms per class
        if 1 > self.nms_thresh > 0:
            result = box_nms(result, overlap_thresh=self.nms_thresh, topk=self.nms_topk,
                             score_index=1, coord_start=2, id_index=0)
            if self.post_nms > 0:
                result = result.narrow(1, 0, self.post_nms)
        ids = result.narrow(-1, 0, 1)
        scores = result.narrow(-1, 1, 1)
        bboxes = result.narrow(-1, 2, 4)
        return ids, scores, bboxes

    @property
    def num_class(self):
        """Number of (non-background) categories.
        Returns
        -------
        int
            Number of (non-background) categories.
        """
        return self._num_class

    @property
    def classes(self):
        """Return names of (non-background) categories.
        Returns
        -------
        iterable of str
            Names of (non-background) categories.
        """
        return self._classes

    def set_nms(self, nms_thresh=0.45, nms_topk=400, post_nms=100):
        """Set non-maximum suppression parameters.
        Parameters
        ----------
        nms_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        post_nms : int, default is 100
            Only return top `post_nms` detection results, the rest is discarded. The number is
            based on COCO dataset which has maximum 100 objects per image. You can adjust this
            number if expecting more objects. You can use -1 to return all detections.
        Returns
        -------
        None
        """
        # self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    # def reset_class(self, classes):
    #     """Reset class categories and class predictors.
    #     Parameters
    #     ----------
    #     classes : iterable of str
    #         The new categories. ['apple', 'orange'] for example.
    #     """
    #     self._clear_cached_op()
    #     self._classes = classes
    #     if self._pos_iou_thresh >= 1:
    #         self._target_generator = YOLOV3TargetMerger(len(classes), self._ignore_iou_thresh)
    #     for outputs in self.yolo_outputs:
    #         outputs.reset_class(classes)


def get_yolov3(name, stages, out_channels, block_channels, filters, anchors, strides, classes, dataset,
               pretrained=False, root=os.path.join(os.path.expanduser('~'), '.torch', 'models'), **kwargs):
    """Get YOLOV3 models.
    Parameters
    ----------
    name : str or None
        Model name, if `None` is used, you must specify `features` to be a `nn.Module`.
    stages : iterable of str or `nn.Module`
        List of network internal output names, in order to specify which layers are
        used for predicting bbox values.
        If `name` is `None`, `features` must be a `nn.Module` which generate multiple
        outputs for prediction.
    filters : iterable of float or None
        List of convolution layer channels which is going to be appended to the base
        network feature extractor. If `name` is `None`, this is ignored.
    sizes : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SSD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer. Its length must be equals
        to the number of SSD output layers.
    steps : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of categories.
    dataset : str
        Name of dataset. This is used to identify model name because models trained on
        different datasets are going to be very different.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    root : str
        Model weights storing path.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    HybridBlock
        A YOLOV3 detection network.
    """
    net = YOLOV3(stages, out_channels, block_channels, filters, anchors, strides, classes=classes, **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        full_name = '_'.join(('yolo3', name, dataset))
        net.load_state_dict(torch.load(get_model_file(full_name, root=root)))
    return net


def yolo3_darknet53_voc(pretrained_base=True, pretrained=False,
                        norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with darknet53 base network on VOC dataset.
    Parameters
    ----------
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    from data.pascal_voc.detection import VOCDetection
    pretrained_base = False if pretrained else pretrained_base
    base_net = darknet53(
        pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    stages = [base_net.features[:15], base_net.features[15:24], base_net.features[24:]]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    classes = VOCDetection.CLASSES
    return get_yolov3('darknet53', stages, [1024, 512, 256], [1024, 768, 384], [512, 256, 128], anchors, strides,
                      classes, 'voc', pretrained=pretrained, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)


def yolo3_darknet53_coco(pretrained_base=True, pretrained=False,
                         norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with darknet53 base network on COCO dataset.
    Parameters
    ----------
    pretrained_base : boolean
        Whether fetch and load pretrained weights for base network.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    from data.mscoco.detection import COCODetection
    pretrained_base = False if pretrained else pretrained_base
    base_net = darknet53(pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    stages = [base_net.features[:15], base_net.features[15:24], base_net.features[24:]]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    classes = COCODetection.CLASSES
    return get_yolov3('darknet53', stages, [1024, 512, 256], [1024, 768, 384], [512, 256, 128],
                      anchors, strides, classes, 'coco', pretrained=pretrained,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)


def yolo3_mobilenet1_0_voc(pretrained_base=True, pretrained=False,
                           norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with mobilenet base network on VOC dataset.
    Parameters
    ----------
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    from data.pascal_voc.detection import VOCDetection

    pretrained_base = False if pretrained else pretrained_base
    base_net = get_mobilenet(multiplier=1, pretrained=pretrained_base,
                             norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    stages = [base_net.features[:33],
              base_net.features[33:69],
              base_net.features[69:]]

    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    classes = VOCDetection.CLASSES
    return get_yolov3('mobilenet1.0', stages, [1024, 512, 256], [1024, 768, 384], [512, 256, 128],
                      anchors, strides, classes, 'voc', pretrained=pretrained,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)


def yolo3_mobilenet1_0_coco(pretrained_base=True, pretrained=False,
                            norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with mobilenet base network on COCO dataset.
    Parameters
    ----------
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    from data.mscoco.detection import COCODetection

    pretrained_base = False if pretrained else pretrained_base
    base_net = get_mobilenet(multiplier=1, pretrained=pretrained_base,
                             norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    stages = [base_net.features[:33],
              base_net.features[33:69],
              base_net.features[69:]]

    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    classes = COCODetection.CLASSES
    return get_yolov3('mobilenet1.0', stages, [1024, 512, 256], [1024, 768, 384], [512, 256, 128],
                      anchors, strides, classes, 'coco', pretrained=pretrained,
                      norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)


if __name__ == '__main__':
    net = yolo3_darknet53_voc(pretrained_base=False)
    a = torch.randn(1, 3, 416, 416)
    net.eval()
    out = net(a)
    print(out[0].shape)
