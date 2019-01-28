"""Single-shot Multi-box Detector."""
from __future__ import absolute_import

import os
import torch
from torch import nn
import torch.nn.functional as F

from .anchor import SSDAnchorGenerator
from model.module.predictor import ConvPredictor
from model.module.features import FeatureExpander
from model.module.nms import box_nms
from model.module.coder import MultiPerClassDecoder, NormalizedBoxCenterDecoder
from model.models_zoo.ssd.vgg_atrous import vgg16_atrous_300, vgg16_atrous_512
from data.pascal_voc.detection import VOCDetection

__all__ = ['SSD', 'get_ssd',
           # vgg + voc
           'ssd_300_vgg16_atrous_voc',
           'ssd_512_vgg16_atrous_voc',
           # resnet + voc
           'ssd_512_resnet50_v1_voc',
           ]


class SSD(nn.Module):
    """Single-shot Object Detection Network: https://arxiv.org/abs/1512.02325.

    Parameters
    ----------
    network : string or None
        Name of the base network, if `None` is used, will instantiate the
        base network from `features` directly instead of composing.
    base_size : int
        Base input size, it is specified so SSD can support dynamic input shapes.
    features : list of str or nn.Module
        Intermediate features to be extracted or a network with multi-output.
        If `network` is `None`, `features` is expected to be a multi-output network.
    num_filters : list of int
        Number of channels for the appended layers, ignored if `network`is `None`.
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
        Names of all categories.
    use_1x1_transition : bool
        Whether to use 1x1 convolution as transition layer between attached layers,
        it is effective reducing model capacity.
    use_bn : bool
        Whether to use BatchNorm layer after each attached convolution layer.
    reduce_ratio : float
        Channel reduce ratio (0, 1) of the transition layer.
    min_depth : int
        Minimum channels for the transition layers.
    global_pool : bool
        Whether to attach a global average pooling layer as the last output layer.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
    stds : tuple of float, default is (0.1, 0.1, 0.2, 0.2)
        Std values to be divided/multiplied to box encoded values.
    nms_thresh : float, default is 0.45.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    anchor_alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `anchor_alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, scalar, etc.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
        This will only apply to base networks that has `norm_layer` specified, will ignore if the
        base network (e.g. VGG) don't accept this argument.
    norm_kwargs : dict
        Additional `norm_layer` arguments

    """

    def __init__(self, network, base_size, features, num_filters, channels, sizes, ratios,
                 steps, classes, use_1x1_transition=True, use_bn=True,
                 reduce_ratio=1.0, min_depth=128, global_pool=False, pretrained=False,
                 stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.45, nms_topk=400, post_nms=100,
                 anchor_alloc_size=128, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(SSD, self).__init__(**kwargs)
        if norm_kwargs is None:
            norm_kwargs = {}
        if network is None:
            num_layers = len(ratios)
        else:
            num_layers = len(features) + len(num_filters) - 1 + int(global_pool)
        assert len(sizes) == num_layers + 1
        sizes = list(zip(sizes[:-1], sizes[1:]))
        assert isinstance(ratios, list), "Must provide ratios as list or list of list"
        if not isinstance(ratios[0], (tuple, list)):
            ratios = ratios * num_layers  # propagate to all layers if use same ratio
        assert num_layers == len(sizes) == len(ratios), \
            "Mismatched (number of layers) vs (sizes) vs (ratios): {}, {}, {}".format(
                num_layers, len(sizes), len(ratios))
        assert num_layers > 0, "SSD require at least one layer, suggest multiple."
        self._num_layers = num_layers
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

        # TODO: 还是存在bug
        if network is None:
            # use fine-grained manually designed block as features
            try:
                self.features = features(pretrained=pretrained, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            except TypeError:
                self.features = features(pretrained=pretrained)
        else:
            try:
                self.features = FeatureExpander(
                    network=network, outputs=features, num_filters=num_filters,
                    channels=channels, use_1x1_transition=use_1x1_transition,
                    use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                    global_pool=global_pool, pretrained=pretrained,
                    norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            except TypeError:
                self.features = FeatureExpander(
                    network=network, outputs=features, num_filters=num_filters,
                    channels=channels, use_1x1_transition=use_1x1_transition,
                    use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                    global_pool=global_pool, pretrained=pretrained)
        self.class_predictors = nn.ModuleList()
        self.box_predictors = nn.ModuleList()
        self.anchor_generators = nn.ModuleList()
        asz = anchor_alloc_size
        im_size = (base_size, base_size)
        for channel, s, r, st in zip(self.features.channel, sizes, ratios, steps):
            anchor_generator = SSDAnchorGenerator(im_size, s, r, st, (asz, asz))
            self.anchor_generators.append(anchor_generator)
            asz = max(asz // 2, 16)  # pre-compute larger than 16x16 anchor map
            num_anchors = anchor_generator.num_depth
            self.class_predictors.append(ConvPredictor(channel, num_anchors * (len(self.classes) + 1)))
            self.box_predictors.append(ConvPredictor(channel, num_anchors * 4))
        self.bbox_decoder = NormalizedBoxCenterDecoder(stds)
        self.cls_decoder = MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)

    @property
    def num_classes(self):
        """Return number of foreground classes.

        Returns
        -------
        int
            Number of foreground classes

        """
        return len(self.classes)

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
        # self.clear()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    def forward(self, x):
        features = self.features(x)
        b = x.shape[0]
        # print(len(features))
        cls_preds = [(cp(feat).permute(0, 2, 3, 1)).flatten(1)
                     for feat, cp in zip(features, self.class_predictors)]
        box_preds = [(bp(feat).permute(0, 2, 3, 1)).flatten(1)
                     for feat, bp in zip(features, self.box_predictors)]
        anchors = [ag(feat).view(1, -1)
                   for feat, ag in zip(features, self.anchor_generators)]
        cls_preds = torch.cat(cls_preds, dim=1).view((b, -1, self.num_classes + 1))
        box_preds = torch.cat(box_preds, dim=1).view((b, -1, 4))
        anchors = torch.cat(anchors, dim=1).view((1, -1, 4))
        if self.training:
            return [cls_preds, box_preds, anchors]
        bboxes = self.bbox_decoder(box_preds, anchors)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_preds, -1))
        results = []
        for i in range(self.num_classes):
            cls_id = cls_ids.narrow(-1, i, 1)
            score = scores.narrow(-1, i, 1)
            # per class results
            per_result = torch.cat([cls_id, score, bboxes], dim=-1)
            results.append(per_result)
        result = torch.cat(results, dim=1)
        if 1 > self.nms_thresh > 0:
            result = box_nms(result, overlap_thresh=self.nms_thresh, topk=self.nms_topk,
                             score_index=1, coord_start=2, id_index=0)
            if self.post_nms > 0:
                result = result.narrow(1, 0, self.post_nms)
        ids = result.narrow(2, 0, 1)
        scores = result.narrow(2, 1, 1)
        bboxes = result.narrow(2, 2, 4)
        return ids, scores, bboxes

    def reset_class(self, classes):
        """Reset class categories and class predictors.

        Parameters
        ----------
        classes : iterable of str
            The new categories. ['apple', 'orange'] for example.

        """
        # TODO: check here
        # self.clear()
        # self.classes = classes
        # # replace class predictors
        # with self.name_scope():
        #     class_predictors = nn.HybridSequential(prefix=self.class_predictors.prefix)
        #     for i, ag in zip(range(len(self.class_predictors)), self.anchor_generators):
        #         prefix = self.class_predictors[i].prefix
        #         new_cp = ConvPredictor(ag.num_depth * (self.num_classes + 1), prefix=prefix)
        #         new_cp.collect_params().initialize()
        #         class_predictors.add(new_cp)
        #     self.class_predictors = class_predictors
        #     self.cls_decoder = MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)


def get_ssd(name, base_size, features, filters, channels, sizes, ratios, steps, classes,
            dataset, pretrained=False, pretrained_base=True,
            root=os.path.join(os.path.expanduser('~'), '.torch', 'models'), **kwargs):
    """Get SSD models.

    Parameters
    ----------
    name : str or None
        Model name, if `None` is used, you must specify `features` to be a `nn.Module`.
    base_size : int
        Base image size for training, this is fixed once training is assigned.
        A fixed base size still allows you to have variable input size during test.
    features : iterable of str or `nn.Module`
        List of network internal output names, in order to specify which layers are
        used for predicting bbox values.
        If `name` is `None`, `features` must be a `nn.Module` which generate multiple
        outputs for prediction.
    filters : iterable of float or None
        List of convolution layer channels which is going to be appended to the base
        network feature extractor. If `name` is `None`, this is ignored.
    channels : iterable of float or None
        List of convolution layer in channels
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
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    root : str
        Model weights storing path.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments

    Returns
    -------
    nn.Module
        A SSD detection network.
    """
    pretrained_base = False if pretrained else pretrained_base
    base_name = None if callable(features) else name
    net = SSD(base_name, base_size, features, filters, channels, sizes, ratios, steps,
              pretrained=pretrained_base, classes=classes, **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        full_name = '_'.join(('ssd', str(base_size), name, dataset))
        net.load_state_dict(torch.load(get_model_file(full_name, root=root)))
    return net


def ssd_300_vgg16_atrous_voc(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with VGG16 atrous 300x300 base network for Pascal VOC.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    classes = VOCDetection.CLASSES
    net = get_ssd('vgg16_atrous', 300, features=vgg16_atrous_300, filters=None, channels=None,
                  sizes=[30, 60, 111, 162, 213, 264, 315],
                  ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0 / 3]] * 3 + [[1, 2, 0.5]] * 2,
                  steps=[8, 16, 32, 64, 100, 300],
                  classes=classes, dataset='voc', pretrained=pretrained,
                  pretrained_base=pretrained_base, **kwargs)
    return net


def ssd_512_vgg16_atrous_voc(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with VGG16 atrous 512x512 base network.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    classes = VOCDetection.CLASSES
    net = get_ssd('vgg16_atrous', 512, features=vgg16_atrous_512, filters=None, channels=None,
                  sizes=[51.2, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
                  ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0 / 3]] * 4 + [[1, 2, 0.5]] * 2,
                  steps=[8, 16, 32, 64, 128, 256, 512],
                  classes=classes, dataset='voc', pretrained=pretrained,
                  pretrained_base=pretrained_base, **kwargs)
    return net


def ssd_512_resnet50_v1_voc(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with ResNet v1 50 layers.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    norm_layer : object
        Normalization layer used (default: :class:`nn.BatchNorm`)
        Can be :class:`nn.BatchNorm` or :class:`other normalization`.
    norm_kwargs : dict
        Additional `norm_layer` arguments

    Returns
    -------
    nn.Module
        A SSD detection network.
    """
    classes = VOCDetection.CLASSES
    return get_ssd('resnet50_v1', 512,
                   features=[[6, 5, 0, 5], [7, 2, 0, 5]],
                   filters=[512, 512, 512, 256, 256],
                   channels=[256],
                   sizes=[51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0 / 3]] * 3 + [[1, 2, 0.5]] * 2,
                   steps=[16, 32, 64, 128, 256, 512],
                   classes=classes, dataset='voc', pretrained=pretrained,
                   pretrained_base=pretrained_base, **kwargs)
