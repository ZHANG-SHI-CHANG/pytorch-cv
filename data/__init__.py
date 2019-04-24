from .ade2k.segmentation import ADE20KSegmentation
from .pascal_voc.segmentation import VOCSegmentation
from .pascal_voc.segmentation_paper import VOCSegmentationPaper
from .pascal_aug.segmentation import VOCAugSegmentation
from .mscoco.segmentation import COCOSegmentation
from .cityscapes.segmentation import CitySegmentation

datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_paper': VOCSegmentationPaper,
    'pascal_aug': VOCAugSegmentation,
    'coco': COCOSegmentation,
    'citys': CitySegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
