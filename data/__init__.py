from data.ade2k.segmentation import ADE20KSegmentation
from data.pascal_voc.segmentation import VOCSegmentation
from data.pascal_aug.segmentation import VOCAugSegmentation
from data.mscoco.segmentation import COCOSegmentation
from data.cityscapes.segmentation import CitySegmentation

datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'coco': COCOSegmentation,
    'citys': CitySegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
