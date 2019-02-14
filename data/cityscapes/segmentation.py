"""Cityscapes Dataloader"""

from data.segbase import SegmentationDataset


class CitySegmentation(SegmentationDataset):
    """Cityscapes Dataloader"""
    # pylint: disable=abstract-method
    BASE_DIR = 'cityscapes'
    NUM_CLASS = 19
