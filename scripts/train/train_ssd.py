from data.pascal_voc.detection import VOCDetection
from data.mscoco.detection import COCODetection
from utils.metrics import VOC07MApMetric, COCODetectionMetric


def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = COCODetection(splits='instances_train2017')
        val_dataset = COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(args.data_shape, args.data_shape))
        # coco validation is slow, consider increase the validation interval
        if args.val_interval == 1:
            args.val_interval = 10
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric

