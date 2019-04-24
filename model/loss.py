import torch
from torch import nn
import torch.nn.functional as F
from utils.bbox_pt import hard_negative_mining


# class MixSoftmaxCrossEntropyLoss(nn.Module):
#     def __init__(self, aux=True, aux_weight=0.2, ignore_label=-1, **kwargs):
#         super(MixSoftmaxCrossEntropyLoss, self).__init__(**kwargs)
#         self.aux = aux
#         self.aux_weight = aux_weight
#         self.criterion1 = nn.CrossEntropyLoss(ignore_index=ignore_label)
#         if aux:
#             self.criterion2 = nn.CrossEntropyLoss(ignore_index=ignore_label)
#
#     def _aux_forward(self, pred1, pred2, label):
#         loss1 = self.criterion1(pred1, label)
#         loss2 = self.criterion2(pred2, label)
#
#         return loss1 + self.aux_weight * loss2
#
#     def forward(self, preds, target):
#         if self.aux:
#             return dict(loss=self._aux_forward(*preds, target))
#         else:
#             return dict(loss=self.criterion1(*preds, target))

class MixSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(**kwargs)
        self.aux = aux
        self.aux_weight = aux_weight
        self.ignore_label = ignore_label

    def _aux_forward(self, pred1, pred2, label):
        loss1 = F.cross_entropy(pred1, label, ignore_index=self.ignore_label)
        loss2 = F.cross_entropy(pred2, label, ignore_index=self.ignore_label)

        return loss1 + self.aux_weight * loss2

    def _mix_forward(self, preds, label):
        for i, pred in enumerate(preds):
            if i == 0:
                loss = F.cross_entropy(pred, label, ignore_index=self.ignore_label)
            else:
                loss = loss + F.cross_entropy(pred, label, ignore_index=self.ignore_label)
        return loss

    def forward(self, preds, target):
        if self.aux:
            return dict(loss=self._aux_forward(*preds, target))
        elif len(preds) > 1:
            return dict(loss=self._mix_forward(preds, target))
        else:
            return dict(loss=F.cross_entropy(*preds, target, ignore_index=self.ignore_label))


class YOLOV3Loss(nn.Module):
    """Losses of YOLO v3.

    Parameters
    ----------
    batch_axis : int, default 0
        The axis that represents mini-batch.
    weight : float or None
        Global scalar weight for loss.

    """

    def __init__(self):
        super(YOLOV3Loss, self).__init__()
        self._sigmoid_ce = nn.BCEWithLogitsLoss(reduction='none')
        self._l1_loss = nn.L1Loss(reduction='none')

    def forward(self, objness, box_centers, box_scales, cls_preds,
                objness_t, center_t, scale_t, weight_t, class_t, class_mask):
        """Compute YOLOv3 losses.

        Parameters
        ----------
        objness : mxnet.nd.NDArray
            Predicted objectness (B, N), range (0, 1).
        box_centers : mxnet.nd.NDArray
            Predicted box centers (x, y) (B, N, 2), range (0, 1).
        box_scales : mxnet.nd.NDArray
            Predicted box scales (width, height) (B, N, 2).
        cls_preds : mxnet.nd.NDArray
            Predicted class predictions (B, N, num_class), range (0, 1).
        objness_t : mxnet.nd.NDArray
            Objectness target, (B, N), 0 for negative 1 for positive, -1 for ignore.
        center_t : mxnet.nd.NDArray
            Center (x, y) targets (B, N, 2).
        scale_t : mxnet.nd.NDArray
            Scale (width, height) targets (B, N, 2).
        weight_t : mxnet.nd.NDArray
            Loss Multipliers for center and scale targets (B, N, 2).
        class_t : mxnet.nd.NDArray
            Class targets (B, N, num_class).
            It's relaxed one-hot vector, i.e., (1, 0, 1, 0, 0).
            It can contain more than one positive class.
        class_mask : mxnet.nd.NDArray
            0 or 1 mask array to mask out ignored samples (B, N, num_class).

        Returns
        -------
        tuple of NDArrays
            obj_loss: sum of objectness logistic loss
            center_loss: sum of box center logistic regression loss
            scale_loss: sum of box scale l1 loss
            cls_loss: sum of per class logistic loss

        """
        # compute some normalization count, except batch-size
        denorm = objness_t.numel() / objness_t.shape[0]
        weight_t = weight_t * objness_t
        hard_objness_t = torch.where(objness_t > 0, torch.ones_like(objness_t), objness_t)
        new_objness_mask = torch.where(objness_t > 0, objness_t, (objness_t >= 0).type(objness_t.dtype))
        obj_loss = (self._sigmoid_ce(objness, hard_objness_t) * new_objness_mask * denorm).mean()
        center_loss = (self._sigmoid_ce(box_centers, center_t) * weight_t * denorm * 2).mean()
        scale_loss = (self._l1_loss(box_scales, scale_t) * weight_t * denorm * 2).mean()
        denorm_class = class_t.numel() / class_t.shape[0]
        class_mask = class_mask * objness_t
        cls_loss = (self._sigmoid_ce(cls_preds, class_t) * class_mask * denorm_class).mean()
        return obj_loss, center_loss, scale_loss, cls_loss


class SSDMultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.
        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(SSDMultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, cls_pred, box_pred, cls_target, box_target):
        """Compute classification loss and smooth l1 loss.
        Args:
            cls_pred (batch_size, num_priors, num_classes): class predictions.
            box_pred (batch_size, num_priors, 4): predicted locations.
            cls_target (batch_size, num_priors): real cls_target of all the priors.
            box_target (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = cls_pred.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(cls_pred, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, cls_target, self.neg_pos_ratio)

        cls_pred = cls_pred[mask, :]
        classification_loss = F.cross_entropy(cls_pred.view(-1, num_classes), cls_target[mask].long(), reduction='sum')

        pos_mask = cls_target > 0
        box_pred = box_pred[pos_mask, :].view(-1, 4)
        box_target = box_target[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(box_pred, box_target, reduction='sum')
        num_pos = box_target.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


if __name__ == '__main__':
    import torch

    import numpy as np

    block = SSDMultiBoxLoss(3.0)
    import numpy as np

    np.random.seed(100)
    cls_pred = np.random.randn(8, 8732, 21)
    box_pred = np.random.randn(8, 8732, 4)
    cls_target = np.random.randint(0, 20, (8, 8732))
    box_target = np.random.randn(8, 8732, 4)

    cls_pred = torch.from_numpy(cls_pred)
    box_pred = torch.from_numpy(box_pred)
    cls_target = torch.from_numpy(cls_target)
    box_target = torch.from_numpy(box_target)

    out = block(cls_pred, box_pred, cls_target, box_target)
    print(out)

    # np.random.seed(10)
    # objness = np.random.random(size=(1, 10, 1))
    # box_centers = np.random.random(size=(1, 10, 2))
    # box_scales = np.random.random(size=(1, 10, 2))
    # cls_preds = np.random.random(size=(1, 10, 20))
    # objness_t = np.random.random(size=(1, 10, 1))
    # center_t = np.random.random(size=(1, 10, 2))
    # scale_t = np.random.random(size=(1, 10, 2))
    # weight_t = np.random.random(size=(1, 10, 2))
    # class_t = np.random.random(size=(1, 10, 20))
    # class_mask = np.random.random(size=(1, 10, 20))
    #
    # objness = torch.from_numpy(objness)
    # box_centers = torch.from_numpy(box_centers)
    # box_scales = torch.from_numpy(box_scales)
    # cls_preds = torch.from_numpy(cls_preds)
    # objness_t = torch.from_numpy(objness_t)
    # center_t = torch.from_numpy(center_t)
    # scale_t = torch.from_numpy(scale_t)
    # weight_t = torch.from_numpy(weight_t)
    # class_t = torch.from_numpy(class_t)
    # class_mask = torch.from_numpy(class_mask)
    # loss = YOLOV3Loss()
    # print(loss(objness, box_centers, box_scales, cls_preds,
    #            objness_t, center_t, scale_t, weight_t, class_t, class_mask))
    #
    # # a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
    # # b = torch.Tensor([0, 1]).long()
    # # loss = MixSoftmaxCrossEntropyLoss(aux=False)
    # # print(loss([a], b))
