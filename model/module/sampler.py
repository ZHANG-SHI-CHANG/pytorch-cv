import torch
from torch import nn


class NaiveSampler(nn.Module):
    """A naive sampler that take all existing matching results.
    There is no ignored sample in this case.
    """

    def __init__(self):
        super(NaiveSampler, self).__init__()

    def forward(self, x):
        """Hybrid forward"""
        marker = torch.ones_like(x)
        y = torch.where(x >= 0, marker, marker * -1)
        return y

# class OHEMSampler(nn.Module):
#     """A sampler implementing Online Hard-negative mining.
#     As described in paper https://arxiv.org/abs/1604.03540.
#
#     Parameters
#     ----------
#     ratio : float
#         Ratio of negative vs. positive samples. Values >= 1.0 is recommended.
#     min_samples : int, default 0
#         Minimum samples to be selected regardless of positive samples.
#         For example, if positive samples is 0, we sometimes still want some num_negative
#         samples to be selected.
#     thresh : float, default 0.5
#         IOU overlap threshold of selected negative samples. IOU must not exceed
#         this threshold such that good matching anchors won't be selected as
#         negative samples.
#
#     """
#
#     def __init__(self, ratio, min_samples=0, thresh=0.5):
#         super(OHEMSampler, self).__init__()
#         assert ratio > 0, "OHEMSampler ratio must > 0, {} given".format(ratio)
#         self._ratio = ratio
#         self._min_samples = min_samples
#         self._thresh = thresh
#
#     # pylint: disable=arguments-differ
#     def forward(self, x, logits, ious):
#         """Forward"""
#         num_positive = torch.sum(x > -1, axis=1)
#         num_negative = self._ratio * num_positive
#         num_total = x.shape[1]  # scalar
#         num_negative = torch.clamp(num_negative, self._min_samples, num_total-num_positive)
#         positive = logits.narrow(2, 1, logits.shape[2]-1)
#         background = logits.narrow(2, 0, 1).reshape((0, -1))
#         maxval, _ = positive.max(dim=2)
#         esum = torch.exp(logits - maxval.reshape((0, 0, 1))).sum(dim=2)
#         score = -torch.log(torch.exp(background - maxval) / esum)
#         mask = torch.ones_like(score) * -1
#         score = torch.where(x < 0, score, mask)  # mask out positive samples
#         if ious.ndimension() == 3:
#             ious = torch.max(ious, dim=2)
#         score = torch.where(ious < self._thresh, score, mask)  # mask out if iou is large
#         argmaxs = torch.argsort(score, dim=1, is_ascend=False)
#
#         # neg number is different in each batch, using dynamic numpy operations.
#         y = np.zeros(x.shape)
#         y[np.where(x.asnumpy() >= 0)] = 1  # assign positive samples
#         argmaxs = argmaxs.asnumpy()
#         for i, num_neg in zip(range(x.shape[0]), num_negative.asnumpy().astype(np.int32)):
#             indices = argmaxs[i, :num_neg]
#             y[i, indices.astype(np.int32)] = -1  # assign negative samples
#         return F.array(y, ctx=x.context)
