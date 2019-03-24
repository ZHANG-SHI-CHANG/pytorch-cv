from torch import nn


class MixSoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(**kwargs)
        self.aux = aux
        self.aux_weight = aux_weight
        self.criterion1 = nn.CrossEntropyLoss(ignore_index=ignore_label)
        if aux:
            self.criterion2 = nn.CrossEntropyLoss(ignore_index=ignore_label)

    def _aux_forward(self, pred1, pred2, label):
        loss1 = self.criterion1(pred1, label)
        loss2 = self.criterion2(pred2, label)

        return loss1 + self.aux_weight * loss2

    def forward(self, preds, target):
        if self.aux:
            return self._aux_forward(*preds, target)
        else:
            return self.criterion1(*preds, target)


if __name__ == '__main__':
    import torch
    a = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.Tensor([0, 1]).long()
    loss = MixSoftmaxCrossEntropyLoss(aux=False)
    print(loss([a], b))