import torch

from utils.metrics.metric import EvalMetric, check_label_shapes


class Accuracy(EvalMetric):
    """Computes accuracy classification score.

    The accuracy score is defined as

    .. math::
        \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i=0}^{n-1}
        \\text{1}(\\hat{y_i} == y_i)

    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> predicts = [torch.tensor([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [torch.tensor([0, 1, 1])]
    >>> acc = Accuracy()
    >>> acc.update(preds = predicts, labels = labels)
    >>> print acc.get()
    ('accuracy', 0.6666666666666666)
    """

    def __init__(self, axis=1, name='accuracy',
                 output_names=None, label_names=None):
        super(Accuracy, self).__init__(
            name, axis=axis, output_names=output_names,
            label_names=label_names, has_global_stats=True)
        self.axis = axis
        self.reset()

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data with class indices as values, one per sample.

        preds : list of `NDArray`
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = torch.argmax(pred_label, dim=self.axis)
            pred_label = pred_label.long()
            label = label.long()
            # flatten before checking shapes to avoid shape miss match
            label = label.flatten()
            pred_label = pred_label.flatten()

            check_label_shapes(label, pred_label)

            num_correct = torch.sum(pred_label == label).item()
            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += pred_label.numel()
            self.global_num_inst += pred_label.numel()

    def get_value(self):
        return {'sum_metric': self.sum_metric, 'num_inst': self.num_inst}

    def combine_value(self, values):
        self.sum_metric += values['sum_metric']
        self.num_inst += values['num_inst']


class TopKAccuracy(EvalMetric):
    """Computes top k predictions accuracy.

    `TopKAccuracy` differs from Accuracy in that it considers the prediction
    to be ``True`` as long as the ground truth label is in the top K
    predicated labels.

    If `top_k` = ``1``, then `TopKAccuracy` is identical to `Accuracy`.

    Parameters
    ----------
    top_k : int
        Whether targets are in top k predictions.
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> np.random.seed(999)
    >>> top_k = 3
    >>> labels = [mx.nd.array([2, 6, 9, 2, 3, 4, 7, 8, 9, 6])]
    >>> predicts = [mx.nd.array(np.random.rand(10, 10))]
    >>> acc = mx.metric.TopKAccuracy(top_k=top_k)
    >>> acc.update(labels, predicts)
    >>> print acc.get()
    ('top_k_accuracy', 0.3)
    """

    def __init__(self, top_k=1, name='top_k_accuracy',
                 output_names=None, label_names=None):
        super(TopKAccuracy, self).__init__(
            name, top_k=top_k, output_names=output_names,
            label_names=label_names, has_global_stats=True)
        self.top_k = top_k
        assert (self.top_k > 1), 'Please use Accuracy if top_k is no more than 1'
        self.name += '_%d' % self.top_k
        self.reset()

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            assert (len(pred_label.shape) <= 2), 'Predictions should be no more than 2 dims'
            # Using argpartition here instead of argsort is safe because
            # we do not care about the order of top k elements. It is
            # much faster, which is important since that computation is
            # single-threaded due to Python GIL.
            pred_label = torch.argsort(pred_label, dim=-1, descending=True)
            check_label_shapes(label, pred_label)
            num_samples = pred_label.shape[0]
            num_dims = len(pred_label.shape)
            if num_dims == 1:
                self.sum_metric += (pred_label == label).sum().item()
            elif num_dims == 2:
                num_classes = pred_label.shape[1]
                top_k = min(num_classes, self.top_k)
                for j in range(top_k):
                    # num_correct = (pred_label[:, num_classes - 1 - j] == label).sum().item()
                    num_correct = (pred_label[:, j] == label).sum().item()
                    self.sum_metric += num_correct
                    self.global_sum_metric += num_correct
            self.num_inst += num_samples
            self.global_num_inst += num_samples

    def get_value(self):
        return {'sum_metric': self.sum_metric, 'num_inst': self.num_inst}

    def combine_value(self, values):
        self.sum_metric += values['sum_metric']
        self.num_inst += values['num_inst']


if __name__ == '__main__':
    # predicts = [torch.tensor([[0.3, 0.7], [0, 1.], [0.4, 0.6]]).cuda()]
    # labels = [torch.tensor([0, 1, 1]).cuda()]
    # acc = Accuracy()
    # acc.update(preds=predicts, labels=labels)
    # # acc2 = Accuracy()
    # # acc2.update(preds=predicts, labels=labels)
    # # acc.combine_metric(acc2)
    # print(acc.get())
    import numpy as np

    np.random.seed(999)
    top_k = 3
    labels = [torch.tensor([2, 6, 9, 2, 3, 4, 7, 8, 9, 6])]
    predicts = [torch.from_numpy(np.random.rand(10, 10))]
    acc = TopKAccuracy(top_k=top_k)
    acc.update(labels, predicts)
    print(acc.get())
