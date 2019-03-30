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
            pred_label = pred_label.cpu().numpy().astype('int32')
            label = label.cpu().numpy().astype('int32')
            # flatten before checking shapes to avoid shape miss match
            label = label.flat
            pred_label = pred_label.flat

            check_label_shapes(label, pred_label)

            num_correct = (pred_label == label).sum()
            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += len(pred_label)
            self.global_num_inst += len(pred_label)

    def combine_metric(self, metric):
        assert isinstance(metric, Accuracy)
        self.sum_metric += metric.sum_metric
        self.num_inst += metric.num_inst


if __name__ == '__main__':
    predicts = [torch.tensor([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    labels = [torch.tensor([0, 1, 1])]
    acc = Accuracy()
    acc.update(preds=predicts, labels=labels)
    acc2 = Accuracy()
    acc2.update(preds=predicts, labels=labels)
    acc.combine_metric(acc2)
    print(acc.get())
