from math import pi, cos
from torch.optim import Optimizer


# TODO: rewrite it
class LRScheduler(object):
    """Learning Rate Scheduler

    For mode='step', we multiply lr with `step_factor` at each epoch in `step`.

    For mode='poly'::

        lr = targetlr + (baselr - targetlr) * (1 - iter / maxiter) ^ power

    For mode='cosine'::

        lr = targetlr + (baselr - targetlr) * (1 + cos(pi * iter / maxiter)) / 2

    If warmup_epochs > 0, a warmup stage will be inserted before the main lr scheduler.

    For warmup_mode='linear'::

        lr = warmup_lr + (baselr - warmup_lr) * iter / max_warmup_iter

    For warmup_mode='constant'::

        lr = warmup_lr

    Parameters
    ----------
    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'step', 'poly' and 'cosine'.
    baselr : float
        Base learning rate, i.e. the starting learning rate.
    niters : int
        Number of iterations in each epoch.
    nepochs : int
        Number of training epochs.
    step : list
        A list of epochs to decay the learning rate.
    step_factor : float
        Learning rate decay factor.
    targetlr : float
        Target learning rate for poly and cosine, as the ending learning rate.
    power : float
        Power of poly function.
    warmup_epochs : int
        Number of epochs for the warmup stage.
    warmup_lr : float
        The base learning rate for the warmup stage.
    warmup_mode : str
        Modes for the warmup stage.
        Currently it supports 'linear' and 'constant'.
    """

    def __init__(self, optimizer, mode, n_iters, n_epochs, n_step=(30, 60, 90),
                 step_factor=0.1, targetlr=0, power=0.9, warmup_epochs=0,
                 warmup_lr=0, warmup_mode='linear', last_epoch=-1):
        super(LRScheduler, self).__init__()
        assert (mode in ['step', 'poly', 'cosine'])
        assert (warmup_mode in ['linear', 'constant'])

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))

        self.mode = mode
        self.n_iters = n_iters

        self.n_step = n_step
        self.step_factor = step_factor
        self.targetlr = targetlr
        self.power = power
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.warmup_mode = warmup_mode

        self.N = n_epochs * n_iters
        self.warmup_N = warmup_epochs * n_iters
        self.step(0, last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self, i, epoch):
        T = epoch * self.n_iters + i
        assert (0 <= T <= self.N)
        if self.warmup_epochs > epoch:
            # Warm-up Stage
            if self.warmup_mode == 'linear':
                return [self.warmup_lr + (base_lr - self.warmup_lr) * T / self.warmup_N for base_lr in self.base_lrs]
            elif self.warmup_mode == 'constant':
                return [self.warmup_lr for _ in self.base_lrs]
            else:
                raise NotImplementedError
        else:
            if self.mode == 'step':
                count = sum([1 for s in self.n_step if s <= epoch])
                return [base_lr * pow(self.step_factor, count) for base_lr in self.base_lrs]
            elif self.mode == 'poly':
                return [self.targetlr + (base_lr - self.targetlr) *
                        pow(1 - (T - self.warmup_N) / (self.N - self.warmup_N), self.power)
                        for base_lr in self.base_lrs]
            elif self.mode == 'cosine':
                return [self.targetlr + (base_lr - self.targetlr) *
                        (1 + cos(pi * (T - self.warmup_N) / (self.N - self.warmup_N))) / 2
                        for base_lr in self.base_lrs]
            else:
                raise NotImplementedError

    def step(self, i, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.get_lr(i, epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr(i, epoch)):
            param_group['lr'] = lr


def adjust_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    from torch import nn, optim

    net = nn.Conv2d(20, 30, 3, 1, 1)
    optimizer = optim.Adam(net.parameters(), lr=1e-2)
    adjust_lr(optimizer, 1e-3)
    print(optimizer.param_groups[0]['lr'])
