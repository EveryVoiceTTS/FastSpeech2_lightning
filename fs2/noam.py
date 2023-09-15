from torch.optim.lr_scheduler import _LRScheduler

class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """

    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def _get_lr_scale(self):
        # TODO: implement annealing
        # for s in self.anneal_steps:
        #     if self.current_step > s:
        #         lr = lr * self.anneal_rate
        # return lr
        return min((self._step_count ** -0.5), (self.warmup_steps ** -1.5) * self._step_count)

    def get_lr(self):
        # from https://github.com/dan-wells/fastpitch/blob/main/train.py
        if self.warmup_steps == 0:
            scale = 1.0
        elif self._step_count > self.warmup_steps:
            scale = 1. / (self._step_count ** 0.5)
        else:
            scale = self._step_count / (self.warmup_steps ** 1.5)
        # last_epoch = max(1, self.last_epoch)
        # scale = self.warmup_steps**0.5 * min(
        #     last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5)
        # )
        return [base_lr * scale for base_lr in self.base_lrs]
