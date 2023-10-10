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

    def get_lr(self):
        # Protect against raising 0 to negative power
        if self.warmup_steps == 0:
            scale = 1.0
        # After warmup scale according to the step count
        elif self._step_count > self.warmup_steps:
            scale = self._step_count**-0.5
        else:
            scale = self._step_count * (self.warmup_steps**-1.5)
        return [base_lr * scale for base_lr in self.base_lrs]
