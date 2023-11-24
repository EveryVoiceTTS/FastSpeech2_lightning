from torch.optim.lr_scheduler import _LRScheduler
from loguru import logger

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
        self._normalize = 1 ** (-0.5)
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            logger.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )
        # Protect against raising 0 to negative power
        last_epoch = max(1, self.last_epoch)

        if self.warmup_steps > 0:
            scale = self._normalize * min(last_epoch ** (-0.5), last_epoch * (self.warmup_steps ** (-1.5)))
        else:
            scale = self._normalize * last_epoch ** (-0.5)
        scaled_lrs = []
        for lr in self.base_lrs:
            if lr < 0.0:
                raise ValueError(
                    f"{self} received an initial learning rate that was lower than the minimum learning rate."
                )
            scaled_lr = lr * scale
            if last_epoch > self.warmup_steps:
                scaled_lr = max(scaled_lr, 0.0)
            scaled_lrs.append(scaled_lr)
        return scaled_lrs
        # scale = self.warmup_steps**0.5 * min(
        #     last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5)
        # )
        # return [base_lr * scale for base_lr in self.base_lrs]
