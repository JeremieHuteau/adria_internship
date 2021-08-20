import torch

class LinearLRWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        if self.last_epoch > self.num_warmup_steps:
            lrs = [group['lr'] for group in self.optimizer.param_groups]
        else:
            lrs = [
                base_lr * min(self.last_epoch / self.num_warmup_steps, 1.0)
                for base_lr in self.base_lrs
            ]
        return lrs
