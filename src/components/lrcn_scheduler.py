import torch.optim as optim

class LRCNCustomScheduler(optim.lr_scheduler._LRScheduler):
    """
    Learning Rate Scheduler described in LRCN paper.
    This scheduler scales the learning rate linearly for warmup steps and decays it at specific epoch milestones.
    The learning rate is also choosen from minimum of (2.5te-5, 1e-4) where t is the current epoch.
    """
    def __init__(
            self,
            optimizer: optim.Optimizer,
            base_lr: float = 1e-4,
            compare_lr: float = 2.5e-5,
            decay_factor: int = 0.2,
            milestones: list[int] = [10, 12],
            last_epoch: int = -1,
    ):
        self.decay_factor = decay_factor
        self.compare_lr = compare_lr
        self.base_lr = base_lr
        self.milestones = set(milestones)
        super(LRCNCustomScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        t = self.last_epoch + 1
        current_step_lr = min(self.compare_lr * t, self.base_lr)

        if t in self.milestones:
            current_step_lr *= self.decay_factor
        
        return [lr for _ in self.optimizer.param_groups]