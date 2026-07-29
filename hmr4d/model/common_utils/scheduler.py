import torch
from bisect import bisect_right


class WarmupMultiStepLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, milestones, warmup=0, gamma=0.1, last_epoch=-1, verbose="deprecated"):
        """optimizer가 lr을 바꾸지 않고 scheduler가 epoch 단위로 호출된다고 가정한다."""
        self.milestones = milestones
        self.warmup = warmup
        assert warmup < milestones[0]
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        base_lrs = self.base_lrs  # 각 group의 기준 lr
        n_groups = len(base_lrs)
        comming_epoch = self.last_epoch  # lr을 적용할 epoch이며 0부터 시작한다.

        # warmup을 추가한다.
        if comming_epoch < self.warmup:
            # 예: warmup == 3이면 comming_epoch은 [0, 1, 2]이다.
            # lr은 base_lr * (last_epoch+1) / (warmup + 1), 즉 [0.25, 0.5, 0.75] * base_lr이다.
            lr_factor = (self.last_epoch + 1) / (self.warmup + 1)
            return [base_lrs[i] * lr_factor for i in range(n_groups)]
        else:
            # bisect_right([3,5,7], 0) -> 0; bisect_right([3,5,7], 5) -> 2
            p = bisect_right(self.milestones, comming_epoch)
            lr_factor = self.gamma**p
            return [base_lrs[i] * lr_factor for i in range(n_groups)]
