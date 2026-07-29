from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.checkpoint import Checkpoint
from pytorch_lightning.utilities import rank_zero_only

from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds


class SimpleCkptSaver(Checkpoint):
    """
    이 콜백은 각 학습 epoch가 끝날 때 실행된다.
    {every_n_epochs} 간격을 확인하고 최대 {save_top_k}개의 모델을 저장한다.
    """

    def __init__(
        self,
        output_dir,
        filename="e{epoch:03d}-s{step:06d}.ckpt",
        save_top_k=1,
        every_n_epochs=1,
        save_last=None,
        save_weights_only=True,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.save_top_k = save_top_k
        self.every_n_epochs = every_n_epochs
        self.save_last = save_last
        self.save_weights_only = save_weights_only

        # 출력 디렉터리를 준비한다.
        if rank_zero_only.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            Log.info(f"[Simple Ckpt Saver]: Save to `{self.output_dir}'")

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """학습 epoch가 끝날 때 체크포인트를 저장한다."""
        if self.every_n_epochs >= 1 and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            if self.save_top_k == 0:
                return

            # 출력 디렉터리에 현재 저장된 체크포인트를 찾는다.
            model_paths = []
            for p in sorted(list(self.output_dir.glob("*.ckpt"))):
                model_paths.append(p)
            model_to_remove = model_paths[0] if len(model_paths) >= self.save_top_k else None

            # 현재 체크포인트를 저장한다.
            filepath = self.output_dir / self.filename.format(epoch=trainer.current_epoch, step=trainer.global_step)
            checkpoint = {
                "epoch": trainer.current_epoch,
                "global_step": trainer.global_step,
                "pytorch-lightning_version": pl.__version__,
                "state_dict": pl_module.state_dict(),
            }
            pl_module.on_save_checkpoint(checkpoint)

            if not self.save_weights_only:
                # 옵티마이저 상태
                optimizer_states = []
                for i, optimizer in enumerate(trainer.optimizers):
                    # 가속기 구현을 통해 옵티마이저 상태를 추출한다.
                    optimizer_state = trainer.strategy.optimizer_state(optimizer)
                    optimizer_states.append(optimizer_state)
                checkpoint["optimizer_states"] = optimizer_states

                # 학습률 스케줄러 상태
                lr_schedulers = []
                for config in trainer.lr_scheduler_configs:
                    lr_schedulers.append(config.scheduler.state_dict())
                checkpoint["lr_schedulers"] = lr_schedulers

            # trainer.strategy.checkpoint_io.save_checkpoint(checkpoint, filepath)
            torch.save(checkpoint, filepath)

            # 가장 오래된 체크포인트를 제거한다.
            if model_to_remove:
                trainer.strategy.remove_checkpoint(model_paths[0])


group_name = "callbacks/simple_ckpt_saver"
base = builds(SimpleCkptSaver, output_dir="${output_dir}/checkpoints/", populate_full_signature=True)
MainStore.store(name="base", node=base, group=group_name)
MainStore.store(name="every1e", node=base, group=group_name)
MainStore.store(name="every2e", node=base(every_n_epochs=2), group=group_name)
MainStore.store(name="every5e", node=base(every_n_epochs=5), group=group_name)
MainStore.store(name="every5e_top100", node=base(every_n_epochs=5, save_top_k=100), group=group_name)
MainStore.store(name="every10e", node=base(every_n_epochs=10), group=group_name)
MainStore.store(name="every10e_top100", node=base(every_n_epochs=10, save_top_k=100), group=group_name)
