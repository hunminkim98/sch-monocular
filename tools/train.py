import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks.checkpoint import Checkpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger

from hmr4d.utils.pylogger import Log
from hmr4d.configs import register_store_footmr
from hmr4d.utils.vis.rich_logger import print_cfg
from hmr4d.utils.net_utils import load_pretrained_model, get_resume_ckpt_path


def get_callbacks(cfg: DictConfig) -> list:
    """설정에 정의된 callback을 해석하고 모두 생성합니다."""
    if not hasattr(cfg, "callbacks") or cfg.callbacks is None:
        return None
    # 특수 callback을 처리합니다.
    enable_checkpointing = cfg.pl_trainer.get("enable_checkpointing", True)
    # 모든 callback을 생성합니다.
    callbacks = []
    for callback in cfg.callbacks.values():
        if callback is not None:
            cb = hydra.utils.instantiate(callback, _recursive_=False)
            # checkpointing이 비활성화된 경우 Checkpoint callback을 건너뜁니다.
            if not enable_checkpointing and isinstance(cb, Checkpoint):
                continue
            else:
                callbacks.append(cb)
    return callbacks


def train(cfg: DictConfig) -> None:
    """학습 또는 평가를 실행합니다."""
    Log.info(f"[Exp Name]: {cfg.exp_name}")
    if cfg.task == "fit":
        Log.info(f"[GPU x Batch] = {cfg.pl_trainer.devices} x {cfg.data.loader_opts.train.batch_size}")
    # pl.seed_everything(cfg.seed)

    # 실행 준비
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data, _recursive_=False)
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)
    if cfg.ckpt_path is not None:
        load_pretrained_model(model, cfg.ckpt_path)

    # PyTorch Lightning callback과 logger를 준비합니다.
    callbacks = get_callbacks(cfg)
    has_ckpt_cb = any([isinstance(cb, Checkpoint) for cb in callbacks])
    if not has_ckpt_cb and cfg.pl_trainer.get("enable_checkpointing", True):
        Log.warning("No checkpoint-callback found. Disabling PL auto checkpointing.")
        cfg.pl_trainer = {**cfg.pl_trainer, "enable_checkpointing": False}
    exp_loggers = []
    logger = hydra.utils.instantiate(cfg.logger, _recursive_=False)
    exp_loggers.append(logger)

    # W&B logging을 설정합니다.
    USE_WANDB = True
    if USE_WANDB and "DEBUG" not in cfg.exp_name:
        # W&B run 이름을 줄입니다.
        run_name = cfg.exp_name[2:].split("_")
        run_name = "_".join(run_name[:1] + run_name[2:])  # 시-분-초를 제거합니다.
        wandb_logger = WandbLogger(
            project="footmr",
            name=run_name,
            save_dir=cfg.output_dir,
        )
        exp_loggers.append(wandb_logger)

    # PyTorch Lightning Trainer를 생성합니다.
    if cfg.task == "test":
        Log.info("Test mode forces full-precision.")
        cfg.pl_trainer = {**cfg.pl_trainer, "precision": 32}
    callbacks.append(TQDMProgressBar(refresh_rate=100))
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=exp_loggers if exp_loggers is not None else False,
        callbacks=callbacks,
        # limit_train_batches=2,
        **cfg.pl_trainer,
    )

    if cfg.task == "fit":
        resume_path = None
        if cfg.resume_mode is not None:
            resume_path = get_resume_ckpt_path(cfg.resume_mode, ckpt_dir=cfg.callbacks.model_checkpoint.dirpath)
            Log.info(f"Resume training from {resume_path}")
        Log.info("Start Fitting...")
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader(), ckpt_path=resume_path)
    elif cfg.task == "test":
        Log.info("Start Testing...")
        trainer.test(model, datamodule.test_dataloader())
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

    Log.info("End of script.")
    raise SystemExit("*** Run SystemExit ***")


@hydra.main(version_base="1.3", config_path="../hmr4d/configs", config_name="train")
def main(cfg) -> None:
    print_cfg(cfg, use_rich=True)
    train(cfg)


if __name__ == "__main__":
    register_store_footmr()
    main()
