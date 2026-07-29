from collections import OrderedDict
from numbers import Number
from datetime import datetime, timedelta
from typing import Any, Dict, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar, Tqdm, convert_inf
from pytorch_lightning.callbacks.progress import ProgressBar
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl

from hmr4d.utils.pylogger import Log
from time import time
from collections import deque
import sys
from hmr4d.configs import MainStore, builds

# ========== 보조 함수 ========== #


def format_num(n):
    f = "{0:.3g}".format(n).replace("+0", "+").replace("-0", "-")
    n = str(n)
    return f if len(f) < len(n) else n


def convert_kwargs_to_str(**kwargs):
    # 항상 같은 결과가 나오도록 알파벳순으로 정렬한다.
    postfix = OrderedDict([])
    for key in sorted(kwargs.keys()):
        new_key = key.split("/")[-1]
        postfix[new_key] = kwargs[key]
    # 자료형에 따라 통계 값을 전처리한다.
    for key in postfix.keys():
        # 숫자: 문자열 길이를 제한한다.
        if isinstance(postfix[key], Number):
            postfix[key] = format_num(postfix[key])
        # 그 외 자료형은 문자열로 변환한다.
        elif not isinstance(postfix[key], str):
            postfix[key] = str(postfix[key])
        # 문자열이면 별도 전처리가 필요 없다.
    # 최종 postfix 문자열을 조합한다.
    postfix = ", ".join(key + "=" + postfix[key].strip() for key in postfix.keys())
    return postfix


def convert_t_to_str(t):
    """초 단위 시간을 시:분:초 형식의 문자열로 변환한다.
    시간이 0이면 표시하지 않고, 분과 초는 항상 표시한다.
    """
    t_str = timedelta(seconds=t)  # e.g. 0:00:00.704186
    t_str = str(t_str).split(".")[0]  # e.g. 0:00:00
    if t_str[:2] == "0:":
        t_str = t_str[2:]
    return t_str


class MyTQDMProgressBar(TQDMProgressBar, pl.Callback):
    def init_train_tqdm(self):
        bar = Tqdm(
            desc="Training",  # 이후에 덮어쓰이는 초기값이다.
            bar_format="{desc}{percentage:3.0f}%[{bar:10}][{n_fmt}/{total_fmt}, {elapsed}→{remaining},{rate_fmt}]{postfix}",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            smoothing=0,
            dynamic_ncols=False,
        )
        return bar

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 상위 함수에서 기본 진행 표시줄도 갱신한다.
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        # 여기서는 기본 진행 표시줄의 postfix만 설정한다.
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            # postfix 문자열을 설정한다.
            # 1. 최대 GPU 사용량
            max_mem = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
            post_fix_str = f"maxGPU={max_mem:.1f}GB"

            # 2. 학습 지표
            training_metrics = self.get_metrics(trainer, pl_module)
            training_metrics.pop("v_num", None)
            post_fix_str += ", " + convert_kwargs_to_str(**training_metrics)

            # 추가 메시지가 있으면 함께 표시한다.
            if "message" in outputs:
                post_fix_str += ", " + outputs["message"]

            self.train_progress_bar.set_postfix_str(post_fix_str)


class ProgressReporter(ProgressBar, pl.Callback):
    def __init__(
        self,
        log_every_percent: float = 0.1,  # 보고 간격
        exp_name=None,  # None이면 pl_module.exp_name 또는 "Unnamed Experiment"를 사용한다.
        data_name=None,  # None이면 pl_module.data_name 또는 "Unknown Data"를 사용한다.
        **kwargs,
    ):
        super().__init__()
        self.enable = True
        # 1. 실험 메타데이터를 저장한다.
        self.log_every_percent = log_every_percent
        self.exp_name = exp_name
        self.data_name = data_name
        self.batch_time_queue = deque(maxlen=5)
        self.start_prompt = "🚀"
        self.finish_prompt = "✅"
        # 2. 평가용 상태
        self.n_finished = 0

    def disable(self):
        self.enable = False

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        # trainer 객체와 연결한다.
        super().setup(trainer, pl_module, stage)
        self.stage = stage
        self.time_exp_start = time()
        self.epoch_exp_start = trainer.current_epoch

        if self.exp_name is None:
            if hasattr(pl_module, "exp_name"):
                self.exp_name = pl_module.exp_name
            else:
                self.exp_name = "Unnamed Experiment"
        if self.data_name is None:
            if hasattr(pl_module, "data_name"):
                self.data_name = pl_module.data_name
            else:
                self.data_name = "Unknown Data"

    def print(self, *args: Any, **kwargs: Any) -> None:
        print(*args)

    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Dict[str, Union[str, float]]:
        """진행 표시에 사용할 지표를 trainer에서 가져온다."""
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items

    def _should_update(self, n_finished: int, total: int) -> bool:
        """
        `log_every_percent` 비율마다 또는 마지막 배치에서 로그를 기록한다.
        """
        log_interval = max(int(total * self.log_every_percent), 1)
        able = n_finished % log_interval == 0 or n_finished == total
        if log_interval > 10:
            able = able or n_finished in [5, 10]  # 초기 진행 상황은 항상 기록한다.
        able = able and self.enable
        return able

    @rank_zero_only
    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        self.print("=" * 80)
        Log.info(
            f"{self.start_prompt}[FIT][Epoch {trainer.current_epoch}] Data: {self.data_name} Experiment: {self.exp_name}"
        )
        self.time_train_epoch_start = time()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # 기본 상태 갱신을 유지한다.
        total = self.total_train_batches

        # 속도
        n_finished = batch_idx + 1
        percent = 100 * n_finished / total
        time_current = time()
        self.batch_time_queue.append(time_current)
        time_elapsed = time_current - self.time_train_epoch_start  # 초
        time_remaining = time_elapsed * (total - n_finished) / n_finished  # 초
        if len(self.batch_time_queue) == 1:  # 아직 속도를 계산할 수 없다.
            speed = 1 / time_elapsed
        else:
            speed = (len(self.batch_time_queue) - 1) / (self.batch_time_queue[-1] - self.batch_time_queue[0])

        # 갱신 시점이 아니면 건너뛴다.
        if not self._should_update(n_finished, total):
            return

        # ===== prefix 문자열 설정 ===== #
        # 기본 설명
        desc = f"[Train]"

        # 속도: 경과 시간과 예상 잔여 시간을 계산한다.
        time_elapsed_str = convert_t_to_str(time_elapsed)
        time_remaining_str = convert_t_to_str(time_remaining)
        speed_str = f"{speed:.2f}it/s" if speed > 1 else f"{1/speed:.1f}s/it"
        n_digit = len(str(total))
        desc_speed = (
            f"[{n_finished:{n_digit}d}/{total}={percent:3.0f}%, {time_elapsed_str} → {time_remaining_str}, {speed_str}]"
        )

        # ===== postfix 문자열 설정 ===== #
        # 1. 최대 GPU 사용량
        max_mem = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
        post_fix_str = f"maxGPU={max_mem:.1f}GB"

        # 2. 학습 스텝 지표
        train_metrics = self.get_metrics(trainer, pl_module)
        train_metrics = {k: v for k, v in train_metrics.items() if ("train" in k and "epoch" not in k)}
        post_fix_str += ", " + convert_kwargs_to_str(**train_metrics)

        # 추가 메시지가 있으면 함께 표시한다.
        if "message" in outputs:
            post_fix_str += ", " + outputs["message"]
        post_fix_str = f"[{post_fix_str}]"

        # ===== 출력 ===== #
        bar_output = f"{desc}{desc_speed}{post_fix_str}"
        self.print(bar_output)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)

        # 상태를 비운다.
        self.batch_time_queue.clear()

        # 전체 epoch 소요 시간을 추정한다.
        n_finished = trainer.current_epoch + 1 - self.epoch_exp_start
        n_to_finish = trainer.max_epochs - trainer.current_epoch - 1
        time_current = time()
        time_elapsed = time_current - self.time_exp_start
        time_remaining = time_elapsed * n_to_finish / n_finished
        time_elapsed_str = convert_t_to_str(time_elapsed)
        time_remaining_str = convert_t_to_str(time_remaining)

        # 지표
        # 학습 epoch 지표
        train_metrics = self.get_metrics(trainer, pl_module)
        train_metrics = {k: v for k, v in train_metrics.items() if ("train" in k and "epoch" in k)}
        train_metrics_str = convert_kwargs_to_str(**train_metrics)

        Log.info(
            f"{self.finish_prompt}[FIT][Epoch {trainer.current_epoch}] finished! {time_elapsed_str}→{time_remaining_str} | {train_metrics_str}"
        )

    # ===== 검증/테스트/예측 ===== #
    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        self.time_val_epoch_start = time()

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.n_finished += 1
        n_finished = self.n_finished
        total = self.total_val_batches
        if not self._should_update(n_finished, total):
            return

        # 기본 설명
        desc = f"[Val]"

        # 속도
        percent = 100 * n_finished / total
        time_current = time()
        time_elapsed = time_current - self.time_val_epoch_start  # 초
        time_remaining = time_elapsed * (total - n_finished) / n_finished  # 초
        time_elapsed_str = convert_t_to_str(time_elapsed)
        time_remaining_str = convert_t_to_str(time_remaining)
        desc_speed = f"[{n_finished}/{total} ={percent:3.0f}%, {time_elapsed_str}→{time_remaining_str}]"

        # 출력
        bar_output = f"{desc} {desc_speed}"
        self.print(bar_output)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # 상태를 초기화한다.
        self.n_finished = 0


class EmojiProgressReporter(ProgressBar, pl.Callback):
    def __init__(
        self,
        refresh_rate_batch: Union[int, None] = 1,  # 배치 보고 간격이며, None이면 비활성화한다.
        refresh_rate_epoch: int = 1,  # epoch 보고 간격
        **kwargs,
    ):
        super().__init__()
        self.enable = True
        # 실험 메타데이터를 저장한다.
        self.refresh_rate_batch = refresh_rate_batch
        self.refresh_rate_epoch = refresh_rate_epoch

        # 진행 표시줄 스타일
        self.title_prompt = "📝"
        self.prog_prompt = "🚀"
        self.timer_prompt = "⌛️"
        self.metric_prompt = "📌"
        self.finish_prompt = "✅"

    def disable(self):
        self.enable = False

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        # trainer 객체와 연결한다.
        super().setup(trainer, pl_module, stage)
        self.stage = stage
        self.time_start_batch = None
        self.time_start_epoch = None
        if hasattr(pl_module, "exp_name"):
            self.exp_name = pl_module.exp_name
        else:
            self.exp_name = "Unnamed Experiment"
            Log.warn("Experiment name not found, please set it to `pl_module.exp_name`!")

    def print(self, *args: Any, **kwargs: Any):
        print(*args)

    def get_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> Dict[str, Union[str, float]]:
        """진행 표시에 사용할 지표를 trainer에서 가져온다."""
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return dict(sorted(items.items()))

    def _should_log_batch(self, n: int) -> bool:
        # 배치 로그를 비활성화한다.
        if self.refresh_rate_batch is None:
            return False
        # 첫 배치, 마지막 배치, `self.refresh_rate_batch` 간격마다 기록한다.
        able = n % self.refresh_rate_batch == 0 or n == self.total_train_batches - 1
        able = able and self.enable
        return able

    def _should_log_epoch(self, n: int) -> bool:
        # 첫 epoch, 마지막 epoch, `self.refresh_rate_epoch` 간격마다 기록한다.
        able = n % self.refresh_rate_epoch == 0 or n == self.trainer.max_epochs - 1
        able = able and self.enable
        return able

    def timestamp_delta_to_str(self, timestamp_delta: float):
        """시간 차이를 읽기 쉬운 문자열로 변환한다."""
        time_rest = timedelta(seconds=timestamp_delta)
        hours, remainder = divmod(time_rest.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = ""

        # 표시할 시간이 유효한지 확인한다. 시간이 표시되면 분도 함께 표시해야 한다.
        if hours <= 0:
            hours = None
            if minutes <= 0:
                minutes = None
                if seconds <= 0:
                    seconds = None

        time_str += f"{hours}h " if hours is not None else ""
        time_str += f"{minutes}m " if minutes is not None else ""
        time_str += f"{seconds}s" if seconds is not None else ""
        return time_str

    @rank_zero_only
    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
        # 메타데이터를 초기화한다.
        if self.time_start_batch is None:
            self.time_start_batch = datetime.now().timestamp()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # 기본 상태 갱신을 유지한다.
        # 메타데이터를 가져온다.
        epoch_idx = trainer.current_epoch
        percent = 100 * (batch_idx + 1) / (self.total_train_batches + 1)
        metrics = self.get_metrics(trainer, pl_module)

        # 현재 시간
        time_cur_stamp = datetime.now().timestamp()
        time_cur_str = datetime.fromtimestamp(time_cur_stamp).strftime("%m-%d %H:%M:%S")
        # 잔여 시간
        time_rest_stamp = (time_cur_stamp - self.time_start_batch) * (100 - percent) / percent
        time_rest_str = self.timestamp_delta_to_str(time_rest_stamp)

        if not self._should_log_batch(batch_idx):
            return

        # 로그를 출력한다.
        self.print(f"{self.title_prompt} [{self.stage.upper()}] Exp: {self.exp_name}...")
        self.print(
            f"{self.prog_prompt} Ep {epoch_idx}: {int(percent):02d}% <= [{batch_idx}/{self.total_train_batches}]"
        )
        self.print(f"{self.timer_prompt} Time: {time_cur_str} | Ep Rest: {time_rest_str}")
        for k, v in metrics.items():
            self.print(f"{self.metric_prompt} {k}: {v}")
        self.print("")  # 빈 줄을 추가한다.

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        super().on_train_epoch_start(trainer, pl_module)
        # 메타데이터를 초기화한다.
        self.time_start_batch = None
        if self.time_start_epoch is None:
            self.time_start_epoch = datetime.now().timestamp()

    @rank_zero_only
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        super().on_train_epoch_end(trainer, pl_module)
        # 메타데이터를 가져온다.
        epoch_idx = trainer.current_epoch
        percent = 100 * (epoch_idx + 1) / (self.trainer.max_epochs + 1)
        metrics = self.get_metrics(trainer, pl_module)

        # 현재 시간
        time_cur = datetime.now().timestamp()
        time_str = datetime.fromtimestamp(time_cur).strftime("%m-%d %H: %M:%S")
        # 잔여 시간
        time_rest_stamp = (time_cur - self.time_start_epoch) * (100 - percent) / percent
        time_rest_str = self.timestamp_delta_to_str(time_rest_stamp)

        if not self._should_log_batch(epoch_idx):
            return

        # 로그를 출력한다.
        self.print(f">> >> >> >>")
        self.print(f"{self.title_prompt} [{self.stage.upper()}] Exp: {self.exp_name}")
        self.print(f"{self.finish_prompt} Ep {epoch_idx} finished!")
        self.print(f"{self.timer_prompt} Time: {time_str} | Rest: {time_rest_str}")
        for k, v in metrics.items():
            self.print(f"{self.metric_prompt} {k}: {v}")
        self.print(f"<< << << <<")
        self.print("")  # 빈 줄을 추가한다.


group_name = "callbacks/prog_bar"
prog_reporter_base = builds(
    ProgressReporter,
    log_every_percent=0.1,
    exp_name="${exp_name}",
    data_name="${data_name}",
    populate_full_signature=True,
)
MainStore.store(name="prog_reporter_every0.1", node=prog_reporter_base, group=group_name)
MainStore.store(name="prog_reporter_every0.2", node=prog_reporter_base(log_every_percent=0.2), group=group_name)
