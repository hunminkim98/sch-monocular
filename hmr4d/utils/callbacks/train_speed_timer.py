import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from time import time
from collections import deque

from hmr4d.configs import MainStore, builds


class TrainSpeedTimer(pl.Callback):
    def __init__(self, N_avg=5):
        """
        이 콜백은 최근 N회 반복의 평균 학습 속도를 측정한다.
            1. 데이터 대기 시간: 값이 크면 데이터 로딩을 개선해야 한다.
            2. 단일 배치 시간: 데이터 대기를 제외한 한 배치의 학습 시간이다.
        """
        super().__init__()
        self.last_batch_end = None
        self.this_batch_start = None

        # 평균 계산용 시간 큐
        self.data_waiting_time_queue = deque(maxlen=N_avg)
        self.single_batch_time_queue = deque(maxlen=N_avg)

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """데이터 대기 시간을 측정한다."""
        if self.last_batch_end is not None:
            # 값이 크면 데이터 로딩을 개선해야 한다.
            data_waiting = time() - self.last_batch_end

            # 평균 시간을 계산한다.
            self.data_waiting_time_queue.append(data_waiting)
            average_time = sum(self.data_waiting_time_queue) / len(self.data_waiting_time_queue)

            # 진행 표시줄에 기록한다.
            pl_module.log(
                "train_timer/data_waiting", average_time, on_step=True, on_epoch=True,
                prog_bar=True, logger=True, batch_size=batch["B"]
            )

        self.this_batch_start = time()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 데이터 대기를 제외한 실제 학습 시간을 계산한다.
        single_batch = time() - self.this_batch_start

        # 평균 시간을 계산한다.
        self.single_batch_time_queue.append(single_batch)
        average_time = sum(self.single_batch_time_queue) / len(self.single_batch_time_queue)

        # 반복 시간을 기록한다.
        pl_module.log(
            "train_timer/single_batch", average_time, on_step=True, on_epoch=True,
            prog_bar=False, logger=True, batch_size=batch["B"]
        )

        # 다음 데이터 대기 시간을 측정하도록 타이머를 설정한다.
        self.last_batch_end = time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        # 타이머를 초기화한다.
        self.last_batch_end = None
        self.this_batch_start = None
        # 큐를 비운다.
        self.data_waiting_time_queue.clear()
        self.single_batch_time_queue.clear()


group_name = "callbacks/train_speed_timer"
base = builds(TrainSpeedTimer, populate_full_signature=True)
MainStore.store(name="base", node=base, group=group_name)
