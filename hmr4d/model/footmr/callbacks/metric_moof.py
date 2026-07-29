import torch
import pytorch_lightning as pl
from pathlib import Path
from hmr4d.configs import MainStore, builds
from hmr4d.utils.pylogger import Log
from hmr4d.utils.comm.gather import all_gather
from hmr4d.utils.eval.eval_utils import n_mpjpe
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo.hmr_cam import perspective_projection


def pck_accuracy(pred, gt, mask, thr):
    """
    각 keypoint의 PCK pose 정확도와 전체 keypoint의 평균 정확도를 계산합니다.

    인자:
        pred (torch.Tensor[N, K, 2]): 정규화된 예측 keypoint 위치
        gt (torch.Tensor[N, K, 2]): 정규화된 GT keypoint 위치
        mask (torch.Tensor[N, K]): target의 visibility. 보이지 않으면 False,
            보이면 True이며, 보이지 않는 joint는 정확도 계산에서 제외합니다.
        thr (float): PCK 계산 threshold

    반환:
        tuple: keypoint 정확도를 담은 tuple

        - acc (torch.Tensor[N, K]): instance별 각 keypoint의 정확도
        - avg_acc (float): 전체 keypoint의 평균 정확도
        - cnt (int): 유효한 keypoint 수
    """
    # 예측 keypoint와 GT keypoint 사이의 Euclidean distance를 계산합니다.
    distances = torch.norm(pred - gt, dim=-1)
    # 보이는 keypoint의 거리만 고려합니다.
    visible_distances = distances * mask.float()
    # 거리가 threshold 이내인지 판정합니다.
    within_threshold = visible_distances < thr

    # instance별 각 keypoint의 정확도를 계산합니다.
    acc = within_threshold.float()

    # 유효한 keypoint 수를 계산합니다.
    valid_mask = mask.float()
    valid_keypoints = valid_mask.sum()

    # 평균 정확도를 계산합니다.
    avg_acc = (acc * valid_mask).sum() / valid_keypoints if valid_keypoints > 0 else 0

    return acc, avg_acc.item(), valid_keypoints.item()


class MetricMocap(pl.Callback):
    def __init__(self):
        super().__init__()
        self.dataset_name = "MOOF"
        # video ID별 결과
        self.metric_aggregator = {
            "pckf005": {},  # Correct foot keypoint 비율 @ 0.05
            "nfke2d": {},  # 정규화된 2D foot keypoint 오차
        }
        self.num_valid_kpts_dict = {}
        self.num_valid_feet_dict = {}  # 해당 발의 모든 keypoint가 유효하면 발도 유효합니다.

        # SMPL-X 및 SMPL
        self.smplx = make_smplx("supermotion").cuda()
        self.J_regressor24 = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
        self.smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()

        # validation, test, predict에서 같은 방식으로 metric을 계산합니다.
        self.on_test_batch_end = self.on_validation_batch_end = self.on_predict_batch_end

        # validation 단계에서만 logger에 metric을 기록합니다.
        self.on_test_epoch_end = self.on_validation_epoch_end = self.on_predict_epoch_end

    # ================== Batch 단위 계산 ================== #
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """validation, test, predict에서 동일하게 동작합니다."""
        assert batch["B"] == 1
        dataset_id = batch["meta"][0]["dataset_id"]
        if dataset_id != self.dataset_name:
            return

        vid = batch["meta"][0]["vid"]

        # 예측 결과로 metric을 계산합니다.
        smpl_out = self.smplx(**outputs["pred_smpl_params_incam"])
        pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])

        # 2D metric을 계산합니다.
        # SMPL mesh에서 발 keypoint를 선택합니다.
        smpl_foot_indices = [3216, 3226, 3387, 6617, 6624, 6787]
        pred_foot_kpts = pred_c_verts[:, smpl_foot_indices]
        pred_feet_2d = perspective_projection(pred_foot_kpts, batch["K_fullimg"].squeeze())

        gt_feet = batch["gt_foot_kpts"].squeeze()
        gt_feet_2d = gt_feet[:, :, :2]
        feet_kpts_vis = gt_feet[:, :, 2]
        # bounding box를 기준으로 keypoint를 정규화합니다.
        pck_bbox_center = batch["pck_bbox_center"][0].unsqueeze(1)
        pck_bbox_scale = batch["pck_bbox_scale"][0].unsqueeze(1)
        pred_feet_2d = (pred_feet_2d - pck_bbox_center) / pck_bbox_scale
        gt_feet_2d = (gt_feet_2d - pck_bbox_center) / pck_bbox_scale

        # 보이는 keypoint에 대해서만 2D metric을 계산합니다.
        kpts_mask = feet_kpts_vis == 1.0
        num_valid_kpts = kpts_mask.sum()

        # 1) PCK_F@0.05 계산
        pck005, _, _ = pck_accuracy(pred_feet_2d, gt_feet_2d, kpts_mask, thr=0.05)
        avgpck_005 = 100 * (pck005 * kpts_mask).sum() / num_valid_kpts if num_valid_kpts > 0 else 0

        # 2) 정규화된 2D foot keypoint 오차 계산
        # 발의 모든 joint가 보일 때만 계산합니다. 그렇지 않으면 발의 중심을 구할 수 없습니다.
        # GT 2D 발 좌표를 중심 정렬합니다.
        center_left_foot = gt_feet_2d[:, :3].mean(dim=1, keepdims=True)
        center_right_foot = gt_feet_2d[:, 3:].mean(dim=1, keepdims=True)
        c_target_feet = gt_feet_2d.clone()
        c_target_feet[:, :3] = gt_feet_2d[:, :3] - center_left_foot
        c_target_feet[:, 3:] = gt_feet_2d[:, 3:] - center_right_foot
        # 예측 발 좌표를 중심 정렬합니다.
        center_left_foot = pred_feet_2d[:, :3].mean(dim=1, keepdims=True)
        center_right_foot = pred_feet_2d[:, 3:].mean(dim=1, keepdims=True)
        c_pred_feet = pred_feet_2d.clone()
        c_pred_feet[:, :3] = pred_feet_2d[:, :3] - center_left_foot
        c_pred_feet[:, 3:] = pred_feet_2d[:, 3:] - center_right_foot

        left_foot_mask = kpts_mask[:, :3].clone()
        left_foot_mask = left_foot_mask * torch.all(left_foot_mask, dim=1)[:, None]
        right_foot_mask = kpts_mask[:, 3:].clone()
        right_foot_mask = right_foot_mask * torch.all(right_foot_mask, dim=1)[:, None]
        foot_mask = torch.cat((left_foot_mask, right_foot_mask), dim=1)
        num_valid_feet = foot_mask.sum()
        # 발마다 중심과 scale을 정렬합니다.
        nmpjpe_left = n_mpjpe(c_pred_feet[:, :3], c_target_feet[:, :3], return_mean=False)
        nmpjpe_right = n_mpjpe(c_pred_feet[:, 3:], c_target_feet[:, 3:], return_mean=False)
        cs_eucl_dist = torch.cat((nmpjpe_left, nmpjpe_right), dim=1)
        avg_nfke2d = (
            100 * (cs_eucl_dist * foot_mask).sum() / num_valid_feet if num_valid_feet > 0 else 0
        )

        mean_metrics = {"pckf005": avgpck_005.item(), "nfke2d": avg_nfke2d.item()}
        self.num_valid_kpts_dict[vid] = num_valid_kpts.item()
        self.num_valid_feet_dict[vid] = num_valid_feet.item()
        for key, val in mean_metrics.items():
            self.metric_aggregator[key][vid] = val

    # ================== Epoch 요약 ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        """logger 없이 epoch 결과를 집계합니다."""
        local_rank, _ = trainer.local_rank, trainer.world_size
        monitor_metric = "nfke2d"

        # 모든 process의 metric_aggregator를 모아 reduce합니다.
        metric_keys = list(self.metric_aggregator.keys())
        with torch.inference_mode(False):  # all_gather의 in-place 연산을 허용합니다.
            metric_aggregator_gathered = all_gather(self.metric_aggregator)  # dictionary 목록
        for metric_key in metric_keys:
            for d in metric_aggregator_gathered:
                self.metric_aggregator[metric_key].update(d[metric_key])

        total = len(self.metric_aggregator[monitor_metric])
        Log.info(f"{total} sequences evaluated in {self.__class__.__name__}")
        if total == 0:
            return

        # sequence별 monitoring metric을 출력합니다.
        mm_per_seq = {k: v for k, v in self.metric_aggregator[monitor_metric].items()}
        if len(mm_per_seq) > 0:
            sorted_mm_per_seq = sorted(mm_per_seq.items(), key=lambda x: x[1], reverse=True)
            n_worst = 5 if trainer.state.stage == "validate" else len(sorted_mm_per_seq)
            if local_rank == 0:
                Log.info(
                    f"monitored metric {monitor_metric} per sequence\n"
                    + "\n".join([f"{m:.2f} : {s}" for s, m in sorted_mm_per_seq[:n_worst]])
                    + "\n------"
                )

        # 모든 batch의 가중 평균을 구합니다.
        metrics_avg = {}
        total_valid_kpts = sum(self.num_valid_kpts_dict.values())
        total_valid_feet = sum(self.num_valid_feet_dict.values())
        for metric, v in self.metric_aggregator.items():
            weighted_metrics = 0
            for vid, avg_seq_metric in v.items():
                if metric == "nfke2d":
                    num_valid_kpts = self.num_valid_feet_dict[vid]
                else:
                    num_valid_kpts = self.num_valid_kpts_dict[vid]
                weighted_metric = avg_seq_metric * num_valid_kpts
                weighted_metrics += weighted_metric
            if metric == "nfke2d":
                metrics_avg[metric] = weighted_metrics / total_valid_feet
            else:
                metrics_avg[metric] = weighted_metrics / total_valid_kpts

        if local_rank == 0:
            Log.info(
                f"[Metrics] {self.dataset_name}:\n"
                + "\n".join(f"{k}: {v:.2f}" for k, v in metrics_avg.items())
                + "\n------"
            )

        for k, v in metrics_avg.items():
            pl_module.log_dict({f"val_metric_{self.dataset_name}/{k}": v}, logger=True)

        # 집계 상태를 초기화합니다.
        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}


MainStore.store(
    name="metric_moof",
    node=builds(MetricMocap),
    group="callbacks",
    package="callbacks.metric_moof",
)
