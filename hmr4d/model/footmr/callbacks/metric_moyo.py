import torch
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from einops import einsum
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from hmr4d.configs import MainStore, builds
from hmr4d.utils.pylogger import Log
from hmr4d.utils.comm.gather import all_gather
from hmr4d.utils.eval.eval_utils import (
    compute_camcoord_metrics,
    as_np_array,
)
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo_transform import apply_T_on_points


class MetricMocap(pl.Callback):
    def __init__(self):
        super().__init__()
        self.dataset_name = "MOYO"
        # video ID별 결과
        self.metric_aggregator = {
            "pa_mpjpe": {},
            "n_mpjpef": {},  # foot keypoint의 scale-normalized MPJPE
            "ajae": {},  # 발목 joint angle 오차
        }

        # SMPL-X 모델
        self.smplx_model = {
            "male": make_smplx("rich-smplx", gender="male").cuda(),
            "female": make_smplx("rich-smplx", gender="female").cuda(),
            "neutral": make_smplx("rich-smplx", gender="neutral").cuda(),
        }

        self.moyo_dir = Path("inputs/MOYO/hmr4d_support")
        v_template = torch.load(self.moyo_dir / "moyo_test_gt_v_template.pt")["v_template"]
        self.gt_smplx_model = make_smplx(
            "rich-smplx", gender="female", v_template=v_template
        ).cuda()

        self.J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
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
        T_w2c = batch["T_w2c"][0]

        # camera 좌표계의 GT
        target_w_params = {k: v[0] for k, v in batch["smplx_params"].items()}
        target_w_output = self.gt_smplx_model(**target_w_params)
        target_w_verts = torch.stack(
            [torch.matmul(self.smplx2smpl, v_) for v_ in target_w_output.vertices]
        )
        target_c_verts = apply_T_on_points(target_w_verts, T_w2c)
        target_c_j3d = torch.matmul(self.J_regressor, target_c_verts)
        # camera 좌표계의 GT global_orient를 계산합니다.
        gt_global_orient = target_w_params["global_orient"]
        gt_global_rotmat = axis_angle_to_matrix(gt_global_orient)
        cam_rot = T_w2c[..., :3, :3]
        gt_incam_rotmat = torch.matmul(cam_rot[None], gt_global_rotmat)
        gt_incam_orient = matrix_to_axis_angle(gt_incam_rotmat)
        gt_incam_full_body_pose = torch.cat((gt_incam_orient, target_w_params["body_pose"]), dim=1)

        # 예측 결과로 metric을 계산합니다.
        pred_smpl_params_incam = outputs["pred_smpl_params_incam"]
        smpl_out = self.smplx_model["neutral"](**pred_smpl_params_incam)
        pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smpl_out.vertices])
        pred_c_j3d = einsum(self.J_regressor, pred_c_verts, "j v, l v i -> l j i")
        del smpl_out  # out-of-memory를 방지합니다.
        pred_incam_full_body_pose = torch.cat(
            (pred_smpl_params_incam["global_orient"], pred_smpl_params_incam["body_pose"]), dim=1
        )

        # 현재 sequence의 metric
        batch_eval = {
            "pred_j3d": pred_c_j3d,
            "target_j3d": target_c_j3d,
            "pred_verts": pred_c_verts,
            "target_verts": target_c_verts,
            "gt_incam_pose": gt_incam_full_body_pose,
            "pred_incam_pose": pred_incam_full_body_pose,
            "parents": self.smplx_model["neutral"].bm.parents,
        }
        camcoord_metrics = compute_camcoord_metrics(batch_eval)
        for k in camcoord_metrics:
            self.metric_aggregator[k][vid] = as_np_array(camcoord_metrics[k])

    # ================== Epoch 요약 ================== #
    def on_predict_epoch_end(self, trainer, pl_module):
        """logger 없이 epoch 결과를 집계합니다."""
        local_rank, _ = trainer.local_rank, trainer.world_size
        monitor_metric = "ajae"

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
        mm_per_seq = {k: v.mean() for k, v in self.metric_aggregator[monitor_metric].items()}
        if len(mm_per_seq) > 0:
            sorted_mm_per_seq = sorted(mm_per_seq.items(), key=lambda x: x[1], reverse=True)
            n_worst = 5 if trainer.state.stage == "validate" else len(sorted_mm_per_seq)
            if local_rank == 0:
                Log.info(
                    f"monitored metric {monitor_metric} per sequence\n"
                    + "\n".join([f"{m:5.1f} : {s}" for s, m in sorted_mm_per_seq[:n_worst]])
                    + "\n------"
                )

        # 모든 batch의 평균을 구합니다.
        metrics_avg = {
            k: np.concatenate(list(v.values())).mean() for k, v in self.metric_aggregator.items()
        }
        if local_rank == 0:
            Log.info(
                f"[Metrics] {self.dataset_name}:\n"
                + "\n".join(f"{k}: {v:.1f}" for k, v in metrics_avg.items())
                + "\n------"
            )

        for k, v in metrics_avg.items():
            pl_module.log_dict({f"val_metric_{self.dataset_name}/{k}": v}, logger=True)

        # 집계 상태를 초기화합니다.
        for k in self.metric_aggregator:
            self.metric_aggregator[k] = {}


node_moyo = builds(MetricMocap)
MainStore.store(
    name="metric_moyo", node=node_moyo, group="callbacks", package="callbacks.metric_moyo"
)
