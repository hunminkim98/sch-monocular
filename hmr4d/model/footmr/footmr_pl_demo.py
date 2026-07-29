import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds

from hmr4d.utils.geo.hmr_cam import normalize_kp2d


class DemoPL(pl.LightningModule):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = instantiate(pipeline, _recursive_=False)

    @torch.no_grad()
    def predict(self, data, static_cam=False, no_postproc=False):
        """batch 차원을 자동으로 추가합니다.

        data: {
            "length": int, or Torch.Tensor,
            "kp2d": (F, 3)
            "bbx_xys": (F, 3)
            "K_fullimg": (F, 3, 3)
            "cam_angvel": (F, 3)
            "f_imgseq": (F, 3, 256, 256)
        }

        """
        # RoPE 추론
        obs = normalize_kp2d(data["kp2d"], data["bbx_xys"])[None]
        batch = {
            "length": data["length"][None],
            "obs": obs,
            "foot_obs": obs[:, :, 15:23].clone(),
            "bbx_xys": data["bbx_xys"][None],
            "K_fullimg": data["K_fullimg"][None],
            "cam_angvel": data["cam_angvel"][None],
            "f_imgseq": data["f_imgseq"][None],
        }
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = self.pipeline.forward(
            batch, train=False, postproc=not no_postproc, static_cam=static_cam
        )

        pred = {
            "smpl_params_global": {k: v[0] for k, v in outputs["pred_smpl_params_global"].items()},
            "smpl_params_incam": {k: v[0] for k, v in outputs["pred_smpl_params_incam"].items()},
            "K_fullimg": data["K_fullimg"],
            "net_outputs": outputs,  # 중간 출력
        }
        return pred

    def load_pretrained_model(self, ckpt_path):
        """사전 학습 checkpoint를 불러와 각 weight를 해당 부분에 할당합니다."""
        Log.info(f"[PL-Trainer] Loading ckpt type: {ckpt_path}")

        state_dict = torch.load(ckpt_path, "cpu")["state_dict"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            Log.warn(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            Log.warn(f"Unexpected keys: {unexpected}")


MainStore.store(
    name="footmr_pl_demo", node=builds(DemoPL, pipeline="${pipeline}"), group="model/footmr"
)
