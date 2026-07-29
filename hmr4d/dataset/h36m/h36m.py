import torch
import numpy as np
import os
from pathlib import Path
from hmr4d.configs import MainStore, builds
from hmr4d.utils.pylogger import Log
from hmr4d.dataset.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.geo.hmr_global import get_c_rootparam, get_R_c2gv
from hmr4d.utils.net_utils import get_valid_mask, repeat_to_max_len, repeat_to_max_len_dict


class H36mSmplDataset(ImgfeatMotionDatasetBase):
    def __init__(
        self,
        original_coord="az",
        motion_frames=120,  # H36M video는 25fps이며 길이가 깁니다.
        lazy_load=False,
    ):

        self.root = Path("inputs/H36M/hmr4d_support")

        # 좌표계
        self.original_coord = original_coord

        # 설정
        self.motion_frames = motion_frames
        self.lazy_load = lazy_load
        self.dataset_name = "H36M"
        super().__init__()

    def _load_dataset(self):
        # SMPL pose 데이터
        tic = Log.time()
        fn = self.root / "smplxpose_v1.pt"
        self.smpl_model = make_smplx("supermotion")
        Log.info(f"[{self.dataset_name}] Loading from {fn} ...")
        self.motion_files = torch.load(fn)
        # dictionary 구성:
        #          "smpl_params_glob": {'body_pose', 'global_orient', 'transl', 'betas'}, FxC
        #          "cam_Rt": tensor(F, 3),
        #          "cam_K": tensor(1, 10),
        #         }
        self.seqs = list(self.motion_files.keys())
        Log.info(
            f"[{self.dataset_name}] {len(self.seqs)} sequences. Elapsed: {Log.time() - tic:.2f}s"
        )

        # image feature를 불러옵니다.
        # vid를 (features, vid, meta {bbx_xys, K_fullimg})에 매핑합니다.
        if not self.lazy_load:
            tic = Log.time()
            fn = self.root / "vitfeat_h36m.pt"
            Log.info(f"[{self.dataset_name}] Fully Loading to RAM ViT-Feat: {fn}")
            self.f_img_dicts = torch.load(fn)
            Log.info(f"[{self.dataset_name}] Finished. Elapsed: {Log.time() - tic:.2f}s")
        else:
            raise NotImplementedError  # lazy_load 구현은 BEDLAM-SMPL을 참고합니다.

    def _get_idx2meta(self):
        # 한 epoch에서 전체 sequence를 보도록 각 sequence를
        # max(SeqLength // MotionFrames, 1)번 샘플링합니다.
        seq_lengths = []
        self.idx2meta = []
        for vid in self.f_img_dicts:
            seq_length = self.f_img_dicts[vid]["bbx_xys"].shape[0]
            num_samples = max(seq_length // self.motion_frames, 1)
            seq_lengths.append(seq_length)
            self.idx2meta.extend([vid] * num_samples)
        hours = sum(seq_lengths) / 25 / 3600
        Log.info(
            f"[{self.dataset_name}] has {hours:.1f} hours motion -> Resampled to {len(self.idx2meta)} samples."
        )

    def _load_data(self, idx):
        sampled_motion = {}
        vid = self.idx2meta[idx]
        motion = self.motion_files[vid]
        seq_length = self.f_img_dicts[vid]["bbx_xys"].shape[0]  # feature 길이를 기준으로 사용합니다.
        sampled_motion["vid"] = vid

        # subset을 무작위로 선택합니다.
        target_length = self.motion_frames
        if target_length > seq_length:  # 일반적으로 발생하지 않아야 합니다.
            start = 0
            length = seq_length
            Log.info(
                f"[H36M] ({idx}) target length < sequence length: {target_length} <= {seq_length}"
            )
        else:
            start = np.random.randint(0, seq_length - target_length)
            length = target_length
        end = start + length
        sampled_motion["length"] = length
        sampled_motion["start_end"] = (start, end)

        # motion subset을 선택합니다.
        # body_pose, global_orient, transl, betas를 선택합니다.
        sampled_motion["smpl_params_global"] = {
            k: v[start:end] for k, v in motion["smpl_params_glob"].items()
        }

        # image feature를 선택합니다.
        f_img_dict = self.f_img_dicts[vid]
        sampled_motion["f_imgseq"] = f_img_dict["features"][start:end].float()  # (L, 1024)
        sampled_motion["bbx_xys"] = f_img_dict["bbx_xys"][start:end]
        sampled_motion["K_fullimg"] = f_img_dict["K_fullimg"]
        sampled_motion["kp2d"] = torch.zeros((end - start), 23, 3)  # (L, 17, 3)

        # camera 정보
        sampled_motion["T_w2c"] = motion["cam_Rt"]  # (4, 4)

        return sampled_motion

    def _process_data(self, data, idx):
        length = data["length"]

        # world 좌표계의 SMPL 파라미터
        smpl_params_w = data["smpl_params_global"].copy()  # az 좌표계

        # camera 좌표계의 SMPL 파라미터
        T_w2c = data["T_w2c"]  # (4, 4)
        offset = self.smpl_model.get_skeleton(smpl_params_w["betas"][0])[0]  # (3)
        global_orient_c, transl_c = get_c_rootparam(
            smpl_params_w["global_orient"],
            smpl_params_w["transl"],
            T_w2c,
            offset,
        )
        smpl_params_c = {
            "body_pose": smpl_params_w["body_pose"].clone(),  # (F, 63)
            "betas": smpl_params_w["betas"].clone(),  # (F, 10)
            "global_orient": global_orient_c,  # (F, 3)
            "transl": transl_c,  # (F, 3)
        }
        # world 좌표계 파라미터
        gravity_vec = torch.tensor([0, 0, -1]).float()  # (3), H36M은 az 좌표계입니다.
        T_w2c = T_w2c.repeat(length, 1, 1)  # (F, 4, 4)
        R_c2gv = get_R_c2gv(T_w2c[..., :3, :3], axis_gravity_in_w=gravity_vec)  # (F, 3, 3)

        # image 관련 입력
        bbx_xys = data["bbx_xys"]  # (F, 3)
        K_fullimg = data["K_fullimg"].repeat(length, 1, 1)  # (F, 3, 3)
        f_imgseq = data["f_imgseq"]  # (F, 1024)
        cam_angvel = compute_cam_angvel(T_w2c[:, :3, :3])  # (F, 6) WHAM과 약간 다릅니다.

        # 반환 직전에 batch로 묶을 수 있는 길이로 맞춰야 합니다.
        max_len = self.motion_frames
        subj, action, seq = data["vid"].split("@")
        action = action.replace("_", " ")
        video_path = f"{subj}/{action}.{seq}.mp4"
        video_path = os.path.join("inputs/H36M/hmr4d_support/videos", video_path)
        start_end_video = list(data["start_end"])  # video는 50fps, annotation은 25fps입니다.
        start_end_video[0] = start_end_video[0] * 2
        start_end_video[1] = start_end_video[1] * 2
        return_data = {
            "meta": {
                "data_name": "h36m",
                "idx": idx,
                "vid": data["vid"],
                "video_path": video_path,
                "start_end": start_end_video,
            },
            "length": length,
            "smpl_params_c": smpl_params_c,
            "smpl_params_w": smpl_params_w,
            "R_c2gv": R_c2gv,  # (F, 3, 3)
            "gravity_vec": gravity_vec,  # (3)
            "bbx_xys": bbx_xys,  # (F, 3)
            "K_fullimg": K_fullimg,  # (F, 3, 3)
            "f_imgseq": f_imgseq,  # (F, D)
            "kp2d": data["kp2d"],  # (F, 17, 3)
            "cam_angvel": cam_angvel,  # (F, 6)
            "mask": {
                "valid": get_valid_mask(max_len, length),
                "vitpose": False,
                "bbx_xys": True,
                "f_imgseq": True,
                "spv_incam_only": False,
            },
        }

        # batch로 묶을 수 있는 길이로 맞춥니다.
        return_data["smpl_params_c"] = repeat_to_max_len_dict(return_data["smpl_params_c"], max_len)
        return_data["smpl_params_w"] = repeat_to_max_len_dict(return_data["smpl_params_w"], max_len)
        return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
        return_data["bbx_xys"] = repeat_to_max_len(return_data["bbx_xys"], max_len)
        return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
        return_data["f_imgseq"] = repeat_to_max_len(return_data["f_imgseq"], max_len)
        return_data["kp2d"] = repeat_to_max_len(return_data["kp2d"], max_len)
        return_data["cam_angvel"] = repeat_to_max_len(return_data["cam_angvel"], max_len)

        return return_data


group_name = "train_datasets/imgfeat_h36m"
MainStore.store(name="v1", node=builds(H36mSmplDataset), group=group_name)
