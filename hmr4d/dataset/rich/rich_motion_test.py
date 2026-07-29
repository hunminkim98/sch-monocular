from pathlib import Path
import torch
from torch.utils import data
from hmr4d.utils.pylogger import Log

from .rich_utils import (
    get_cam2params,
    get_w2az_sahmr,
    parse_seqname_info,
    get_cam_key_wham_vid,
)
from hmr4d.utils.geo_transform import transform_mat, compute_cam_angvel
from pytorch3d.transforms import axis_angle_to_matrix
from hmr4d.utils.geo.hmr_cam import resize_K, estimate_K


from hmr4d.configs import MainStore, builds


VID_PRESETS = {
    "easytohard": [
        "test/Gym_013_burpee4/cam_06",
        "test/Gym_011_pushup1/cam_02",
        "test/LectureHall_019_wipingchairs1/cam_03",
        "test/ParkingLot2_009_overfence1/cam_04",
        "test/LectureHall_021_sidebalancerun1/cam_00",
        "test/Gym_010_dips2/cam_05",
    ],
}


class RichSmplFullSeqDataset(data.Dataset):
    def __init__(self, vid_presets=None):
        """
        인자:
            vid_presets: VID_PRESETS의 key
        """
        super().__init__()
        self.dataset_name = "RICH"
        self.dataset_id = "RICH"
        Log.info(f"[{self.dataset_name}] Full sequence, Test")
        tic = Log.time()

        # WHAM label에서 평가 protocol을 불러옵니다.
        self.rich_dir = Path("inputs/RICH/hmr4d_support")
        self.labels = torch.load(self.rich_dir / "rich_test_labels.pt")
        # ['vid', 'frame_id', 'gender', 'gt_smplx_params']
        self.preproc_data = torch.load(self.rich_dir / "rich_test_preproc.pt")
        # ['vid', 'f_imgseq', 'bbx_xys', 'img_wh', 'kp2d']
        self.sapiens133 = torch.load(self.rich_dir / "test_sapiens133.pt")

        vids = select_subset(self.labels, vid_presets)

        # dataset index를 구성합니다.
        self.idx2meta = []
        for vid in vids:
            seq_length = len(self.labels[vid]["frame_id"])
            self.idx2meta.append((vid, 0, seq_length))  # start=0, end=sequence 길이
        # print(sum([end - start for _, _, start, end in self.idx2meta]))

        # ay 좌표계의 GT motion을 준비합니다.
        self.w2az = get_w2az_sahmr()  # scan_name -> T_w2az, world 좌표계는 cam-1 좌표계를 뜻합니다.
        self.cam2params = get_cam2params()  # cam_key를 (T_w2c, K)에 매핑합니다.
        seqname_info = parse_seqname_info(skip_multi_persons=True)  # {k: (scan_name, subject_id, gender, cam_ids)}
        self.seqname_to_scanname = {k: v[0] for k, v in seqname_info.items()}

        Log.info(f"[RICH] {len(self.idx2meta)} sequences. Elapsed: {Log.time() - tic:.2f}s")

    def __len__(self):
        return len(self.idx2meta)

    def _load_data(self, idx):
        data = {}

        # label에서 [start, end) 범위의 데이터를 불러옵니다.
        vid, start, end = self.idx2meta[idx]
        label = self.labels[vid]
        preproc_data = self.preproc_data[vid]

        length = end - start
        meta = {"dataset_id": "RICH", "vid": vid, "vid-start-end": (start, end)}
        data.update({"meta": meta, "length": length, "num_seqs": len(self.idx2meta)})
        # SMPL-X 파라미터
        data.update({"gt_smpl_params": label["gt_smplx_params"], "gender": label["gender"]})

        # camera 파라미터
        cam_key = get_cam_key_wham_vid(vid)
        scan_name = self.seqname_to_scanname[vid.split("/")[1]]
        T_w2c, K = self.cam2params[cam_key]  # (4, 4)  (3, 3)
        data.update({"gt_K": K})
        T_w2az = self.w2az[scan_name]
        # 근사 intrinsic 파라미터를 사용합니다.
        K = estimate_K(preproc_data["img_wh"][0], preproc_data["img_wh"][1])
        # K = preproc_data["pred_K"].mean(dim=0)
        data.update({"T_w2c": T_w2c, "T_w2az": T_w2az, "K": K})

        # image feature를 불러옵니다.
        data.update(
            {
                "f_imgseq": preproc_data["f_imgseq"],
                "bbx_xys": preproc_data["bbx_xys"],
                "img_wh": preproc_data["img_wh"],
            }
        )

        kp2d = preproc_data["kp2d"]
        kp2d_feet = self.sapiens133[vid][:, 17:23]
        kp2d = torch.cat((kp2d, kp2d_feet), dim=1)
        data.update({"kp2d": kp2d})

        # video rendering용 정보
        video_path = self.rich_dir / f"videos/{vid}.mp4"
        frame_id = label["frame_id"]  # (F,)
        ds = 0.25  # 원본 해상도의 0.25배로 video를 만듭니다.
        width, height = data["img_wh"] * ds  # 저장된 video는 1/4로 downsample됩니다.
        K_render_gt = resize_K(data["gt_K"], ds)
        K_render = resize_K(K, ds)
        bbx_xys_render = data["bbx_xys"] * ds
        kp2d_render = kp2d.clone()
        kp2d_render[..., :2] *= ds
        data["meta_render"] = {
            "name": vid,
            "video_path": str(video_path),
            "frame_id": frame_id,
            "width_height": (width, height),
            "K_gt": K_render_gt,
            "K": K_render,
            "bbx_xys": bbx_xys_render,
            "kp2d": kp2d_render,
            "ds": ds
        }

        return data

    def _process_data(self, data):
        # T_w2az는 바닥 단서로 미리 계산하며, az2ay는 x축 회전을 사용합니다.
        R_az2ay = axis_angle_to_matrix(torch.tensor([1.0, 0.0, 0.0]) * -torch.pi / 2)  # (3, 3)
        T_w2ay = transform_mat(R_az2ay, R_az2ay.new([0, 0, 0])) @ data["T_w2az"]  # (4, 4)

        # xys를 사용해 image feature를 처리합니다.
        length = data["length"]
        f_imgseq = data["f_imgseq"]  # (F, 1024)
        R_w2c = data["T_w2c"][:3, :3].repeat(length, 1, 1)  # (L, 4, 4)
        cam_angvel = compute_cam_angvel(R_w2c)  # (L, 6)

        # 반환값 구성
        data = {
            # --- batch로 묶지 않는 값
            "task": "CAP-Seq",
            "num_seqs": data["num_seqs"],
            "meta": data["meta"],
            "meta_render": data["meta_render"],
            # --- 단일 sequence를 평가하므로 key-value를 직접 설정합니다.
            "length": length,
            "f_imgseq": f_imgseq,
            "cam_angvel": cam_angvel,
            "bbx_xys": data["bbx_xys"],  # (F, 3)
            "K_fullimg": data["K"][None].expand(length, -1, -1),  # (F, 3, 3)
            "gt_K_fullimg": data["gt_K"][None].expand(length, -1, -1),
            "kp2d": data["kp2d"],  # (F, 17, 3)
            # --- dataset 전용 값
            "model": "smplx",
            "gender": data["gender"],
            "gt_smpl_params": data["gt_smpl_params"],
            "T_w2ay": T_w2ay,  # (4, 4)
            "T_w2c": data["T_w2c"],  # (4, 4)
        }
        return data

    def __getitem__(self, idx):
        data = self._load_data(idx)
        data = self._process_data(data)
        return data


def select_subset(labels, vid_presets):
    vids = list(labels.keys())
    if vid_presets != None:  # video subset을 사용합니다.
        vids = VID_PRESETS[vid_presets]
    return vids


#
group_name = "test_datasets/rich"
base_node = builds(RichSmplFullSeqDataset, vid_presets=None, populate_full_signature=True)
MainStore.store(name="all", node=base_node(), group=group_name)
MainStore.store(name="easy_to_hard", node=base_node(vid_presets="easytohard"), group=group_name)
MainStore.store(name="postproc", node=base_node(vid_presets="postproc"), group=group_name)
