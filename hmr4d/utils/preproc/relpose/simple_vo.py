import numpy as np
from .utils import focal_length_from_mm
from .matcher_wrapper import Matcher
from .solver_two_view import TwoPairSolver, CameraParams, interpolate_missing_frames
from tqdm import tqdm

from hmr4d.utils.video_io_utils import get_video_lwh, read_video_np


class SimpleVO:
    def __init__(self, video_path, scale=0.5, step=8, method="sift", f_mm=None):
        self.video_path = video_path
        self.scale = scale
        self.step = step
        self.method = method
        self.f_mm = 24 if f_mm is None else f_mm  # full-frame camera의 mm 단위 focal length

    def compute(self):
        # video를 읽습니다.
        frames = read_video_np(self.video_path, scale=self.scale)

        # frame을 downsample하고 누락된 frame을 보간합니다.
        F_all = frames.shape[0]
        sample_idxs = np.arange(0, F_all, self.step)
        if sample_idxs[-1] != F_all - 1:
            sample_idxs = np.concatenate([sample_idxs, [F_all - 1]])
        frames = frames[sample_idxs]
        F, H, W, C = frames.shape
        print(f"[SimpleVO] Choosen frames shape: {frames.shape}")

        matcher: Matcher = Matcher(self.method)
        camera_params = CameraParams(W, H, focal_length=focal_length_from_mm(W, H, self.f_mm))
        solver: TwoPairSolver = TwoPairSolver(camera_params, solver="pycolmap")

        # TODO: method별로 서로 다른 pipeline을 사용해야 합니다.
        T_w2c_list = self.process_video_T_w2c_list_np(frames, matcher, solver)

        # 누락된 frame을 보간합니다.
        T_w2c_list = interpolate_missing_frames(T_w2c_list, sample_idxs)

        return T_w2c_list

    def process_video_T_w2c_list_np(self, frames, matcher: Matcher, solver: TwoPairSolver):
        T_w2c_list = [np.eye(4)]  # camera pose는 T_w2c @ p_w = p_c로 정의합니다.
        prev_frame = frames[0]
        for frame_idx in tqdm(range(1, len(frames))):
            curr_frame = frames[frame_idx]

            # 인접 frame을 matching합니다.
            pts0, pts1 = matcher.match_np(prev_frame, curr_frame)
            T_delta = solver.solve(pts0, pts1)  # T_delta = T_curr @ T_last^-1

            # 현재 frame의 transformation matrix를 계산합니다.
            T_w2c_list.append(T_delta @ T_w2c_list[-1])

            # 다음 반복에서는 현재 frame을 이전 frame으로 사용합니다.
            prev_frame = curr_frame

        return T_w2c_list
