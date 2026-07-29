import cv2
import numpy as np
from dataclasses import dataclass
import pycolmap
from .transformation_np import *


@dataclass
class CameraParams:
    width: int
    height: int
    focal_length: float = None  # 미지정 시 sqrt(width^2 + height^2)를 사용하며 FoV는 약 53도입니다.
    cx: float = None  # 미지정 시 width의 절반을 사용합니다.
    cy: float = None  # 미지정 시 height의 절반을 사용합니다.


class Cv2RansacEssentialSolver:
    def __init__(self, camera_params: CameraParams):
        width = camera_params.width
        height = camera_params.height
        focal_length = camera_params.focal_length
        if focal_length is None:
            focal_length = (width**2 + height**2) ** 0.5
        cx = camera_params.cx
        cy = camera_params.cy
        if cx is None:
            cx = width / 2
        if cy is None:
            cy = height / 2

        self.camera_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

    def get_K(self):
        """
        반환:
            K: np.ndarray, shape (3, 3), dtype=np.float32
        """
        return self.camera_matrix

    def solve(self, pts0, pts1):
        # 더 엄격한 RANSAC으로 essential matrix를 구합니다.
        E, mask = cv2.findEssentialMat(
            pts0,
            pts1,
            self.camera_matrix,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        # pose를 복원합니다.
        _, R, t, mask = cv2.recoverPose(E, pts0, pts1, self.camera_matrix, mask=mask)

        return R, t


class PycolmapRansacTwoViewGeometrySolver:
    def __init__(self, camera_params: CameraParams):
        width = camera_params.width
        height = camera_params.height
        focal_length = camera_params.focal_length
        if focal_length is None:
            focal_length = (width**2 + height**2) ** 0.5
        cx = camera_params.cx
        cy = camera_params.cy
        if cx is None:
            cx = width / 2
        if cy is None:
            cy = height / 2
        self.camera_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])

        # PyCOLMAP camera를 구성합니다.
        self.camera = pycolmap.Camera(
            camera_id=0,
            model="SIMPLE_PINHOLE",
            width=width,
            height=height,
            params=[focal_length, cx, cy],
        )

        # 연속 frame에 사용할 옵션을 설정합니다.
        self.options = pycolmap.TwoViewGeometryOptions(
            min_num_inliers=10,
            min_E_F_inlier_ratio=0.8,
            max_H_inlier_ratio=0.9,
            compute_relative_pose=True,
        )
        print(self.options.summary())

    def get_K(self):
        return self.camera_matrix

    def solve(self, pts0, pts1):
        matches = np.stack([np.arange(len(pts0)), np.arange(len(pts0))], axis=-1)
        answer = pycolmap.estimate_calibrated_two_view_geometry(
            self.camera,
            pts0.astype(np.float64),
            self.camera,
            pts1.astype(np.float64),
            matches=matches,
            options=self.options,
        )

        # cam2_from_cam1은 이 코드의 표기법으로 T_0_to_1을 뜻합니다.
        Rt = answer.cam2_from_cam1.matrix().astype(np.float32)  # shape (3, 4)
        T = np.eye(4)
        T[:3] = Rt
        return T


two_pair_solver_map = {
    # "cv2": Cv2RansacEssentialSolver,  # 안정성이 부족합니다.
    "pycolmap": PycolmapRansacTwoViewGeometrySolver,  # Essential과 Homography를 함께 계산합니다.
}


class TwoPairSolver:
    def __init__(self, params: CameraParams, solver: str = "pycolmap"):
        self.solver = two_pair_solver_map[solver](params)

    def get_K(self):
        """
        반환:
            K: np.ndarray, shape (3, 3), dtype=np.float32
        """
        return self.solver.get_K()

    def solve(self, pts0, pts1):
        """
        인자:
            pts0: np.ndarray, shape (N, 2), dtype=np.float32
            pts1: np.ndarray, shape (N, 2), dtype=np.float32
        반환:
            T: np.ndarray, shape (4, 4), dtype=np.float32
        """
        return self.solver.solve(pts0, pts1)


########################################################
# 누락된 frame 보간
########################################################


def interpolate_missing_frames(T_w2c_list, sample_idxs):
    """
    알려진 frame의 transformation matrix인 T_w2c_list를 부드럽게 보간하여
    모든 frame의 transformation matrix를 생성합니다. Translation은 선형 보간하고,
    rotation은 SLERP 구면 선형 보간하여 부드러운 회전을 유지합니다.

    인자:
        T_w2c_list (numpy.ndarray): (F, 4, 4) 형태의 알려진 transformation matrix
        sample_idxs (list 또는 numpy.ndarray): 원본 sequence에서 알려진 F개 frame의
            index. 첫 index는 0, 마지막은 F_all - 1이라고 가정합니다.

    반환:
        numpy.ndarray: 누락된 frame을 보간한 (F_all, 4, 4) transformation matrix
    """
    sample_idxs = np.array(sample_idxs)
    # 마지막으로 알려진 frame index에서 전체 frame 수를 구합니다(index는 0부터 시작).
    F_all = sample_idxs[-1] + 1
    new_T_list = []

    # translation과 rotation을 분리합니다.
    translations = np.array([T[:3, 3] for T in T_w2c_list])
    rotations = np.array([T[:3, :3] for T in T_w2c_list])
    # rotation matrix를 quaternion으로 변환합니다.
    quaternions = np.array([rotation_matrix_to_quaternion(R) for R in rotations])

    for i in range(F_all):
        # 알려진 frame이면 해당 transformation matrix를 그대로 사용합니다.
        if i in sample_idxs:
            known_index = np.where(sample_idxs == i)[0][0]
            new_T_list.append(T_w2c_list[known_index])
        else:
            # 양쪽에서 가장 가까운 알려진 frame을 찾습니다.
            next_known = np.searchsorted(sample_idxs, i)
            prev_known = next_known - 1
            # 보간 비율 t를 계산합니다.
            t_interp = (i - sample_idxs[prev_known]) / (sample_idxs[next_known] - sample_idxs[prev_known])
            # translation은 선형 보간합니다.
            trans_interp = (1 - t_interp) * translations[prev_known] + t_interp * translations[next_known]
            # rotation은 SLERP로 보간합니다.
            q0 = quaternions[prev_known]
            q1 = quaternions[next_known]
            q_interp = slerp(q0, q1, t_interp)
            rot_interp = quaternion_to_rotation_matrix(q_interp)
            # 최종 4x4 transformation matrix를 구성합니다.
            T_interp = np.eye(4)
            T_interp[:3, :3] = rot_interp
            T_interp[:3, 3] = trans_interp
            new_T_list.append(T_interp)

    return np.array(new_T_list)
