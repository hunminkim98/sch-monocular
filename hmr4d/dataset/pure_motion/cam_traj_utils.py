import torch
import torch.nn.functional as F
import numpy as np
from numpy.random import rand, randn
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from einops import rearrange
from hmr4d.utils.geo.hmr_cam import create_camera_sensor
from hmr4d.utils.geo_transform import transform_mat, apply_T_on_points
from hmr4d.utils.geo.transforms import axis_rotate_to_matrix
import hmr4d.utils.matrix as matrix

halfpi = np.pi / 2
R_y_upsidedown = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()


def noisy_interpolation(x, length, step_noise_perc=0.2):
    """noise를 포함하되 jitter가 매우 작은 비선형 보간을 수행합니다.

    인자:
        x: (2, C)
        length: scalar
        step_noise_perc: [x0, x1 +-(step_noise_perc * step), x2], where step = x1-x0
    """
    assert x.shape[0] == 2 and len(x.shape) == 2
    dim = x.shape[-1]
    output = np.zeros((length, dim))

    # linspace(0, 1)에 noise를 더한 값을 기준으로 사용합니다.
    linspace = np.repeat(np.linspace(0, 1, length)[None], dim, axis=0)  # (D, L)
    noise = (linspace[0, 1] - linspace[0, 0]) * step_noise_perc
    space_noise = np.random.uniform(-noise, noise, (dim, length - 2))  # (D, L-2)
    linspace[:, 1:-1] = linspace[:, 1:-1] + space_noise

    # 1D 보간을 수행합니다.
    for i in range(dim):
        output[:, i] = np.interp(linspace[i], np.array([0.0, 1.0]), x[:, i])
    return output


def noisy_impluse_interpolation(data1, data2, step_noise_perc=0.2):
    """noise를 포함한 impulse를 비선형 보간합니다."""

    dim = data1.shape[-1]
    L = data1.shape[0]

    linspace1 = np.stack([np.linspace(0, 1, L // 2) for _ in range(dim)])
    linspace2 = np.stack([np.linspace(0, 1, L // 2)[::-1] for _ in range(dim)])
    linspace = np.concatenate([linspace1, linspace2], axis=-1)
    noise = (linspace[0, 1] - linspace[0, 0]) * step_noise_perc
    space_noise = np.stack([np.random.uniform(-noise, noise, L - 2) for _ in range(dim)])

    linspace[:, 1:-1] = linspace[:, 1:-1] + space_noise
    linspace = linspace.T
    output = data1 * (1 - linspace) + data2 * linspace
    return output


def create_camera(w_root, cfg):
    """static camera pose를 생성합니다.

    인자:
        w_root: (3,), y-up 좌표
    반환:
        R_w2c: (3, 3)
        t_w2c: (3)
    """
    # 설정값을 해석합니다.
    pitch_std = cfg["pitch_std"]
    pitch_mean = cfg["pitch_mean"]
    roll_std = cfg["roll_std"]
    tz_range1_prob = cfg["tz_range1_prob"]
    tz_range1 = cfg["tz_range1"]
    tz_range2 = cfg["tz_range2"]
    f = cfg["f"]
    w = cfg["w"]

    # camera pose를 계산합니다.
    yaw = rand() * 2 * np.pi  # xz 평면의 임의 방향을 바라봅니다.
    pitch = np.clip(randn() * pitch_std + pitch_mean, -halfpi, halfpi)
    roll = np.clip(randn() * roll_std, -halfpi, halfpi)  # 정규분포

    # 먼저 R_y_upsidedown을 적용해 OpenCV camera 좌표계를 사용합니다.
    yaw_rm = axis_rotate_to_matrix(yaw, axis="y")
    pitch_rm = axis_rotate_to_matrix(pitch, axis="x")
    roll_rm = axis_rotate_to_matrix(roll, axis="z")
    R_w2c = (roll_rm @ pitch_rm @ yaw_rm @ R_y_upsidedown).squeeze(0)  # (3, 3)

    # 사람을 scene 안에 배치합니다.
    if rand() < tz_range1_prob:
        tz = rand() * (tz_range1[1] - tz_range1[0]) + tz_range1[0]
        max_dist_in_fov = (w / 2) / f * tz
        tx = (rand() * 2 - 1) * 0.7 * max_dist_in_fov
        ty = (rand() * 2 - 1) * 0.5 * max_dist_in_fov

    else:
        tz = rand() * (tz_range2[1] - tz_range2[0]) + tz_range2[0]
        max_dist_in_fov = (w / 2) / f * tz
        max_dist_in_fov *= 0.9  # 여유 threshold를 둡니다.
        tx = torch.randn(1) * 1.6
        tx = torch.clamp(tx, -max_dist_in_fov, max_dist_in_fov)
        ty = torch.randn(1) * 0.8
        ty = torch.clamp(ty, -max_dist_in_fov, max_dist_in_fov)

    dist = torch.tensor([tx, ty, tz], dtype=torch.float)
    t_w2c = dist - torch.matmul(R_w2c, w_root)

    return R_w2c, t_w2c


def create_rotation_move(R, length, r_xyz_w_std=[np.pi / 8, np.pi / 4, np.pi / 8]):
    """camera의 회전 움직임을 생성합니다.

    인자:
        R: (3, 3)
    반환:
        R_move: (L, 3, 3)
    """
    # 마지막 camera pose를 생성합니다.
    assert len(R.size()) == 2
    r_xyz = (2 * rand(3) - 1) * r_xyz_w_std
    Rf = R @ axis_angle_to_matrix(torch.from_numpy(r_xyz).float())

    # 두 pose 사이를 보간합니다.
    Rs = torch.stack((R, Rf))  # (2, 3, 3)
    rs = matrix_to_rotation_6d(Rs).numpy()  # (2, 6)
    rs_move = noisy_interpolation(rs, length)  # (L, 6)
    R_move = rotation_6d_to_matrix(torch.from_numpy(rs_move).float())

    return R_move


def create_translation_move(R_w2c, t_w2c, length, t_xyz_w_std=[1.0, 0.25, 1.0]):
    """camera의 translation 움직임을 생성합니다.

    인자:
        R_w2c: (3, 3),
        t_w2c: (3,),
    """
    # subject의 마지막 displacement를 생성합니다.
    subj_start_final = np.array([[0, 0, 0], randn(3) * t_xyz_w_std])
    subj_move = noisy_interpolation(subj_start_final, length)
    subj_move = torch.from_numpy(subj_move).float()  # (L, 3)

    # camera 움직임으로 변환합니다.
    t_move = t_w2c + torch.einsum("ij,lj->li", R_w2c, subj_move)

    return t_move


class CameraAugmentorV11:
    cfg_create_camera = {
        "pitch_mean": np.pi / 36,
        "pitch_std": np.pi / 8,
        "roll_std": np.pi / 24,
        "tz_range1_prob": 0.4,
        "tz_range1": [1.0, 6.0],  # 균등 분포에서 샘플링합니다.
        "tz_range2": [4.0, 12.0],
        "tx_scale": 0.7,
        "ty_scale": 0.3,
    }

    # r_xyz_w_std = [np.pi / 8, np.pi / 4, np.pi / 8]  # world 좌표계 기준
    r_xyz_w_std = [np.pi / 6, np.pi / 3, np.pi / 6]  # world 좌표계 기준
    t_xyz_w_std = [1.0, 0.25, 1.0]  # world 좌표계 기준
    r_xyz_w_std_half = [x / 2 for x in r_xyz_w_std]
    t_xyz_w_std_half = [x / 2 for x in t_xyz_w_std]

    t_factor = 1.0
    tz_bias_factor = 1.0

    rotx_impluse_noise = np.pi / 36
    roty_impluse_noise = np.pi / 36
    rotz_impluse_noise = np.pi / 36
    rot_impluse_n = 1

    tx_step_noise = 0.0025
    ty_step_noise = 0.0025
    tz_step_noise = 0.0025

    tx_impluse_noise = 0.15
    ty_impluse_noise = 0.15
    tz_impluse_noise = 0.15
    t_impluse_n = 1

    # === 후처리 === #
    height_max = 4.0
    height_min = -2.0  # 위쪽을 바라볼 수 있도록 -1.5에서 -2.0으로 확장합니다.
    tz_post_min = 0.5

    def __init__(self):
        self.w = 1000
        self.f = create_camera_sensor(1000, 1000, 24)[2][0, 0]  # 24mm camera를 사용합니다.
        self.half_fov_tol = (self.w / 2) / self.f

    def create_rotation_track(self, cam_mat, root, rx_factor=1.0, ry_factor=1.0, rz_factor=1.0):
        """회전하는 사람을 추적하는 camera rotation을 생성합니다."""
        human_mat = matrix.get_TRS(matrix.identity_mat()[None, :3, :3], root)
        cam2human_mat = matrix.get_mat_BtoA(human_mat, cam_mat)
        R = matrix.get_rotation(cam2human_mat)

        # 마지막 camera pose를 생성합니다.
        yaw = np.random.normal(scale=ry_factor)
        pitch = np.random.normal(scale=rx_factor)
        roll = np.random.normal(scale=rz_factor)

        yaw_rm = axis_angle_to_matrix(torch.tensor([0, yaw, 0]).float())
        pitch_rm = axis_angle_to_matrix(torch.tensor([pitch, 0, 0]).float())
        roll_rm = axis_angle_to_matrix(torch.tensor([0, 0, roll]).float())
        Rf = roll_rm @ pitch_rm @ yaw_rm @ R[0]

        # 두 pose 사이를 보간합니다.
        Rs = torch.stack((R[0], Rf))
        rs = matrix_to_rotation_6d(Rs).numpy()
        rs_move = noisy_interpolation(rs, self.l)
        R_move = rotation_6d_to_matrix(torch.from_numpy(rs_move).float())
        R_move = torch.inverse(R_move)
        return R_move

    def create_translation_track(self, cam_mat, root, t_factor=1.0, tz_bias_factor=0.0):
        """사람을 추적하는 camera translation을 생성합니다."""
        delta_T0 = matrix.get_position(cam_mat)[0] - root[0]
        T_new = matrix.get_position(cam_mat)

        tz_bias = delta_T0.norm(dim=-1) * tz_bias_factor * np.clip(1 + np.random.normal(scale=0.1), 0.67, 1.5)

        T_new[1:] = root[1:] + delta_T0
        cam_mat = matrix.get_TRS(matrix.get_rotation(cam_mat), T_new)
        w2c = torch.inverse(cam_mat)
        T_new = matrix.get_position(w2c)

        # 마지막 camera 위치를 생성합니다.
        tx = np.random.normal(scale=t_factor)
        ty = np.random.normal(scale=t_factor)
        tz = np.random.normal(scale=t_factor) + tz_bias
        Ts = np.array([[0, 0, 0], [tx, ty, tz]])

        T_move = noisy_interpolation(Ts, self.l)
        T_move = torch.from_numpy(T_move).float()
        return T_move + T_new

    def add_stepnoise(self, R, T):
        w2c = matrix.get_TRS(R, T)
        cam_mat = torch.inverse(w2c)
        R_new = matrix.get_rotation(cam_mat)
        T_new = matrix.get_position(cam_mat)

        L = R_new.shape[0]
        window = 10

        def add_impulse_rot(R_new):
            N = np.random.randint(1, self.rot_impluse_n + 1)
            rx = np.random.normal(scale=self.rotx_impluse_noise, size=N)
            ry = np.random.normal(scale=self.roty_impluse_noise, size=N)
            rz = np.random.normal(scale=self.rotz_impluse_noise, size=N)
            R_impluse_noise = axis_angle_to_matrix(torch.from_numpy(np.array([rx, ry, rz])).float().transpose(0, 1))
            R_noise = R_new.clone()
            last_i = 0
            for i in range(N):
                n_i = np.random.randint(last_i + window, L - (N - i) * window * 2)

                # impulse를 부드럽게 만듭니다.
                window_R = R_noise[n_i - window : n_i + window].clone()
                window_r = matrix_to_rotation_6d(window_R).numpy()
                impluse_R = R_impluse_noise[i] @ window_R[window]
                window_impluse_R = window_R.clone()
                window_impluse_R[:] = impluse_R[None]
                window_impluse_r = matrix_to_rotation_6d(window_impluse_R).numpy()

                window_new_r = noisy_impluse_interpolation(window_r, window_impluse_r)
                window_new_R = rotation_6d_to_matrix(torch.from_numpy(window_new_r).float())
                R_noise[n_i - window : n_i + window] = window_new_R
                last_i = n_i
            R_new = R_noise
            return R_new

        def add_impulse_t(T_new):
            N = np.random.randint(1, self.t_impluse_n + 1)
            tx = np.random.normal(scale=self.tx_impluse_noise, size=N)
            ty = np.random.normal(scale=self.ty_impluse_noise, size=N)
            tz = np.random.normal(scale=self.tz_impluse_noise, size=N)
            T_impluse_noise = torch.from_numpy(np.array([tx, ty, tz])).float().transpose(0, 1)
            T_noise = T_new.clone()
            last_i = 0
            for i in range(N):
                n_i = np.random.randint(last_i + window, L - N * window * 2)

                # impulse를 부드럽게 만듭니다.
                window_T = T_noise[n_i - window : n_i + window].clone()
                window_impluse_T = window_T.clone()
                window_impluse_T += T_impluse_noise[i : i + 1]
                window_impluse_T = window_impluse_T.numpy()
                window_T = window_T.numpy()

                window_new_T = noisy_impluse_interpolation(window_T, window_impluse_T)
                window_new_T = torch.from_numpy(window_new_T).float()
                T_noise[n_i - window : n_i + window] = window_new_T
                last_i = n_i
            T_new = T_noise
            return T_new

        impulse_type_prob = {
            "t": 0.2,
            "r": 0.2,
            "both": 0.1,
            "pass": 0.5,
        }
        impulse_type = np.random.choice(list(impulse_type_prob.keys()), p=list(impulse_type_prob.values()))
        if impulse_type == "t":
            # translation impulse만 적용합니다.
            T_new = add_impulse_t(T_new)
        elif impulse_type == "r":
            # rotation impulse만 적용합니다.
            R_new = add_impulse_rot(R_new)
        elif impulse_type == "both":
            # rotation과 translation impulse를 함께 적용합니다.
            R_new = add_impulse_rot(R_new)
            T_new = add_impulse_t(T_new)
        else:
            assert impulse_type == "pass"

        cam_mat_new = matrix.get_TRS(R_new, T_new)
        w2c_new = torch.inverse(cam_mat_new)
        R_new = matrix.get_rotation(w2c_new)
        T_new = matrix.get_position(w2c_new)
        tx = np.random.normal(scale=self.tx_step_noise, size=L)
        ty = np.random.normal(scale=self.ty_step_noise, size=L)
        tz = np.random.normal(scale=self.tz_step_noise, size=L)
        T_new = T_new + torch.from_numpy(np.array([tx, ty, tz])).float().transpose(0, 1)

        return R_new, T_new

    def __call__(self, w_j3d, length=120):
        """
        인자:
            w_j3d: (L, J, 3)
            length: scalar
        """
        # 입력을 확인합니다.
        self.l = length
        assert w_j3d.size(0) == self.l, "currently, only support fixed length"

        # 초기값을 설정합니다.
        w_j3d = w_j3d.clone()
        w_root = w_j3d[:, 0]  # (L, 3)

        # static camera pose를 시뮬레이션합니다.
        cfg_camera0 = {**self.cfg_create_camera, "w": self.w, "f": self.f}
        R0_w2c, t0_w2c = create_camera(w_root[0], cfg_camera0)  # (3, 3) and (3,)

        # camera 움직임을 생성합니다.
        camera_type_prob = {
            "random": 0.25,
            "track": 0.15,
            "trackrotate": 0.10,
            "trackpush": 0.05,
            "trackpull": 0.05,
            "static": 0.4,
        }
        camera_type = np.random.choice(list(camera_type_prob.keys()), p=list(camera_type_prob.values()))
        if camera_type == "random":  # 무작위 움직임과 camera noise
            R_w2c = create_rotation_move(R0_w2c, length, self.r_xyz_w_std)
            t_w2c = create_translation_move(R0_w2c, t0_w2c, length, self.t_xyz_w_std)
            R_w2c, t_w2c = self.add_stepnoise(R_w2c, t_w2c)

        elif camera_type == "track":  # 사람 추적
            R_w2c = create_rotation_move(R0_w2c, length, self.r_xyz_w_std_half)
            cam_mat = torch.inverse(transform_mat(R0_w2c, t0_w2c)).repeat(length, 1, 1)  # (F, 4, 4)
            t_w2c = self.create_translation_track(cam_mat, w_root, 0.5)
            R_w2c, t_w2c = self.add_stepnoise(R_w2c, t_w2c)

        elif camera_type == "trackrotate":  # 사람을 추적하며 회전
            cam_mat = torch.inverse(transform_mat(R0_w2c, t0_w2c)).repeat(length, 1, 1)  # (F, 4, 4)
            t_w2c = self.create_translation_track(cam_mat, w_root, 0.5)
            cam_mat = matrix.get_TRS(matrix.get_rotation(cam_mat), t_w2c)
            R_w2c = self.create_rotation_track(cam_mat, w_root, np.pi / 16, np.pi, np.pi / 16)
            R_w2c, t_w2c = self.add_stepnoise(R_w2c, t_w2c)

        elif camera_type == "trackpush":  # 사람을 추적하며 가까이 이동
            R_w2c = create_rotation_move(R0_w2c, length, self.r_xyz_w_std_half)
            # [1/tz_bias_factor, 1] * dist
            cam_mat = torch.inverse(transform_mat(R0_w2c, t0_w2c)).repeat(length, 1, 1)  # (F, 4, 4)
            t_w2c = self.create_translation_track(cam_mat, w_root, 0.5, (1.0 / (1 + self.tz_bias_factor) - 1))
            R_w2c, t_w2c = self.add_stepnoise(R_w2c, t_w2c)

        elif camera_type == "trackpull":  # 사람을 추적하며 멀리 이동
            R_w2c = create_rotation_move(R0_w2c, length, self.r_xyz_w_std_half)
            # [1, (tz_bias_factor + 1)] * dist
            cam_mat = torch.inverse(transform_mat(R0_w2c, t0_w2c)).repeat(length, 1, 1)  # (F, 4, 4)
            t_w2c = self.create_translation_track(cam_mat, w_root, 0.5, self.tz_bias_factor)
            R_w2c, t_w2c = self.add_stepnoise(R_w2c, t_w2c)

        else:
            assert camera_type == "static"
            R_w2c = R0_w2c.repeat(length, 1, 1)  # (F, 3, 3)
            t_w2c = t0_w2c.repeat(length, 1)  # (F, 3)

        # 더 적절한 camera 높이를 위해 t_w2c를 다시 계산합니다.
        # cam_w = torch.einsum("lji,lj->li", R_w2c, -t_w2c)  # (L, 3), world의 camera 중심: cam_w = - R_w2c^t_w2c @ t
        # height = cam_w[..., 1] - w_root[:, 1]
        # height = torch.clamp(height, self.height_min, self.height_max)
        # new_pos = cam_w.clone()
        # new_pos[:, 1] = w_root[:, 1] + height
        # t_w2c = torch.einsum("lij,lj->li", R_w2c, -new_pos)  # (L, 3), 새 t = -R_w2c @ cam_w

        # 더 적절한 depth와 FoV를 위해 t_w2c를 다시 계산합니다.
        c_j3d = torch.einsum("lij,lkj->lki", R_w2c, w_j3d) + t_w2c[:, None]  # (L, J, 3)
        delta = torch.zeros_like(t_w2c)  # (L, 3) 이후 t_w2c에 더합니다.
        # 사람이 camera에 너무 가까우면 z 방향으로 밀어냅니다.
        c_j3d_min = c_j3d[..., 2].min()  # scalar
        if c_j3d_min < self.tz_post_min:
            push_away = self.tz_post_min - c_j3d_min
            delta[..., 2] += push_away
            c_j3d[..., 2] += push_away
        # 사람이 FoV 안에 없으면 z 방향으로 밀어냅니다.
        c_root = c_j3d[:, 0]  # (L, 3)
        half_fov = torch.div(c_root[:, :2], c_root[:, 2:]).abs()  # (L, 2), [x/z, y/z]
        if half_fov.max() > self.half_fov_tol:
            max_idx1, max_idx2 = torch.where(torch.max(half_fov) == half_fov)
            max_idx1, max_idx2 = max_idx1[0], max_idx2[0]
            z_trg = c_root[max_idx1, max_idx2].abs() / self.half_fov_tol  # FoV 경계에 맞춘 z값
            push_away = z_trg - c_root[max_idx1, 2]
            delta[..., 2] += push_away
        t_w2c += delta

        T_w2c = transform_mat(R_w2c, t_w2c)  # (F, 4, 4)
        return T_w2c
