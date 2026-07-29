import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pytorch3d.transforms import so3_exp_map, so3_log_map
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle, matrix_to_rotation_6d
import pytorch3d.ops.knn as knn
from hmr4d.utils.pylogger import Log
from pytorch3d.transforms import euler_angles_to_matrix
import hmr4d.utils.matrix as matrix
from einops import einsum, rearrange, repeat
from hmr4d.utils.geo.quaternion import qbetween


def homo_points(points):
    """
    인자:
        points: (..., C)
    반환: 마지막에 1을 채운 (..., C+1)
    """
    return F.pad(points, [0, 1], value=1.0)


def apply_Ts_on_seq_points(points, Ts):
    """
    각 point에 대응하는 transformation matrix를 적용합니다.

    인자:
        points: (..., N, 3)
        Ts: (..., N, 4, 4)
    반환: (..., N, 3)
    """
    points = torch.torch.einsum("...ki,...i->...k", Ts[..., :3, :3], points) + Ts[..., :3, 3]
    return points


def apply_T_on_points(points, T):
    """
    인자:
        points: (..., N, 3)
        T: (..., 4, 4)
    반환: (..., N, 3)
    """
    points_T = torch.einsum("...ki,...ji->...jk", T[..., :3, :3], points) + T[..., None, :3, 3]
    return points_T


def T_transforms_points(T, points, pattern):
    """apply_T_on_points를 einsum pattern으로 직접 수행합니다.
    T: (..., 4, 4)
    points: (..., 3)
    pattern: "... c d, ... d -> ... c"
    """
    return einsum(T, homo_points(points), pattern)[..., :3]


def project_p2d(points, K=None, is_pinhole=True):
    """
    인자:
        points: (..., (N), 3)
        K: (..., 3, 3)
    반환: z 차원을 제외한 points와 같은 shape
    """
    points = points.clone()
    if is_pinhole:
        z = points[..., [-1]]
        z.masked_fill_(z.abs() < 1e-6, 1e-6)
        points_proj = points / z
    else:  # 직교 투영
        points_proj = F.pad(points[..., :2], (0, 1), value=1)

    if K is not None:
        # point 개수 N 차원의 유무를 처리합니다.
        if len(points_proj.shape) == len(K.shape):
            p2d_h = torch.einsum("...ki,...ji->...jk", K, points_proj)
        else:
            p2d_h = torch.einsum("...ki,...i->...k", K, points_proj)
    else:
        p2d_h = points_proj[..., :2]

    return p2d_h[..., :2]


def gen_uv_from_HW(H, W, device="cpu"):
    """float (H, W, 2) UV grid를 반환합니다. IJ 순서가 아닙니다."""
    grid_v, grid_u = torch.meshgrid(torch.arange(H), torch.arange(W))
    return (
        torch.stack(
            [grid_u, grid_v],
            dim=-1,
        )
        .float()
        .to(device)
    )  # (H, W, 2)


def unproject_p2d(uv, z, K):
    """pinhole camera를 가정해 2D point를 unprojection합니다.
    uv: (B, N, 2)
    z: (B, N, 1)
    K: (B, 3, 3)
    반환: (B, N, 3)
    """
    xy_atz1 = (uv - K[:, None, :2, 2]) / K[:, None, [0, 1], [0, 1]]  # (B, N, 2)
    xyz = torch.cat([xy_atz1 * z, z], dim=-1)
    return xyz


def cvt_p2d_from_i_to_c(uv, K):
    """
    인자:
        uv: (..., 2) or (..., N, 2)
        K: (..., 3, 3)
    반환: 입력 uv와 같은 shape
    """
    if len(uv.shape) == len(K.shape):
        xy = (uv - K[..., None, :2, 2]) / K[..., None, [0, 1], [0, 1]]
    else:  # N 차원이 없는 경우
        xy = (uv - K[..., :2, 2]) / K[..., [0, 1], [0, 1]]
    return xy


def cvt_to_bi01_p2d(p2d, bbx_lurb):
    """
    p2d: (..., (N), 2)
    bbx_lurb: (..., 4)
    """
    if len(p2d.shape) == len(bbx_lurb.shape) + 1:
        bbx_lurb = bbx_lurb[..., None, :]

    bbx_wh = bbx_lurb[..., 2:] - bbx_lurb[..., :2]
    bi01_p2d = (p2d - bbx_lurb[..., :2]) / bbx_wh
    return bi01_p2d


def cvt_from_bi01_p2d(bi01_p2d, bbx_lurb):
    """bbx_lurb로 bi01_p2d를 image 좌표계 p2d로 변환합니다.

    인자:
        p2d: (..., 2) or (..., N, 2)
        bbx_lurb: (..., 4)
    반환:
        p2d: 입력과 같은 shape
    """
    bbx_wh = bbx_lurb[..., 2:] - bbx_lurb[..., :2]  # (..., 2)
    if len(bi01_p2d.shape) == len(bbx_wh.shape) + 1:
        p2d = (bi01_p2d * bbx_wh.unsqueeze(-2)) + bbx_lurb[..., None, :2]
    else:
        p2d = (bi01_p2d * bbx_wh) + bbx_lurb[..., :2]
    return p2d


def cvt_p2d_from_bi01_to_c(bi01, bbxs_lurb, Ks):
    """
    인자:
        bi01: (..., (N), 2), bounding box image 안의 (0, 1) 범위 point
        bbxs_lurb: (..., 4)
        Ks: (..., 3, 3)
    반환:
        c: (..., (N), 2)
    """
    i = cvt_from_bi01_p2d(bi01, bbxs_lurb)
    c = cvt_p2d_from_i_to_c(i, Ks)
    return c


def cvt_p2d_from_pm1_to_i(p2d_pm1, bbx_xys):
    """
    인자:
        p2d_pm1: (..., (N), 2), bounding box image 안의 (-1, 1) 범위 point
        bbx_xys: (..., 3)
    반환:
        p2d: (..., (N), 2)
    """
    return bbx_xys[..., :2] + p2d_pm1 * bbx_xys[..., [2]] / 2


def uv2l_index(uv, W):
    return uv[..., 0] + uv[..., 1] * W


def l2uv_index(l, W):
    v = torch.div(l, W, rounding_mode="floor")
    u = l % W
    return torch.stack([u, v], dim=-1)


def transform_mat(R, t):
    """
    인자:
        R: rotation matrix batch의 Bx3x3 array
        t: translation vector batch의 Bx3x(1) array
    반환:
        T: Bx4x4 transformation matrix
    """
    # 좌우는 padding하지 않고 마지막 row만 추가합니다.
    if len(R.shape) > len(t.shape):
        t = t[..., None]
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=-1)


def axis_angle_to_matrix_exp_map(aa):
    """PyTorch3D의 so3_exp_map으로 axis-angle을 matrix로 변환합니다.

    인자:
        aa: (*, 3)
    반환:
        R: (*, 3, 3)
    """
    print("Use pytorch3d.transforms.axis_angle_to_matrix instead!!!")
    ori_shape = aa.shape[:-1]
    return so3_exp_map(aa.reshape(-1, 3)).reshape(*ori_shape, 3, 3)


def matrix_to_axis_angle_log_map(R):
    """PyTorch3D의 so3_log_map으로 matrix를 axis-angle로 변환합니다.

    인자:
        aa: (*, 3, 3)
    반환:
        R: (*, 3)
    """
    print("WARINING! I met singularity problem with this function, use matrix_to_axis_angle instead!")
    ori_shape = R.shape[:-2]
    return so3_log_map(R.reshape(-1, 3, 3)).reshape(*ori_shape, 3)


def matrix_to_axis_angle(R):
    """quaternion을 거쳐 rotation matrix를 axis-angle로 변환합니다.

    인자:
        aa: (*, 3, 3)
    반환:
        R: (*, 3)
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(R))


def ransac_PnP(K, pts_2d, pts_3d, err_thr=10):
    """RANSAC으로 PnP 문제를 풉니다."""
    dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")

    pts_2d = np.ascontiguousarray(pts_2d.astype(np.float64))
    pts_3d = np.ascontiguousarray(pts_3d.astype(np.float64))
    K = K.astype(np.float64)

    try:
        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, dist_coeffs, reprojectionError=err_thr, iterationsCount=10000, flags=cv2.SOLVEPNP_EPNP
        )

        rotation = cv2.Rodrigues(rvec)[0]

        pose = np.concatenate([rotation, tvec], axis=-1)
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)

        inliers = [] if inliers is None else inliers

        return pose, pose_homo, inliers
    except cv2.error:
        print("CV ERROR")
        return np.eye(4)[:3], np.eye(4), []


def ransac_PnP_batch(K_raw, pts_2d, pts_3d, err_thr=10):
    fit_R, fit_t = [], []
    for b in range(K_raw.shape[0]):
        pose, _, inliers = ransac_PnP(K_raw[b], pts_2d[b], pts_3d[b], err_thr=err_thr)
        fit_R.append(pose[:3, :3])
        fit_t.append(pose[:3, 3])
    fit_R = np.stack(fit_R, axis=0)
    fit_t = np.stack(fit_t, axis=0)
    return fit_R, fit_t


def triangulate_point(Ts_w2c, c_p2d, **kwargs):
    from hmr4d.utils.geo.triangulation import triangulate_persp

    print("Deprecated, please import from hmr4d.utils.geo.triangulation")
    return triangulate_persp(Ts_w2c, c_p2d, **kwargs)


def triangulate_point_ortho(Ts_w2c, c_p2d, **kwargs):
    from hmr4d.utils.geo.triangulation import triangulate_ortho

    print("Deprecated, please import from hmr4d.utils.geo.triangulation")
    return triangulate_ortho(Ts_w2c, c_p2d, **kwargs)


def get_nearby_points(points, query_verts, padding=0.0, p=1):
    """
    points: (S, 3)
    query_verts: (V, 3)
    """
    if p == 1:
        max_xyz = query_verts.max(0)[0] + padding
        min_xyz = query_verts.min(0)[0] - padding
        idx = (((points - min_xyz) > 0).all(dim=-1) * ((points - max_xyz) < 0).all(dim=-1)).nonzero().squeeze(-1)
        nearby_points = points[idx]
    elif p == 2:
        squared_dist, _, _ = knn.knn_points(points[None], query_verts[None], K=1, return_nn=False)
        mask = squared_dist[0, :, 0] < padding**2  # (S,)
        nearby_points = points[mask]

    return nearby_points


def unproj_bbx_to_fst(bbx_lurb, K, near_z=0.5, far_z=12.5):
    B = bbx_lurb.size(0)
    uv = bbx_lurb[:, [[0, 1], [2, 1], [2, 3], [0, 3], [0, 1], [2, 1], [2, 3], [0, 3]]]
    if isinstance(near_z, float):
        z = uv.new([near_z] * 4 + [far_z] * 4).reshape(1, 8, 1).repeat(B, 1, 1)
    else:
        z = torch.cat([near_z[:, None, None].repeat(1, 4, 1), far_z[:, None, None].repeat(1, 4, 1)], dim=1)
    c_frustum_points = unproject_p2d(uv, z, K)  # (B, 8, 3)
    return c_frustum_points


def convert_bbx_xys_to_lurb(bbx_xys):
    """
    변환: bbx_xys (..., 3) -> bbx_lurb (..., 4)
    """
    size = bbx_xys[..., 2:]
    center = bbx_xys[..., :2]
    lurb = torch.cat([center - size / 2, center + size / 2], dim=-1)
    return lurb


def convert_lurb_to_bbx_xys(bbx_lurb):
    """
    변환: bbx_lurb (..., 4) -> 정사각형 bbx_xys (..., 3)
    """
    size = (bbx_lurb[..., 2:] - bbx_lurb[..., :2]).max(-1, keepdim=True)[0]
    center = (bbx_lurb[..., :2] + bbx_lurb[..., 2:]) / 2
    return torch.cat([center, size], dim=-1)


# ================== AZ/AY 좌표계 변환 ================== #


def compute_T_ayf2az(joints, inverse=False):
    """
    인자:
        joints: (B, J, 3), 시작 frame의 az 좌표계
    반환:
        inverse가 False이면:
           T_af2az: (B, 4, 4)
        그 외:
            T_az2af: (B, 4, 4)
    """

    t_ayf2az = joints[:, 0, :].detach().clone()
    t_ayf2az[:, 2] = 0  # z축은 수정하지 않습니다.

    RL_xy_h = joints[:, 1, [0, 1]] - joints[:, 2, [0, 1]]  # (B, 2), 왼쪽을 향하는 hip vector
    RL_xy_s = joints[:, 16, [0, 1]] - joints[:, 17, [0, 1]]  # (B, 2), 왼쪽을 향하는 shoulder vector
    RL_xy = RL_xy_h + RL_xy_s
    I_mask = RL_xy.pow(2).sum(-1) < 1e-4  # 정면 방향을 결정할 수 없으면 회전하지 않습니다.
    if I_mask.sum() > 0:
        Log.warn("{} samples can't decide the face direction".format(I_mask.sum()))
    x_dir = F.pad(F.normalize(RL_xy, 2, -1), (0, 1), value=0)  # (B, 3)
    y_dir = torch.zeros_like(x_dir)
    y_dir[..., 2] = 1
    z_dir = torch.cross(x_dir, y_dir, dim=-1)
    R_ayf2az = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # (B, 3, 3)
    R_ayf2az[I_mask] = torch.eye(3).to(R_ayf2az)

    if inverse:
        R_az2ayf = R_ayf2az.transpose(1, 2)  # (B, 3, 3)
        t_az2ayf = -einsum(R_ayf2az, t_ayf2az, "b i j , b i -> b j")  # (B, 3)
        return transform_mat(R_az2ayf, t_az2ayf)
    else:
        return transform_mat(R_ayf2az, t_ayf2az)


def compute_T_ayfz2ay(joints, inverse=False):
    """
    인자:
        joints: (B, J, 3), 시작 frame의 ay 좌표계
    반환:
        inverse가 False이면:
            T_ayfz2ay: (B, 4, 4)
        그 외:
            T_ay2ayfz: (B, 4, 4)
    """
    t_ayfz2ay = joints[:, 0, :].detach().clone()
    t_ayfz2ay[:, 1] = 0  # y축은 수정하지 않습니다.

    RL_xz_h = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]  # (B, 2), 왼쪽을 향하는 hip vector
    RL_xz_s = joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]  # (B, 2), 왼쪽을 향하는 shoulder vector
    RL_xz = RL_xz_h + RL_xz_s
    I_mask = RL_xz.pow(2).sum(-1) < 1e-4  # 정면 방향을 결정할 수 없으면 회전하지 않습니다.
    if I_mask.sum() > 0:
        Log.warn("{} samples can't decide the face direction".format(I_mask.sum()))

    x_dir = torch.zeros_like(t_ayfz2ay)  # (B, 3)
    x_dir[:, [0, 2]] = F.normalize(RL_xz, 2, -1)
    y_dir = torch.zeros_like(x_dir)
    y_dir[..., 1] = 1  # (B, 3)
    z_dir = torch.cross(x_dir, y_dir, dim=-1)
    R_ayfz2ay = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # (B, 3, 3)
    R_ayfz2ay[I_mask] = torch.eye(3).to(R_ayfz2ay)

    if inverse:
        R_ay2ayfz = R_ayfz2ay.transpose(1, 2)
        t_ay2ayfz = -einsum(R_ayfz2ay, t_ayfz2ay, "b i j , b i -> b j")
        return transform_mat(R_ay2ayfz, t_ay2ayfz)
    else:
        return transform_mat(R_ayfz2ay, t_ayfz2ay)


def compute_T_ay2ayrot(joints):
    """
    인자:
        joints: (B, J, 3), 시작 frame의 ay 좌표계
    반환:
        T_ay2ayrot: (B, 4, 4)
    """
    t_ayrot2ay = joints[:, 0, :].detach().clone()
    t_ayrot2ay[:, 1] = 0  # y축은 수정하지 않습니다.

    B = joints.shape[0]
    euler_angle = torch.zeros((B, 3), device=joints.device)
    yrot_angle = torch.rand((B,), device=joints.device) * 2 * torch.pi
    euler_angle[:, 0] = yrot_angle
    R_ay2ayrot = euler_angles_to_matrix(euler_angle, "YXZ")  # (B, 3, 3)

    R_ayrot2ay = R_ay2ayrot.transpose(1, 2)
    t_ay2ayrot = -einsum(R_ayrot2ay, t_ayrot2ay, "b i j , b i -> b j")
    return transform_mat(R_ay2ayrot, t_ay2ayrot)


def compute_root_quaternion_ay(joints):
    """
    인자:
        joints: (B, J, 3), 시작 frame의 ay 좌표계
    반환:
        root_quat: z축에서 fz 방향으로의 (B, 4) quaternion
    """
    joints_shape = joints.shape
    joints = joints.reshape((-1,) + joints_shape[-2:])
    t_ayfz2ay = joints[:, 0, :].detach().clone()
    t_ayfz2ay[:, 1] = 0  # y축은 수정하지 않습니다.

    RL_xz_h = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]  # (B, 2), 왼쪽을 향하는 hip vector
    RL_xz_s = joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]  # (B, 2), 왼쪽을 향하는 shoulder vector
    RL_xz = RL_xz_h + RL_xz_s
    I_mask = RL_xz.pow(2).sum(-1) < 1e-4  # 정면 방향을 결정할 수 없으면 회전하지 않습니다.
    if I_mask.sum() > 0:
        Log.warn("{} samples can't decide the face direction".format(I_mask.sum()))

    x_dir = torch.zeros_like(t_ayfz2ay)  # (B, 3)
    x_dir[:, [0, 2]] = F.normalize(RL_xz, 2, -1)
    y_dir = torch.zeros_like(x_dir)
    y_dir[..., 1] = 1  # (B, 3)
    z_dir = torch.cross(x_dir, y_dir, dim=-1)

    z_dir[..., 2] += 1e-9
    pos_z_vec = torch.tensor([0, 0, 1]).to(joints.device).float()  # (3,)
    root_quat = qbetween(pos_z_vec[None], z_dir)  # (B, 4)
    root_quat = root_quat.reshape(joints_shape[:-2] + (4,))
    return root_quat


# ================== 두 feature 집합 사이의 transformation ================== #


def similarity_transform_batch(S1, S2):
    """
    orthogonal Procrustes 문제를 푸는 similarity transformation (sR, t)를 계산합니다.

    인자:
        S1, S2: (*, L, 3)
    """
    assert S1.shape == S2.shape
    S_shape = S1.shape
    S1 = S1.reshape(-1, *S_shape[-2:])
    S2 = S2.reshape(-1, *S_shape[-2:])

    S1 = S1.transpose(-2, -1)
    S2 = S2.transpose(-2, -1)

    # --- 아래 코드는 WHAM에서 가져왔습니다. ---
    # 1. 평균을 제거합니다.
    mu1 = S1.mean(axis=-1, keepdims=True)  # N축을 따라 평균을 구합니다. S1(B, 3, N)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. scale 계산에 사용할 X1의 variance를 구합니다.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. X1과 X2의 outer product를 구합니다.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. trace(R'K)를 최대화하는 해는 R=U*V'이며 U, V는 K의 singular vector입니다.
    U, s, V = torch.svd(K)

    # det(R)=1이 되도록 R의 orientation을 보정하는 Z를 만듭니다.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # R을 구성합니다.
    R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

    # 5. scale을 복원합니다.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. translation을 복원합니다.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # -------
    # 원래 batch shape로 복원합니다.
    # sR = scale[:, None, None] * R
    # sR = sR.reshape(*S_shape[:-2], 3, 3)
    scale = scale.reshape(*S_shape[:-2], 1, 1)
    R = R.reshape(*S_shape[:-2], 3, 3)
    t = t.reshape(*S_shape[:-2], 3, 1)

    return (scale, R), t


def kabsch_algorithm_batch(X1, X2):
    """
    rigid transformation (R, t)를 계산합니다.

    인자:
        X1, X2: (*, L, 3)
    """
    assert X1.shape == X2.shape
    X_shape = X1.shape
    X1 = X1.reshape(-1, *X_shape[-2:])
    X2 = X2.reshape(-1, *X_shape[-2:])

    # 1. 중심점을 계산합니다.
    centroid_X1 = torch.mean(X1, dim=-2, keepdim=True)
    centroid_X2 = torch.mean(X2, dim=-2, keepdim=True)

    # 2. 중심을 제거합니다.
    X1_centered = X1 - centroid_X1
    X2_centered = X2 - centroid_X2

    # 3. covariance matrix를 계산합니다.
    H = torch.matmul(X1_centered.transpose(-2, -1), X2_centered)

    # 4. singular value decomposition을 수행합니다.
    U, S, Vt = torch.linalg.svd(H)

    # 5. rotation matrix를 계산합니다.
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))

    # reflection matrix를 보정합니다.
    d = (torch.det(R) < 0).unsqueeze(-1).unsqueeze(-1)
    Vt = torch.where(d, -Vt, Vt)
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))

    # 6. translation vector를 계산합니다.
    t = centroid_X2.transpose(-2, -1) - torch.matmul(R, centroid_X1.transpose(-2, -1))

    # -------
    # 원래 batch shape로 복원합니다.
    R = R.reshape(*X_shape[:-2], 3, 3)
    t = t.reshape(*X_shape[:-2], 3, 1)

    return R, t


# ===== WHAM camera 각속도 ===== #


def compute_cam_angvel(R_w2c, padding_last=True):
    """
    R_w2c : (F, 3, 3)
    """
    # R @ R0 = R1, so R = R1 @ R0^T
    cam_angvel = matrix_to_rotation_6d(R_w2c[1:] @ R_w2c[:-1].transpose(-1, -2))  # (F-1, 6)
    # cam_angvel = (cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]])) * FPS
    assert padding_last
    cam_angvel = torch.cat([cam_angvel, cam_angvel[-1:]], dim=0)  # (F, 6)
    return cam_angvel.float()


def ransac_gravity_vec(xyz, num_iterations=100, threshold=0.05, verbose=False):
    # xyz: (L, 3)
    N = xyz.shape[0]
    max_inliers = []
    best_model = None
    norms = xyz.norm(dim=-1)  # (L,)

    for _ in range(num_iterations):
        # sample 하나를 무작위로 선택합니다.
        sample_index = np.random.randint(N)
        sample = xyz[sample_index]  # (3,)

        # 모든 point와 sample point 사이의 각도 차이를 계산합니다.
        dot_product = (xyz * sample).sum(dim=-1)  # (L,)
        angles = dot_product / norms * norms[sample_index]  # (L,)
        angles = torch.clamp(angles, -1, 1)  # 수치 오차로 인한 이상값을 방지합니다.
        angles = torch.acos(angles)

        # inlier를 결정합니다.
        inliers = xyz[angles < threshold]

        if len(inliers) > len(max_inliers):
            max_inliers = inliers
            best_model = sample
        if len(max_inliers) == N:
            break
    if verbose:
        print(f"Inliers: {len(max_inliers)} / {N}")
    result = max_inliers.mean(dim=0)

    return result, max_inliers


def sequence_best_cammat(w_j3d, c_j3d, cam_rot):
    # static camera를 가정해 sequence 전체에서 가장 좋은 camera 추정값을 구합니다.
    # w_j3d: (L, J, 3)
    # c_j3d: (L, J, 3)
    # cam_rot: (L, 3, 3)

    L, J, _ = w_j3d.shape

    root_in_w = w_j3d[:, 0]  # (L, 3)
    root_in_c = c_j3d[:, 0]  # (L, 3)
    cam_mat = matrix.get_TRS(cam_rot, root_in_w)  # (L, 4, 4)
    cam_pos = matrix.get_position_from(-root_in_c[:, None], cam_mat)[:, 0]  # (L, 3)
    cam_mat = matrix.set_position(cam_mat, cam_pos)  # (L, 4, 4)

    w_j3d_expand = w_j3d[None].expand(L, -1, -1, -1)  # (L, L, J, 3)
    w_j3d_expand = w_j3d_expand.reshape(L, -1, 3)  # (L, L*J, 3)

    # reprojection error를 계산합니다.
    w_j3d_expand_in_c = matrix.get_relative_position_to(w_j3d_expand, cam_mat)  # (L, L*J, 3)
    w_j2d_expand_in_c = project_p2d(w_j3d_expand_in_c)  # (L, L*J, 2)
    w_j2d_expand_in_c = w_j2d_expand_in_c.reshape(L, L, J, 2)  # (L, L, J, 2)
    c_j2d = project_p2d(c_j3d)  # (L, J, 2)
    error = w_j2d_expand_in_c - c_j2d[None]  # (L, L, J, 2)
    error = error.norm(dim=-1).mean(dim=-1)  # (L, L)
    error = error.mean(dim=-1)  # (L,)
    ind = error.argmin()
    return cam_mat[ind], ind


def get_sequence_cammat(w_j3d, c_j3d, cam_rot):
    # w_j3d: (L, J, 3)
    # c_j3d: (L, J, 3)
    # cam_rot: (L, 3, 3)

    L, J, _ = w_j3d.shape

    root_in_w = w_j3d[:, 0]  # (L, 3)
    root_in_c = c_j3d[:, 0]  # (L, 3)
    cam_mat = matrix.get_TRS(cam_rot, root_in_w)  # (L, 4, 4)
    cam_pos = matrix.get_position_from(-root_in_c[:, None], cam_mat)[:, 0]  # (L, 3)
    cam_mat = matrix.set_position(cam_mat, cam_pos)  # (L, 4, 4)
    return cam_mat


def ransac_vec(vel, min_multiply=20, verbose=False):
    # xyz: (L, 3)
    # velocity outlier를 제거합니다.
    N = vel.shape[0]
    vel_1 = vel[None].expand(N, -1, -1)  # (L, L, 3)
    vel_2 = vel[:, None].expand(-1, N, -1)  # (L, L, 3)
    dist_mat = (vel_1 - vel_2).norm(dim=-1)  # (L, L)
    big_identity = torch.eye(N, device=vel.device) * 1e6
    dist_mat_ = dist_mat + big_identity
    threshold = dist_mat_.min() * min_multiply
    inner_mask = dist_mat < threshold  # (L, L)
    inner_num = inner_mask.sum(dim=-1)  # (L, )
    ind = inner_num.argmax()
    result = vel[inner_mask[ind]].mean(dim=0)  # (3,)
    if verbose:
        print(inner_mask[ind].sum().item())

    return result, inner_mask[ind]
