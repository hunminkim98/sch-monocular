import torch
import numpy as np
from pytorch3d.transforms import matrix_to_quaternion, matrix_to_axis_angle
from hmr4d.utils.geo.local_to_global import batch_transfer_multi_rotation


@torch.no_grad()
def compute_camcoord_metrics(batch, pelvis_idxs=[1, 2], fps=30, mask=None):
    """
    batch (dict): {
        "pred_j3d": (..., J, 3) tensor
        "target_j3d":
        "pred_verts":
        "target_verts":
        "gt_incam_pose":
        "pred_incam_pose":
        "parents":
    }
    """
    # 모든 데이터는 camera 좌표계에 있습니다.
    pred_j3d = batch["pred_j3d"].cpu()  # (..., J, 3)
    target_j3d = batch["target_j3d"].cpu()
    pred_verts = batch["pred_verts"].cpu()
    target_verts = batch["target_verts"].cpu()

    if mask is not None:
        mask = mask.cpu()
        pred_j3d = pred_j3d[mask].clone()
        target_j3d = target_j3d[mask].clone()
        pred_verts = pred_verts[mask].clone()
        target_verts = target_verts[mask].clone()
    assert "mask" not in batch

    # pelvis를 기준으로 정렬합니다.
    pred_j3d, target_j3d, pred_verts, target_verts = batch_align_by_pelvis(
        [pred_j3d, target_j3d, pred_verts, target_verts], pelvis_idxs=pelvis_idxs
    )

    # metric 계산
    m2mm = 1000
    S1_hat = batch_compute_similarity_transform_torch(pred_j3d, target_j3d)
    pa_mpjpe = compute_jpe(S1_hat, target_j3d) * m2mm
    # 발 metric
    n_mpjpef = compute_3d_foot_metrics(pred_verts, target_verts)
    # 발목 rotation 오차
    pred_incam_full_body_pose = batch["pred_incam_pose"].cpu()
    gt_incam_full_body_pose = batch["gt_incam_pose"].cpu()
    # parent-relative 대신 global joint rotation matrix를 계산합니다.
    joint_indices = [7, 8]  # 발목 joint index
    pred_global_rotations = batch_transfer_multi_rotation(
        batch["parents"],
        pred_incam_full_body_pose,  # global orientation 포함
        joint_indices=joint_indices,  # 변환할 joint index
        transfer_type="l2g",  # local에서 global로 변환
        result_format="rotmat",
    )
    gt_global_rotations = batch_transfer_multi_rotation(
        batch["parents"],
        gt_incam_full_body_pose,  # global orientation 포함
        joint_indices=joint_indices,  # 변환할 joint index
        transfer_type="l2g",  # local에서 global로 변환
        result_format="rotmat",
    )
    angle_errors = joint_angle_error(pred_global_rotations, gt_global_rotations).numpy()
    ankles_angle_err = angle_errors.mean(axis=1)
    camcoord_metrics = {
        "pa_mpjpe": pa_mpjpe,
        "n_mpjpef": n_mpjpef * m2mm,  # 발마다 중심과 scale을 맞춘 뒤의 foot MPJPE
        "ajae": ankles_angle_err,
    }

    return camcoord_metrics


@torch.no_grad()
def compute_global_metrics(batch, mask=None):
    """WHAM 방식을 따르며 입력에서 invalid frame은 이미 제외되었다고 가정합니다.

    인자:
        batch (dict): {
            "pred_j3d_glob": (F, J, 3) tensor
            "target_j3d_glob":
            "pred_verts_glob":
            "target_verts_glob":
        }
    반환:
        global_metrics (dict): {
            "wa2_mpjpe": (F, ) numpy array
            "waa_mpjpe":
            "rte":
            "jitter":
            "fs":
        }
    """
    # 모든 데이터는 global 좌표계에 있습니다.
    pred_j3d_glob = batch["pred_j3d_glob"].cpu()  # (..., J, 3)
    target_j3d_glob = batch["target_j3d_glob"].cpu()
    pred_verts_glob = batch["pred_verts_glob"].cpu()
    target_verts_glob = batch["target_verts_glob"].cpu()
    if mask is not None:
        mask = mask.cpu()
        pred_j3d_glob = pred_j3d_glob[mask].clone()
        target_j3d_glob = target_j3d_glob[mask].clone()
        pred_verts_glob = pred_verts_glob[mask].clone()
        target_verts_glob = target_verts_glob[mask].clone()
    assert "mask" not in batch

    seq_length = pred_j3d_glob.shape[0]

    # chunk 단위로 비교합니다.
    chunk_length = 100
    wa2_mpjpe, waa_mpjpe = [], []
    for start in range(0, seq_length, chunk_length):
        end = min(seq_length, start + chunk_length)

        target_j3d = target_j3d_glob[start:end].clone().cpu()
        pred_j3d = pred_j3d_glob[start:end].clone().cpu()

        w_j3d = first_align_joints(target_j3d, pred_j3d)
        wa_j3d = global_align_joints(target_j3d, pred_j3d)

        if False:
            from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines

            wis3d = make_wis3d(name="debug-metric_utils")
            add_motion_as_lines(target_j3d, wis3d, name="target_j3d")
            add_motion_as_lines(pred_j3d, wis3d, name="pred_j3d")
            add_motion_as_lines(w_j3d, wis3d, name="pred_w2_j3d")
            add_motion_as_lines(wa_j3d, wis3d, name="pred_wa_j3d")

        wa2_mpjpe.append(compute_jpe(target_j3d, w_j3d))
        waa_mpjpe.append(compute_jpe(target_j3d, wa_j3d))

    # 기본 metric
    m2mm = 1000
    wa2_mpjpe = np.concatenate(wa2_mpjpe) * m2mm
    waa_mpjpe = np.concatenate(waa_mpjpe) * m2mm

    # 추가 metric
    rte = compute_rte(target_j3d_glob[:, 0].cpu(), pred_j3d_glob[:, 0].cpu()) * 1e2
    jitter = compute_jitter(pred_j3d_glob, fps=30)
    foot_sliding = compute_foot_sliding(target_verts_glob, pred_verts_glob) * m2mm

    global_metrics = {
        "wa2_mpjpe": wa2_mpjpe,
        "waa_mpjpe": waa_mpjpe,
        "rte": rte,
        "jitter": jitter,
        "fs": foot_sliding,
    }
    return global_metrics


@torch.no_grad()
def compute_camcoord_perjoint_metrics(batch, pelvis_idxs=[1, 2]):
    """
    인자:
        batch (dict): {
            "pred_j3d": (..., J, 3) tensor
            "target_j3d":
        }
    반환:
        cam_coord_metrics (dict): {
            "pa_mpjpe": (..., ) numpy array
            "mpjpe":
            "pve":
            "accel":
        }
    """
    # 모든 데이터는 camera 좌표계에 있습니다.
    pred_j3d = batch["pred_j3d"].cpu()  # (..., J, 3)
    target_j3d = batch["target_j3d"].cpu()
    pred_verts = batch["pred_verts"].cpu()
    target_verts = batch["target_verts"].cpu()

    # pelvis를 기준으로 정렬합니다.
    pred_j3d, target_j3d, pred_verts, target_verts = batch_align_by_pelvis(
        [pred_j3d, target_j3d, pred_verts, target_verts], pelvis_idxs=pelvis_idxs
    )
    # metric 계산
    m2mm = 1000
    perjoint_mpjpe = compute_perjoint_jpe(pred_j3d, target_j3d) * m2mm

    camcoord_perjoint_metrics = {
        "mpjpe": perjoint_mpjpe,
    }
    return camcoord_perjoint_metrics


# ===== 유틸리티 =====


def compute_jpe(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).mean(dim=-1).numpy()


def compute_perjoint_jpe(S1, S2):
    return torch.sqrt(((S1 - S2) ** 2).sum(dim=-1)).numpy()


def compute_3d_foot_metrics(pred_verts, target_verts):
    # LBigToe, LSmallToe, LHeel, RBigToe, RSmallToe, RHeel
    smpl_foot_indices = [3216, 3226, 3387, 6617, 6624, 6787]
    # 발 keypoint를 선택합니다.
    pred_feet = pred_verts[:, smpl_foot_indices]  # (N, 6, 3)
    target_feet = target_verts[:, smpl_foot_indices]  # (N, 6, 3)
    # 발 중심 기준 MPJPE
    # 각 발의 중심을 맞춥니다.
    # 예측값 중심 정렬
    center_left_foot = pred_feet[:, :3].mean(dim=1, keepdims=True)
    center_right_foot = pred_feet[:, 3:].mean(dim=1, keepdims=True)
    c_pred_feet = pred_feet.clone()
    c_pred_feet[:, :3] = pred_feet[:, :3] - center_left_foot
    c_pred_feet[:, 3:] = pred_feet[:, 3:] - center_right_foot
    # GT 중심 정렬
    center_left_foot = target_feet[:, :3].mean(dim=1, keepdims=True)
    center_right_foot = target_feet[:, 3:].mean(dim=1, keepdims=True)
    c_target_feet = target_feet.clone()
    c_target_feet[:, :3] = target_feet[:, :3] - center_left_foot
    c_target_feet[:, 3:] = target_feet[:, 3:] - center_right_foot

    # scale 정렬
    nmpjpe_left = n_mpjpe(c_pred_feet[:, :3], c_target_feet[:, :3]).numpy()
    nmpjpe_right = n_mpjpe(c_pred_feet[:, 3:], c_target_feet[:, 3:]).numpy()
    n_mpjpef = (nmpjpe_left + nmpjpe_right) / 2
    return n_mpjpef


def joint_angle_error(r1, r2):
    # https://github.com/miraymen/3dpw-eval/blob/master/evaluate.py
    """
    두 입력 matrix 사이의 geodesic distance를 계산합니다.

    :param pred_mat: 예측 rotation matrix. Shape: (..., 3, 3)
    :param gt_mat: GT rotation matrix. Shape: (..., 3, 3)
    :return: 입력 matrix 사이의 평균 geodesic distance
    """
    ndims = r1.ndim
    # GT matrix의 마지막 두 차원을 transpose합니다.
    r2t = torch.transpose(r2, ndims-2, ndims-1)

    # R1 * R2.T를 계산합니다. 예측과 target이 같으면 identity matrix가 됩니다.
    r = torch.matmul(r1, r2t)

    # rotation matrix를 axis-angle 표현으로 바꾸고 angle을 구합니다.
    axis_angles = matrix_to_axis_angle(r)
    angles = torch.linalg.norm(axis_angles, dim=-1)
    return torch.rad2deg(angles)


def joint_angle_error2(pred_mat, gt_mat):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    """
    두 입력 matrix 사이의 geodesic distance를 계산합니다.

    :param pred_mat: 예측 rotation matrix. Shape: (Seq, J, 3, 3)
    :param gt_mat: GT rotation matrix. Shape: (Seq, J, 3, 3)
    :return: 입력 matrix 사이의 평균 geodesic distance
    """

    # rotation matrix를 quaternion으로 변환합니다.
    q1 = matrix_to_quaternion(pred_mat).cpu().numpy()
    q2 = matrix_to_quaternion(gt_mat).cpu().numpy()
    q_diff = np.sum(q1 * q2.conjugate(), axis=-1)
    theta = 2 * np.arccos(np.abs(q_diff))
    return np.rad2deg(theta)


def batch_align_by_pelvis(data_list, pelvis_idxs=[1, 2]):
    """
    데이터가 [pred_j3d, target_j3d, pred_verts, target_verts] 순서로 주어졌다고
    가정합니다. 각 데이터의 shape는 (frames, num_points, 3)입니다.
    pelvis는 한두 개의 joint index로 나타내며, 모든 데이터를 해당 pelvis 위치에
    맞춰 정렬합니다.
    """

    pred_j3d, target_j3d, pred_verts, target_verts = data_list

    pred_pelvis = pred_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()
    target_pelvis = target_j3d[:, pelvis_idxs].mean(dim=1, keepdims=True).clone()

    # pelvis를 기준으로 정렬합니다.
    pred_j3d = pred_j3d - pred_pelvis
    target_j3d = target_j3d - target_pelvis
    pred_verts = pred_verts - pred_pelvis
    target_verts = target_verts - target_pelvis

    return (pred_j3d, target_j3d, pred_verts, target_verts)


def batch_compute_similarity_transform_torch(S1, S2):
    """
    3D point 집합 S1(3 x N)을 S2에 가장 가깝게 만드는 similarity
    transformation (sR, t)를 계산합니다. R은 3x3 rotation matrix,
    t는 3x1 translation, s는 scale이며 orthogonal Procrustes 문제를 풉니다.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert S2.shape[1] == S1.shape[1]

    # 1. 평균을 제거합니다.
    mu1 = S1.mean(axis=-1, keepdims=True)
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

    # 7. 정렬된 결과를 계산합니다.
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    return S1_hat


def compute_error_accel(joints_gt, joints_pred, valid_mask=None, fps=None):
    """
    [i-1, i, i+1]을 사용해 frame_i의 acceleration을 계산합니다. 오차식은 다음과 같습니다.
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    보이지 않는 각 frame에서는 acceleration 오차의 세 항목(-1, 0, +1)을 제외합니다.

    인자:
        joints_gt : (F, J, 3)
        joints_pred : (F, J, 3)
        valid_mask : (F)
    반환:
        valid_mask가 None이면 error_accel은 (F-2), 아니면 (F')이며 F' <= F-2입니다.
    """
    # (F, J, 3)에서 joint별 (F-2) 오차를 구합니다.
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]
    normed = np.linalg.norm(accel_pred - accel_gt, axis=-1).mean(axis=-1)
    if fps is not None:
        normed = normed * fps**2

    if valid_mask is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(valid_mask)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)
        if new_vis.sum() == 0:
            print("Warning!!! no valid acceleration error to compute.")

    return normed[new_vis]


def compute_rte(target_trans, pred_trans):
    # global alignment를 계산합니다.
    _, rot, trans = align_pcl(target_trans[None, :], pred_trans[None, :], fixed_scale=True)
    pred_trans_hat = (torch.einsum("tij,tnj->tni", rot, pred_trans[None, :]) + trans[None, :])[0]

    # GT trajectory의 전체 displacement를 계산합니다.
    disps, disp = [], 0
    for p1, p2 in zip(target_trans, target_trans[1:]):
        delta = (p2 - p1).norm(2, dim=-1)
        disp += delta
        disps.append(disp)

    # absolute root translation error(RTE)를 계산합니다.
    rte = torch.norm(target_trans - pred_trans_hat, 2, dim=-1)

    # displacement로 정규화합니다.
    return (rte / disp).numpy()


def compute_jitter(joints, fps=30):
    """motion의 jitter를 계산합니다.

    인자:
        joints (N, J, 3).
        fps (float).
    반환:
        jitter (N-3).
    """
    pred_jitter = torch.norm(
        (joints[3:] - 3 * joints[2:-1] + 3 * joints[1:-2] - joints[:-3]) * (fps**3),
        dim=2,
    ).mean(dim=-1)

    return pred_jitter.cpu().numpy() / 10.0


def compute_foot_sliding(target_verts, pred_verts, thr=1e-2):
    """foot sliding 오차를 계산합니다.

    ground contact label은 frame당 1cm threshold로 계산합니다.

    인자:
        target_verts (N, 6890, 3).
        pred_verts (N, 6890, 3).
    반환:
        contact 상태인 N개 frame의 error
    """
    assert target_verts.shape == pred_verts.shape
    assert target_verts.shape[-2] == 6890

    # 발 vertex index
    foot_idxs = [3216, 3387, 6617, 6787]

    # contact label을 계산합니다.
    foot_loc = target_verts[:, foot_idxs]
    foot_disp = (foot_loc[1:] - foot_loc[:-1]).norm(2, dim=-1)
    contact = foot_disp[:] < thr

    pred_feet_loc = pred_verts[:, foot_idxs]
    pred_disp = (pred_feet_loc[1:] - pred_feet_loc[:-1]).norm(2, dim=-1)

    error = pred_disp[contact]

    return error.cpu().numpy()


def convert_joints22_to_24(joints22, ratio2220=0.3438, ratio2321=0.3345):
    joints24 = torch.zeros(*joints22.shape[:-2], 24, 3).to(joints22.device)
    joints24[..., :22, :] = joints22
    joints24[..., 22, :] = joints22[..., 20, :] + ratio2220 * (joints22[..., 20, :] - joints22[..., 18, :])
    joints24[..., 23, :] = joints22[..., 21, :] + ratio2321 * (joints22[..., 21, :] - joints22[..., 19, :])
    return joints24


def align_pcl(Y, X, weight=None, fixed_scale=False):
    """Umeyama 방식의 similarity transformation으로 X를 Y에 정렬합니다.

    ``X' = s * R * X + t``가 Y에 정렬됩니다.

    :param Y (*, N, 3) 첫 번째 trajectory
    :param X (*, N, 3) 두 번째 trajectory
    :param weight (*, N, 1) 유효한 correspondence의 선택적 가중치
    :returns s (*, 1), R (*, 3, 3), t (*, 3)
    """
    *dims, N, _ = Y.shape
    N = torch.ones(*dims, 1, 1) * N

    if weight is not None:
        Y = Y * weight
        X = X * weight
        N = weight.sum(dim=-2, keepdim=True)  # (*, 1, 1)

    # 평균을 제거합니다.
    my = Y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]  # (*, N, 3)
    x0 = X - mx[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    # correlation을 계산합니다.
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = torch.eye(3).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fixed_scale:
        s = torch.ones(*dims, 1, device=Y.device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
        s = torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) / var[..., 0]  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    return s, R, t


def global_align_joints(gt_joints, pred_joints):
    """
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    s_glob, R_glob, t_glob = align_pcl(gt_joints.reshape(-1, 3), pred_joints.reshape(-1, 3))
    pred_glob = s_glob * torch.einsum("ij,tnj->tni", R_glob, pred_joints) + t_glob[None, None]
    return pred_glob


def first_align_joints(gt_joints, pred_joints):
    """
    처음 두 frame을 기준으로 정렬합니다.
    :param gt_joints (T, J, 3)
    :param pred_joints (T, J, 3)
    """
    # (1, 1), (1, 3, 3), (1, 3)
    s_first, R_first, t_first = align_pcl(gt_joints[:2].reshape(1, -1, 3), pred_joints[:2].reshape(1, -1, 3))
    pred_first = s_first * torch.einsum("tij,tnj->tni", R_first, pred_joints) + t_first[:, None]
    return pred_first


def rearrange_by_mask(x, mask):
    """
    x (L, *)
    mask (M,), M >= L
    """
    M = mask.size(0)
    L = x.size(0)
    if M == L:
        return x
    assert M > L
    assert mask.sum() == L
    x_rearranged = torch.zeros((M, *x.size()[1:]), dtype=x.dtype, device=x.device)
    x_rearranged[mask] = x
    return x_rearranged


def as_np_array(d):
    if isinstance(d, torch.Tensor):
        return d.cpu().numpy()
    elif isinstance(d, np.ndarray):
        return d
    else:
        return np.array(d)


def n_mpjpe(pred, gt, return_mean=True):
    """
    least squares 방식으로 scale을 정규화한 뒤 지정한 criterion을 적용합니다.
    이 과정이 MPJPE 자체를 최소화하는 것은 아닙니다.

    p를 예측, g를 GT, s를 구할 scale이라고 할 때 다음 식을 최소화하는 s를 찾습니다.
      min_x ||s p - q||^2
    = min_x (s p - q) * (s p - q)
    미분값을 0으로 두면 최솟값은 다음과 같습니다.
        2 p * (s p -g) = 0
    <=> s p * p - p * g = 0
    <=> s = p * g / (p * p)
    여기서 *는 dot product입니다.
    """
    # https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/python/losses/poses.py
    # (bs, joints, 3) torch tensors

    pred_orig = pred.clone()
    gt_orig = gt.clone()

    bs = pred.shape[0]
    pred_vec = pred.reshape(bs, -1)
    gt_vec = gt.reshape(bs, -1)
    dot_pose_pose = torch.sum(torch.mul(pred_vec, pred_vec), 1, keepdim=True)
    dot_pose_gt = torch.sum(torch.mul(pred_vec, gt_vec), 1, keepdim=True)

    s_opt = dot_pose_gt / dot_pose_pose

    pred_scaled = pred_orig * s_opt[:, :, None]
    if return_mean:
        nmpjpe = torch.mean(torch.sqrt(torch.sum((pred_scaled - gt_orig) ** 2, dim=-1)), dim=1)
    else:
        nmpjpe = torch.sqrt(torch.sum((pred_scaled - gt_orig) ** 2, dim=-1))
    return nmpjpe
