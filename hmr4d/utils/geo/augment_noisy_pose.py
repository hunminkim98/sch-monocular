import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
import hmr4d.utils.matrix as matrix
from hmr4d import PROJ_ROOT

COCO17_AUG = {k: v.flatten() for k, v in torch.load(PROJ_ROOT / "hmr4d/utils/body_model/coco_aug_dict.pth").items()}
COCO17_AUG_CUDA = {}
COCO17_TREE = [[5, 6], 0, 0, 1, 2, -1, -1, 5, 6, 7, 8, -1, -1, 11, 12, 13, 14, 15, 15, 15, 16, 16, 16]
COCO23_TREE = [[5, 6], 0, 0, 1, 2, -1, -1, 5, 6, 7, 8, -1, -1, 11, 12, 13, 14, 15, 15, 15, 16, 16, 16]


def gaussian_augment(body_pose, std_angle=10.0, to_R=True):
    """
    인자:
        body_pose torch.Tensor: to_R이 True이면 (..., J, 3) axis-angle,
            아니면 (..., J, 3, 3) rotation matrix
        std_angle: degree 단위의 scalar 또는 list
    """

    body_pose = body_pose.clone()

    if to_R:
        body_pose_R = axis_angle_to_matrix(body_pose)  # (B, L, J, 3, 3)
    else:
        body_pose_R = body_pose
    shape = body_pose_R.shape[:-2]
    device = body_pose.device

    # 1. noise를 시뮬레이션합니다.
    # 회전 각도
    std_angle = torch.tensor(std_angle).to(device).reshape(-1)  # scalar와 list를 모두 허용합니다.
    noise_angle = torch.randn(shape, device=device) * std_angle * torch.pi / 180

    # axis: zero vector를 방지합니다.
    noise_axis = torch.rand((*shape, 3), device=device)
    mask_ = torch.norm(noise_axis, dim=-1) < 1e-6
    noise_axis[mask_] = 1

    noise_axis = noise_axis / torch.norm(noise_axis, dim=-1, keepdim=True)
    noise_aa = noise_angle[..., None] * noise_axis  # (B, L, J, 3)
    noise_R = axis_angle_to_matrix(noise_aa)  # (B, L, J, 3, 3)

    # 2. body pose에 noise를 추가합니다.
    new_body_pose_R = matrix.get_mat_BfromA(body_pose_R, noise_R)  # (B, L, J, 3, 3)
    # new_body_pose_R = torch.matmul(noise_R, body_pose_R)
    new_body_pose_r6d = matrix_to_rotation_6d(new_body_pose_R)  # (B, L, J, 6)
    new_body_pose_aa = matrix_to_axis_angle(new_body_pose_R)  # (B, L, J, 3)

    return new_body_pose_R, new_body_pose_r6d, new_body_pose_aa


# ========= 3D joint 증강 ======== #


def get_jitter(shape=(8, 120), s_jittering=5e-2):
    """Gaussian jitter를 모델링합니다."""
    jittering_noise = (
        torch.normal(
            mean=torch.zeros((*shape, 17, 3)),
            std=COCO17_AUG["jittering"].reshape(1, 1, 17, 1).expand(*shape, -1, 3),
        )
        * s_jittering
    )
    return jittering_noise


def get_jitter_cuda(shape=(8, 120, 17), s_jittering=5e-2):
    if "jittering" not in COCO17_AUG_CUDA:
        COCO17_AUG_CUDA["jittering"] = COCO17_AUG["jittering"].cuda().reshape(1, 1, 17, 1)
        assert shape[2] == 17 or shape[2] == 23
        if shape[2] > 17:
            fill_value = 0.1  # COCO17_AUG_CUDA["jittering"][0, 0, 16].item()
            foot_pmask = torch.full((1, 1, 6, 1), fill_value, dtype=torch.float32, device="cuda")
            COCO17_AUG_CUDA["jittering"] = torch.cat((COCO17_AUG_CUDA["jittering"], foot_pmask), dim=2)
    jittering = COCO17_AUG_CUDA["jittering"]
    # joint별 무작위 jitter
    jittering_noise = torch.randn((*shape, 3), device="cuda") * jittering * s_jittering
    return jittering_noise


def get_lfhp(shape=(8, 120), s_peak=3e-1, s_peak_mask=5e-3):
    """low-frequency high-peak noise를 모델링합니다."""

    def get_peak_noise_mask():
        peak_noise_mask = torch.rand(*shape, 17) * COCO17_AUG["pmask"]
        peak_noise_mask = peak_noise_mask < s_peak_mask
        return peak_noise_mask

    peak_noise_mask = get_peak_noise_mask()  # (B, L, 17)
    peak_noise = peak_noise_mask.float().unsqueeze(-1).repeat(1, 1, 1, 3)
    peak_noise = peak_noise * torch.randn(3) * COCO17_AUG["peak"].reshape(17, 1) * s_peak
    return peak_noise


def get_lfhp_cuda(shape=(8, 120, 17), s_peak=3e-1, s_peak_mask=5e-3):
    if "peak" not in COCO17_AUG_CUDA:
        COCO17_AUG_CUDA["pmask"] = COCO17_AUG["pmask"].cuda()
        COCO17_AUG_CUDA["peak"] = COCO17_AUG["peak"].cuda().reshape(17, 1)
        assert shape[2] == 17 or shape[2] == 23
        if shape[2] > 17:
            fill_value = COCO17_AUG_CUDA["pmask"][16].item()
            foot_pmask = torch.full((6,), fill_value, dtype=torch.float32, device="cuda")
            COCO17_AUG_CUDA["pmask"] = torch.cat((COCO17_AUG_CUDA["pmask"], foot_pmask), dim=0)

            fill_value = 0.08  # COCO17_AUG_CUDA["peak"][16].item()
            foot_peak = torch.full((6, 1), fill_value, dtype=torch.float32, device="cuda")
            COCO17_AUG_CUDA["peak"] = torch.cat((COCO17_AUG_CUDA["peak"], foot_peak), dim=0)

    pmask = COCO17_AUG_CUDA["pmask"]
    peak = COCO17_AUG_CUDA["peak"]
    peak_noise_mask = torch.rand(*shape, device="cuda") * pmask < s_peak_mask
    # joint별 큰 무작위 offset
    peak_noise = (
        peak_noise_mask.float().unsqueeze(-1).expand(-1, -1, -1, 3) * torch.randn(3, device="cuda") * peak * s_peak
    )
    return peak_noise


def get_bias(shape=(8, 120), s_bias=1e-1):
    """bias noise를 모델링합니다."""
    b, l = shape
    bias_noise = torch.normal(mean=torch.zeros((b, 17, 3)), std=COCO17_AUG["bias"].reshape(1, 17, 1)) * s_bias
    bias_noise = bias_noise[:, None].expand(-1, l, -1, -1)  # (B, L, J, 3), 전체 sequence에 같은 bias를 적용합니다.
    return bias_noise


def get_bias_cuda(shape=(8, 120, 17), s_bias=1e-1):
    if "bias" not in COCO17_AUG_CUDA:
        COCO17_AUG_CUDA["bias"] = COCO17_AUG["bias"].cuda().reshape(1, 17, 1)
        assert shape[2] == 17 or shape[2] == 23
        if shape[2] > 17:
            fill_value = 0.1  # COCO17_AUG_CUDA["bias"][0, 16].item()
            foot_bias = torch.full((1, 6, 1), fill_value, dtype=torch.float32, device="cuda")
            COCO17_AUG_CUDA["bias"] = torch.cat((COCO17_AUG_CUDA["bias"], foot_bias), dim=1)

    bias = COCO17_AUG_CUDA["bias"]
    # 전체 sequence에 동일한 정규분포 offset을 적용합니다.
    bias_noise = torch.randn((shape[0], shape[2], 3), device="cuda") * bias * s_bias
    bias_noise = bias_noise[:, None].expand(-1, shape[1], -1, -1)
    return bias_noise


def get_wham_aug_kp3d(shape=(8, 120, 17)):
    aug = get_bias_cuda(shape) + get_lfhp_cuda(shape) + get_jitter_cuda(shape)
    return aug


def get_visible_mask(shape=(8, 120, 17), s_mask=0.03):
    """visibility mask를 모델링합니다."""
    B, L, J = shape
    # frame 및 joint 단위
    mask = torch.rand(*shape) < s_mask
    visible = (~mask).clone()  # (B, L, 17)
    visible = visible.reshape(-1, shape[2])  # (BL, 17)
    if shape[2] == 17:
        for child in range(17):
            parent = COCO17_TREE[child]
            if parent == -1:
                continue
            if isinstance(parent, list):
                visible[:, child] *= visible[:, parent[0]] * visible[:, parent[1]]
            else:
                visible[:, child] *= visible[:, parent]
    else:
        assert shape[2] == 23
        for child in range(23):
            parent = COCO23_TREE[child]
            if parent == -1:
                continue
            if isinstance(parent, list):
                visible[:, child] *= visible[:, parent[0]] * visible[:, parent[1]]
            else:
                visible[:, child] *= visible[:, parent]

    visible = visible.reshape(*shape).clone()  # (B, L, J)
    return visible


def get_invisible_legs_mask(shape, s_mask=0.03):
    """
    무작위 구간 동안 양쪽 다리를 보이지 않게 만듭니다.
    """
    B, L, J = shape
    starts = torch.randint(0, L - 90, (B,))
    ends = starts + torch.randint(30, 90, (B,))
    mask_range = torch.arange(L).unsqueeze(0).expand(B, -1)
    mask_to_apply = (mask_range >= starts.unsqueeze(1)) & (mask_range < ends.unsqueeze(1))
    mask_to_apply = mask_to_apply.unsqueeze(2).expand(-1, -1, J).clone()
    mask_to_apply[:, :, :11] = False  # 양쪽 다리만 보이지 않게 합니다.
    mask_to_apply = mask_to_apply & (torch.rand(B, 1, 1) < s_mask)
    return mask_to_apply


def randomly_occlude_lower_half(i_x2d, s_mask=0.03):
    """
    image의 아래쪽 절반을 무작위로 가립니다.
    """
    raise NotImplementedError
    B, L, N, _ = i_x2d.shape
    i_x2d = i_x2d.clone()

    # image 아래쪽 절반이 보이지 않는 구간을 정합니다.
    starts = torch.randint(0, L - 90, (B,))
    ends = starts + torch.randint(30, 90, (B,))
    mask_range = torch.arange(L).unsqueeze(0).expand(B, -1)
    mask_to_apply = (mask_range >= starts.unsqueeze(1)) & (mask_range < ends.unsqueeze(1))
    mask_to_apply = mask_to_apply.unsqueeze(2).expand(-1, -1, N)  # (B, L, N)

    # image 아래쪽 절반만 보이지 않게 합니다.
    i_x2d
    i_x2d[..., 1] / 2

    mask_to_apply = mask_to_apply & (torch.rand(B, 1, 1) < s_mask)
    return mask_to_apply


def randomly_modify_hands_legs(j3d):
    num_joints = j3d.shape[2]
    assert num_joints == 17 or num_joints == 23
    B, L, J, _ = j3d.shape

    body_switch_pairs = [(9, 10), (15, 16)]  # 발목과 손목 joint
    body_wrong_pairs = [(9, 10), (10, 9), (15, 16), (16, 15)]
    p_switch_body = 0.001
    p_wrong_body_kpt = 0.001

    # 왼쪽과 오른쪽 joint를 맞바꿉니다.
    mask = torch.rand((B, L, 2, 1), device=j3d.device) < p_switch_body
    for idx, (j1, j2) in enumerate(body_switch_pairs):
        tmp = j3d[:, :, j1, :].clone()
        j3d[:, :, j1, :] = torch.where(mask[:, :, idx], j3d[:, :, j2, :], j3d[:, :, j1, :])
        j3d[:, :, j2, :] = torch.where(mask[:, :, idx], tmp, j3d[:, :, j2, :])

    # 왼쪽 joint를 오른쪽 joint로, 또는 그 반대로 잘못 인식하게 만듭니다.
    mask = torch.rand((B, L, 4, 1), device=j3d.device) < p_wrong_body_kpt
    for idx, (j1, j2) in enumerate(body_wrong_pairs):
        j3d[:, :, j1, :] = torch.where(mask[:, :, idx], j3d[:, :, j2, :], j3d[:, :, j1, :])

    if num_joints > 17:
        foot_switch_pairs = [(17, 20), (18, 21), (19, 22)]
        foot_wrong_pairs = [(17, 20), (20, 17), (18, 21), (21, 18), (19, 22), (22, 19)]

        p_switch_foot = 0.001
        p_wrong_foot_kpt = 0.001

        # 왼쪽과 오른쪽 joint를 맞바꿉니다.
        mask = torch.rand((B, L, 3, 1), device=j3d.device) < p_switch_foot
        for idx, (j1, j2) in enumerate(foot_switch_pairs):
            tmp = j3d[:, :, j1, :].clone()
            j3d[:, :, j1, :] = torch.where(mask[:, :, idx], j3d[:, :, j2, :], j3d[:, :, j1, :])
            j3d[:, :, j2, :] = torch.where(mask[:, :, idx], tmp, j3d[:, :, j2, :])

        # 왼쪽 joint를 오른쪽 joint로, 또는 그 반대로 잘못 인식하게 만듭니다.
        mask = torch.rand((B, L, 6, 1), device=j3d.device) < p_wrong_foot_kpt
        for idx, (j1, j2) in enumerate(foot_wrong_pairs):
            j3d[:, :, j1, :] = torch.where(mask[:, :, idx], j3d[:, :, j2, :], j3d[:, :, j1, :])

    return j3d
