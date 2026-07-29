import torch
import numpy as np
from hmr4d.utils.geo_transform import project_p2d, convert_bbx_xys_to_lurb, cvt_to_bi01_p2d


def estimate_focal_length(img_w, img_h):
    return (img_w**2 + img_h**2) ** 0.5  # diagonal FoV는 약 53도입니다.


def estimate_K(img_w, img_h):
    focal_length = estimate_focal_length(img_w, img_h)
    K = torch.eye(3).float()
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = img_w / 2.0
    K[1, 2] = img_h / 2.0
    return K


def convert_K_to_K4(K):
    K4 = torch.stack([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]).float()
    return K4


def convert_f_to_K(focal_length, img_w, img_h):
    K = torch.eye(3).float()
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = img_w / 2.0
    K[1, 2] = img_h / 2.0
    return K


def resize_K(K, f=0.5):
    K = K.clone() * f
    K[..., 2, 2] = 1.0
    return K


def create_camera_sensor(width=None, height=None, f_fullframe=None):
    if width is None or height is None:
        # mobile phone image sensor에서 널리 쓰이는 4:3 aspect ratio를 사용합니다.
        if np.random.rand() < 0.5:
            width, height = 1200, 1600
        else:
            width, height = 1600, 1200

    # 일반적인 옵션에서 FoV를 샘플링합니다.
    # 1. mobile phone에는 wide-angle lens가 흔합니다.
    # 2. telephoto lens는 perspective effect가 작아 학습하기 쉽습니다.
    if f_fullframe is None:
        f_fullframe_options = [24, 26, 28, 30, 35, 40, 50, 60, 70]
        f_fullframe = np.random.choice(f_fullframe_options)

    # diagonal 길이를 기준으로 focal length를 매핑합니다.
    # 참고: https://www.nikonians.org/reviews/fov-tables
    diag_fullframe = (24**2 + 36**2) ** 0.5
    diag_img = (width**2 + height**2) ** 0.5
    focal_length = diag_img / diag_fullframe * f_fullframe

    K_fullimg = torch.eye(3)
    K_fullimg[0, 0] = focal_length
    K_fullimg[1, 1] = focal_length
    K_fullimg[0, 2] = width / 2
    K_fullimg[1, 2] = height / 2

    return width, height, K_fullimg


# ====== CLIFF camera 계산 ===== #


def convert_xys_to_cliff_cam_wham(xys, res):
    """
    인자:
        xys: pixel 단위 (N, 3). s에는 200을 적용하지 않습니다.
        res: (2), e.g. [4112., 3008.]  (w,h)
    반환:
        cliff_cam: (N, 3), 정규화된 표현
    """

    def normalize_keypoints_to_image(x, res):
        """
        인자:
            x: (N, 2), 중심 좌표
            res: (2), e.g. [4112., 3008.]
        반환:
            x_normalized: (N, 2)
        """
        res = res.to(x.device)
        scale = res.max(-1)[0].reshape(-1)
        mean = torch.stack([res[..., 0] / scale, res[..., 1] / scale], dim=-1).to(x.device)
        x = 2 * x / scale.reshape(*[1 for i in range(len(x.shape[1:]))]) - mean.reshape(
            *[1 for i in range(len(x.shape[1:-1]))], -1
        )
        return x

    centers = normalize_keypoints_to_image(xys[:, :2], res)  # (N, 2)
    scale = xys[:, 2:] / res.max()
    location = torch.cat((centers, scale), dim=-1)
    return location


def compute_bbox_info_bedlam(bbx_xys, K_fullimg):
    """BEDLAM 방식으로 bounding box 정보를 계산합니다.

    인자:
        bbx_xys: ((B), N, 3), K_fullimg가 정의하는 pixel 좌표계
        K_fullimg: ((B), (N), 3, 3)
    반환:
        bbox_info: ((B), N, 3)
    """
    fl = K_fullimg[..., 0, 0].unsqueeze(-1)
    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]

    cx, cy, b = bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2]
    bbox_info = torch.stack([cx - icx, cy - icy, b], dim=-1)
    bbox_info = bbox_info / fl
    return bbox_info


# ====== 예측값을 camera translation으로 변환 ===== #


def compute_transl_full_cam(pred_cam, bbx_xys, K_fullimg):
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    focal_length = K_fullimg[..., 0, 0]

    icx = K_fullimg[..., 0, 2]
    icy = K_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * focal_length / (sb + 1e-9)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t


def get_a_pred_cam(transl, bbx_xys, K_fullimg):
    """compute_transl_full_cam의 역연산을 수행합니다."""
    assert transl.ndim == bbx_xys.ndim  # (*, L, 3)
    assert K_fullimg.ndim == (bbx_xys.ndim + 1)  # (*, L, 3, 3)
    f = K_fullimg[..., 0, 0]
    cx = K_fullimg[..., 0, 2]
    cy = K_fullimg[..., 1, 2]
    gt_s = 2 * f / (transl[..., 2] * bbx_xys[..., 2])  # (B, L)
    gt_x = transl[..., 0] - transl[..., 2] / f * (bbx_xys[..., 0] - cx)
    gt_y = transl[..., 1] - transl[..., 2] / f * (bbx_xys[..., 1] - cy)
    gt_pred_cam = torch.stack([gt_s, gt_x, gt_y], dim=-1)
    return gt_pred_cam


# ====== 3D를 2D로 투영 ===== #


def project_to_bi01(points, bbx_xys, K_fullimg):
    """
    points: (B, L, J, 3)
    bbx_xys: (B, L, 3)
    K_fullimg: (B, L, 3, 3)
    """
    # p2d = project_p2d(points, K_fullimg)
    p2d = perspective_projection(points, K_fullimg)
    bbx_lurb = convert_bbx_xys_to_lurb(bbx_xys)
    p2d_bi01 = cvt_to_bi01_p2d(p2d, bbx_lurb)
    return p2d_bi01


def perspective_projection(points, K):
    # points: (B, L, J, 3)
    # K: (B, L, 3, 3)
    projected_points = points / points[..., -1].unsqueeze(-1)
    projected_points = torch.einsum("...ij,...kj->...ki", K, projected_points.float())
    return projected_points[..., :-1]


# ====== 2D joint에서 bounding box 계산 ===== #


def normalize_kp2d(obs_kp2d, bbx_xys, clamp_scale_min=False):
    """
    인자:
        obs_kp2d: (B, L, J, 3) [x, y, c]
        bbx_xys: (B, L, 3)
    반환:
        obs: (B, L, J, 3)  [x, y, c]
    """
    obs_xy = obs_kp2d[..., :2]  # (B, L, J, 2)
    obs_conf = obs_kp2d[..., 2]  # (B, L, J)
    center = bbx_xys[..., :2]
    scale = bbx_xys[..., [2]]

    # bounding box 밖의 keypoint는 보이지 않는 것으로 처리합니다.
    xy_max = center + scale / 2
    xy_min = center - scale / 2
    invisible_mask = (
        (obs_xy[..., 0] < xy_min[..., None, 0])
        + (obs_xy[..., 0] > xy_max[..., None, 0])
        + (obs_xy[..., 1] < xy_min[..., None, 1])
        + (obs_xy[..., 1] > xy_max[..., None, 1])
    )
    obs_conf = obs_conf * ~invisible_mask
    if clamp_scale_min:
        scale = scale.clamp(min=1e-5)
    normalized_obs_xy = 2 * (obs_xy - center.unsqueeze(-2)) / scale.unsqueeze(-2)

    return torch.cat([normalized_obs_xy, obs_conf[..., None]], dim=-1)


def get_bbx_xys(i_j2d, bbx_ratio=[192, 256], do_augment=False, base_enlarge=1.2):
    """(B, L, J, 3) [x, y, c]에서 (B, L, 3) bounding box를 구합니다."""
    # 중심
    min_x = i_j2d[..., 0].min(-1)[0]
    max_x = i_j2d[..., 0].max(-1)[0]
    min_y = i_j2d[..., 1].min(-1)[0]
    max_y = i_j2d[..., 1].max(-1)[0]
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # 크기
    h = max_y - min_y  # (B, L)
    w = max_x - min_x  # (B, L)

    if True:  # width와 height를 지정한 aspect ratio에 맞춥니다.
        aspect_ratio = bbx_ratio[0] / bbx_ratio[1]
        mask1 = w > aspect_ratio * h
        h[mask1] = w[mask1] / aspect_ratio
        mask2 = w < aspect_ratio * h
        w[mask2] = h[mask2] * aspect_ratio

    # 공통 factor를 적용해 bounding box를 확대합니다.
    bbx_size = torch.max(h, w) * base_enlarge

    if do_augment:
        B, L = bbx_size.shape[:2]
        device = bbx_size.device
        if True:
            scaleFactor = torch.rand((B, L), device=device) * 0.3 + 1.05  # 1.05~1.35
            txFactor = torch.rand((B, L), device=device) * 1.6 - 0.8  # -0.8~0.8
            tyFactor = torch.rand((B, L), device=device) * 1.6 - 0.8  # -0.8~0.8
        else:
            scaleFactor = torch.rand((B, 1), device=device) * 0.3 + 1.05  # 1.05~1.35
            txFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8  # -0.8~0.8
            tyFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8  # -0.8~0.8

        raw_bbx_size = bbx_size / base_enlarge
        bbx_size = raw_bbx_size * scaleFactor
        center_x += raw_bbx_size / 2 * ((scaleFactor - 1) * txFactor)
        center_y += raw_bbx_size / 2 * ((scaleFactor - 1) * tyFactor)

    return torch.stack([center_x, center_y, bbx_size], dim=-1)


def safely_render_x3d_K(x3d, K_fullimg, thr):
    """
    인자:
        x3d: (B, L, V, 3), 하나 이상의 안전한 point가 있어야 합니다.
        K_fullimg: (B, L, 3, 3)
    반환:
        bbx_xys: (B, L, 3)
        i_x2d: (B, L, V, 2)
    """
    # frame별로 안전하지 않은 z(<thr)를 안전한 z값으로 바꿉니다.
    x3d = x3d.clone()  # (B, L, V, 3)
    x3d_unsafe_mask = x3d[..., 2] < thr  # (B, L, V)
    if (x3d_unsafe_mask).sum() > 0:
        x3d[..., 2][x3d_unsafe_mask] = thr
        if False:
            from hmr4d.utils.wis3d_utils import make_wis3d

            wis3d = make_wis3d(name="debug-update-z")
            bs, ls, vs = torch.where(x3d_unsafe_mask)
            bs = torch.unique(bs)
            for b in bs:
                for f in range(x3d.size(1)):
                    wis3d.set_scene_id(f)
                    wis3d.add_point_cloud(x3d[b, f], name="unsafe")
                pass

    # 2D로 투영합니다.
    i_x2d = perspective_projection(x3d, K_fullimg)  # (B, L, V, 2)
    return i_x2d


def get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2):
    """
    인자:
        bbx_xyxy: (N, 4) [x1, y1, x2, y2]
    반환:
        bbx_xys: (N, 3) [center_x, center_y, size]
    """

    i_p2d = torch.stack([bbx_xyxy[:, [0, 1]], bbx_xyxy[:, [2, 3]]], dim=1)  # (L, 2, 2)
    bbx_xys = get_bbx_xys(i_p2d[None], base_enlarge=base_enlarge)[0]
    return bbx_xys


def bbx_xyxy_from_x(p2d):
    """
    인자:
        p2d: (*, V, 2), 2D point tensor

    반환:
        bbx_xyxy: (*, 4), (xmin, ymin, xmax, ymax) 형식의 bounding box 좌표
    """
    # bounding box의 최소·최대 좌표를 계산합니다.
    xy_min = p2d.min(dim=-2).values  # (*, 2)
    xy_max = p2d.max(dim=-2).values  # (*, 2)

    # 최소·최대 좌표를 연결해 bounding box를 만듭니다.
    bbx_xyxy = torch.cat([xy_min, xy_max], dim=-1)  # (*, 4)

    return bbx_xyxy


def bbx_xyxy_from_masked_x(p2d, mask):
    """
    인자:
        p2d: (*, V, 2), 2D point tensor
        mask: (*, V), 유효한 point를 나타내는 boolean tensor

    반환:
        bbx_xyxy: (*, 4), (xmin, ymin, xmax, ymax) 형식의 bounding box 좌표
    """
    # p2d와 mask의 shape가 호환되는지 확인합니다.
    assert p2d.shape[:-1] == mask.shape, "The shape of p2d and mask are not compatible."

    # batch 처리를 위해 입력 tensor를 펼칩니다.
    p2d_flat = p2d.view(-1, p2d.shape[-2], p2d.shape[-1])
    mask_flat = mask.view(-1, mask.shape[-1])

    # mask 밖의 값을 각각 큰 양수와 큰 음수로 바꿉니다.
    p2d_min = torch.where(mask_flat.unsqueeze(-1), p2d_flat, torch.tensor(float("inf")).to(p2d_flat))
    p2d_max = torch.where(mask_flat.unsqueeze(-1), p2d_flat, torch.tensor(float("-inf")).to(p2d_flat))

    # bounding box의 최소·최대 좌표를 계산합니다.
    xy_min = p2d_min.min(dim=1).values  # (BL, 2)
    xy_max = p2d_max.max(dim=1).values  # (BL, 2)

    # 최소·최대 좌표를 연결해 bounding box를 만듭니다.
    bbx_xyxy = torch.cat([xy_min, xy_max], dim=-1)  # (BL, 4)

    # 원래의 shape prefix로 복원합니다.
    bbx_xyxy = bbx_xyxy.view(*p2d.shape[:-2], 4)

    return bbx_xyxy


def bbx_xyxy_ratio(xyxy1, xyxy2):
    """FoV 및 unbounded 상황을 위한 bounding box 면적 비율을 계산합니다.

    인자:
        xyxy1: (*, 4)
        xyxy2: (*, 4)
    반환:
        ratio: (*), squared_area(xyxy1) / squared_area(xyxy2)
    """
    area1 = (xyxy1[..., 2] - xyxy1[..., 0]) * (xyxy1[..., 3] - xyxy1[..., 1])
    area2 = (xyxy2[..., 2] - xyxy2[..., 0]) * (xyxy2[..., 3] - xyxy2[..., 1])
    # 유효성을 확인합니다.
    area1[~torch.isfinite(area1)] = 0  # area1의 inf를 0으로 바꿉니다.
    assert (area2 > 0).all(), "area2 should be positive"
    return area1 / area2


def get_mesh_in_fov_category(mask):
    """mask: (L, V)
    분류 기준:
    1. FullyVisible: 모든 frame의 mesh 전체가 FoV 안에 있습니다.
    2. PartiallyVisible: 일부 frame에서 mesh 일부만 FoV 밖에 있습니다.
    3. PartiallyOut: 일부 frame에서 mesh 전체가 FoV 밖에 있고 다른 frame에서는 보입니다.
    4. FullyOut: 모든 frame의 mesh 전체가 FoV 밖에 있습니다.
    """
    mask = mask.clone().cpu()
    is_class1 = mask.all()  #  FullyVisible
    is_class2 = mask.any(1).all() * ~is_class1  # PartiallyVisible
    is_class4 = ~(mask.any())  # PartiallyOut
    is_class3 = ~is_class1 * ~is_class2 * ~is_class4  # FullyOut

    mask_frame_any_verts = mask.any(1)
    assert is_class1.int() + is_class2.int() + is_class3.int() + is_class4.int() == 1
    class_type = is_class1.int() + 2 * is_class2.int() + 3 * is_class3.int() + 4 * is_class4.int()
    return class_type.item(), mask_frame_any_verts


def get_infov_mask(p2d, w_real, h_real):
    """
    인자:
        p2d: (B, L, V, 2)
        w_real, h_real: (B, L) or int
    반환:
        mask: (B, L, V)
    """
    x, y = p2d[..., 0], p2d[..., 1]
    if isinstance(w_real, int):
        mask = (x >= 0) * (x < w_real) * (y >= 0) * (y < h_real)
    else:
        mask = (x >= 0) * (x < w_real[..., None]) * (y >= 0) * (y < h_real[..., None])
    return mask
