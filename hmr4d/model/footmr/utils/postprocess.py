import torch
from torch.cuda.amp import autocast
from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)

import hmr4d.utils.matrix as matrix
from hmr4d.utils.ik.ccd_ik import CCD_IK
from hmr4d.utils.geo_transform import get_sequence_cammat, transform_mat, apply_T_on_points
from hmr4d.utils.net_utils import gaussian_smooth
from hmr4d.model.footmr.utils.endecoder import EnDecoder

from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines


@autocast(enabled=False)
def pp_static_joint(outputs, endecoder: EnDecoder):
    # 예측한 stationary joint label과 일치하도록 global root translation만 변경합니다.
    # global forward kinematics를 계산합니다.
    pred_w_j3d = endecoder.fk_v2(**outputs["pred_smpl_params_global"])
    L = pred_w_j3d.shape[1]
    joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    pred_j3d_static = pred_w_j3d.clone()[:, :, joint_ids]  # (B, L, J, 3)

    # static 정보를 사용해 전체 움직임을 갱신하고 displacement가 [0, 0, 0]에 가까워지게 합니다.
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J, 3)

    static_conf_logits = outputs["static_conf_logits"][:, :-1].clone()
    static_label_ = static_conf_logits > 0  # (B, L-1, J) non-contact frame을 제외합니다.
    # static이 아니라고 예측된 joint에는 매우 작은 static confidence logit을 부여합니다.
    static_conf_logits = static_conf_logits.float() - (~static_label_ * 1e6)  # FP16은 softmax를 통과할 수 없습니다.
    is_static = static_label_.sum(dim=-1) > 0  # (B, L-1) 하나 이상의 joint가 static이면 True입니다.
    # 매우 작은 값 때문에 softmax에서 non-static joint의 가중치는 0이 됩니다.
    # frame별 displacement 하나를 가중 평균으로 계산합니다.
    # static joint에는 높은 가중치를, non-static joint에는 0의 가중치를 사용합니다.
    pred_disp = pred_j_disp * static_conf_logits[..., None].softmax(dim=-2)  # (B, L-1, J, 3)
    pred_disp = pred_disp * is_static[..., None, None]  # (B, L-1, J, 3)
    # static으로 예측된 joint의 평균 displacement를 구합니다.
    # joint가 정지하도록 이 움직임을 상쇄해야 합니다.
    # 이를 위해 global translation을 조정합니다.
    pred_disp = pred_disp.sum(-2)  # (B, L-1, 3)
    ####################

    # 결과를 덮어씁니다.
    if False:  # for-loop 구현
        post_w_transl = outputs["pred_smpl_params_global"]["transl"].clone()  # (B, L, 3)
        for i in range(1, L):
            post_w_transl[:, i:] -= pred_disp[:, i - 1 : i]
    else:  # vectorized 구현
        pred_w_transl = outputs["pred_smpl_params_global"]["transl"].clone()  # (B, L, 3)
        pred_w_disp = pred_w_transl[:, 1:] - pred_w_transl[:, :-1]  # (B, L-1, 3)
        pred_w_disp_new = pred_w_disp - pred_disp
        post_w_transl = torch.cumsum(torch.cat([pred_w_transl[:, :1], pred_w_disp_new], dim=1), dim=1)
        post_w_transl[..., 0] = gaussian_smooth(post_w_transl[..., 0], dim=-1)
        post_w_transl[..., 2] = gaussian_smooth(post_w_transl[..., 2], dim=-1)

    # Open3D 시각화를 위해 -min(y)만큼 이동해 sequence를 지면에 놓습니다. 발 높이는 고려하지 않습니다.
    post_w_j3d = pred_w_j3d - pred_w_transl.unsqueeze(-2) + post_w_transl.unsqueeze(-2)
    ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]  # (B,) 최소 y값
    post_w_transl[..., 1] -= ground_y

    return post_w_transl


@autocast(enabled=False)
def pp_static_joint_cam(outputs, endecoder: EnDecoder):
    """static joint와 static camera 가정을 사용해 global translation을 보정합니다."""
    # 입력
    pred_smpl_params_incam = outputs["pred_smpl_params_incam"].copy()
    pred_smpl_params_global = outputs["pred_smpl_params_global"]
    static_conf_logits = outputs["static_conf_logits"].clone()[:, :-1]  # (B, L-1, J)
    joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    B, L = pred_smpl_params_incam["transl"].shape[:2]
    assert B == 1

    # forward kinematics를 계산합니다.
    pred_w_j3d = endecoder.fk_v2(**pred_smpl_params_global)  # (B, L, J, 3)
    # camera 좌표계 결과에는 noise가 있을 수 있으므로 smoothing합니다.
    pred_smpl_params_incam["transl"] = gaussian_smooth(pred_smpl_params_incam["transl"], sigma=5, dim=-2)
    pred_c_j3d = endecoder.fk_v2(**pred_smpl_params_incam)  # (B, L, J, 3)

    # 첫 frame에서 static T_c2w를 계산합니다.
    R_gv = axis_angle_to_matrix(pred_smpl_params_global["global_orient"][:, 0])  # (B, 3, 3)
    R_c = axis_angle_to_matrix(pred_smpl_params_incam["global_orient"][:, 0])  # (B, 3, 3)
    R_c2w = R_gv @ R_c.mT  # (B, 3, 3)
    t_c2w = pred_w_j3d[:, 0, 0] - torch.einsum("bij,bj->bi", R_c2w, pred_c_j3d[:, 0, 0])  # (B, 3)
    T_c2w = transform_mat(R_c2w, t_c2w)  # (B, 4, 4)
    pred_c_j3d_in_w = apply_T_on_points(pred_c_j3d, T_c2w[:, None])

    # 1. translation을 camera 좌표계 결과와 유사하게 만듭니다.
    post_w_transl = pred_smpl_params_global["transl"].clone()  # (B, L, 3)
    post_w_j3d = pred_w_j3d.clone()  # (B, L, J, 3)
    cp_thr = torch.tensor([0.25, 0.25, 0.25]).to(post_w_j3d)  # 오차가 매우 큰 예측만 갱신합니다.
    for i in range(1, L):
        cp_diff = post_w_j3d[:, i, 0] - pred_c_j3d_in_w[:, i, 0]  # (B, 3)
        cp_diff = cp_diff * ~((cp_diff > -cp_thr) * (cp_diff < cp_thr))
        cp_diff = torch.clamp(cp_diff, -0.02, 0.02)
        post_w_transl[:, i:] -= cp_diff
        post_w_j3d[:, i:] -= (cp_diff)[:, None, None]

    # 2. stationary joint가 계속 정지하도록 만듭니다.
    # pred_j3d_static = pred_w_j3d.clone()[:, :, joint_ids]  # (B, L, J, 3)
    pred_j3d_static = post_w_j3d[:, :, joint_ids]  # (B, L, J, 3)
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J, 3)

    static_label = static_conf_logits.sigmoid() > 0.8  # (B, L-1, J)
    static_label_sumJ = static_label.sum(-1, keepdim=True)  # (B, L-1, 1)
    static_label_sumJ = torch.clamp_min(static_label_sumJ, 1)  # 0을 1로 바꿉니다.
    pred_disp_sumJ = (pred_j_disp * static_label[..., None]).sum(-2)  # (B, L-1, 3)
    pred_disp = pred_disp_sumJ / static_label_sumJ  # (B, L-1, 3)
    pred_disp[:, :, 1] = 0  # y축은 수정하지 않습니다.

    # for-loop로 결과를 덮어씁니다.
    for i in range(1, L):
        post_w_transl[:, i:] -= pred_disp[:, [i - 1]]
        post_w_j3d[:, i:] -= pred_disp[:, [i - 1], None]

    # Open3D 시각화를 위해 -min(y)만큼 이동해 sequence를 지면에 놓습니다. 발 높이는 고려하지 않습니다.
    ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]  # (B,) 최소 y값
    post_w_transl[..., 1] -= ground_y

    return post_w_transl


@autocast(enabled=False)
def process_ik(outputs, endecoder):
    static_conf = outputs["static_conf_logits"].sigmoid()  # (B, L, J)
    post_w_j3d, local_mat, post_w_mat = endecoder.fk_v2(**outputs["pred_smpl_params_global"], get_intermediate=True)

    # Sebas 방식의 rollout 병합
    # 모든 frame을 순회하며, static으로 예측된 joint의 global position을
    # 직전 frame과 같게 설정합니다(직전 값과 현재 값의 가중 평균).
    # 이후 inverse kinematics로 새로운 local joint rotation을 구합니다.
    joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    post_target_j3d = post_w_j3d.clone()
    for i in range(1, post_w_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, joint_ids]
        this = post_w_j3d[:, i, joint_ids]
        c_prev = static_conf[:, i - 1, :, None]
        # 예측한 static confidence로 직전 frame과 현재 frame의 가중 평균을 구합니다.
        post_target_j3d[:, i, joint_ids] = prev * c_prev + this * (1 - c_prev)

    # inverse kinematics를 계산합니다.
    global_rot = matrix.get_rotation(post_w_mat)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    left_leg_chain = [0, 1, 4, 7, 10]
    right_leg_chain = [0, 2, 5, 8, 11]
    left_hand_chain = [9, 13, 16, 18, 20]
    right_hand_chain = [9, 14, 17, 19, 21]

    def ik(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        IK_solver = CCD_IK(
            local_mat,
            parents,
            target_ind,
            target_pos,
            target_rot,
            kinematic_chain=chain,
            max_iter=2,
        )

        chain_local_mat = IK_solver.solve()
        chain_rotmat = matrix.get_rotation(chain_local_mat)
        local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]  # (B, L, J, 3, 3)
        return local_mat

    local_mat = ik(local_mat, post_target_j3d[:, :, [7, 10]], global_rot[:, :, [7, 10]], [3, 4], left_leg_chain)
    local_mat = ik(local_mat, post_target_j3d[:, :, [8, 11]], global_rot[:, :, [8, 11]], [3, 4], right_leg_chain)
    local_mat = ik(local_mat, post_target_j3d[:, :, [20]], global_rot[:, :, [20]], [4], left_hand_chain)
    local_mat = ik(local_mat, post_target_j3d[:, :, [21]], global_rot[:, :, [21]], [4], right_hand_chain)

    body_pose = matrix_to_axis_angle(matrix.get_rotation(local_mat[:, :, 1:]))  # (B, L, J-1, 3, 3)
    body_pose = body_pose.flatten(2)  # (B, L, (J-1)*3)

    return body_pose
