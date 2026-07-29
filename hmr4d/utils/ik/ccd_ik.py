# Sebastian IK
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
import hmr4d.utils.matrix as matrix
from hmr4d.utils.geo.quaternion import qbetween, qslerp, qinv, qmul, qrot


class CCD_IK:
    def __init__(
        self,
        local_mat,
        parent,
        target_ind,
        target_pos=None,
        target_rot=None,
        kinematic_chain=None,
        max_iter=2,  # Sebastian은 25로 설정했지만 수렴 판정을 사용하면 2로 충분하다.
        threshold=0.001,
        pos_weight=1.0,
        rot_weight=0.0,  # Sebastian은 1.0을 쓰지만 최적화가 불안정해질 수 있다.
    ):
        if kinematic_chain is None:
            kinematic_chain = range(local_mat.shape[-3])
        global_mat = matrix.forward_kinematics(local_mat, parent)

        # kinematic chain의 local matrix만 가져오고 IK 중 root를 바꾸지 않도록 root matrix를 지정한다.
        local_mat = local_mat.clone()
        local_mat = local_mat[..., kinematic_chain, :, :]
        local_mat[..., 0, :, :] = global_mat[..., kinematic_chain[0], :, :]

        parent = [i - 1 for i in range(len(kinematic_chain))]
        self.local_mat = local_mat
        self.global_mat = matrix.forward_kinematics(local_mat, parent)  # (*, J, 4, 4)
        self.parent = parent

        self.target_ind = target_ind
        if target_pos is not None:
            self.target_pos = target_pos  # (*, O, 3)
        else:
            self.target_pos = None
        if target_rot is not None:
            self.target_q = matrix_to_quaternion(target_rot)  # (*, O, 4)
        else:
            self.target_q = None

        self.threshold = threshold
        self.J_N = self.local_mat.shape[-3]
        self.target_N = len(target_ind)
        self.max_iter = max_iter
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight

    def is_converged(self):
        end_pos = matrix.get_position(self.global_mat)[..., self.target_ind, :]  # (*, OJ, 3)
        converged_mask = (self.target_pos - end_pos).norm(dim=-1) < self.threshold
        self.converged_mask = converged_mask
        if self.converged_mask.sum() > 0:
            return False
        return True

    def solve(self):
        for _ in range(self.max_iter):
            # if self.is_converged():
            #     return self.local_mat
            # root는 최적화하지 않으므로 1부터 시작한다.
            self.optimize(1)
        return self.local_mat

    def optimize(self, i):
        # i: joint_i
        if i == self.J_N - 1:
            return
        pos = matrix.get_position(self.global_mat)[..., i, :]  # (*, 3)
        rot = matrix.get_rotation(self.global_mat)[..., i, :, :]  # (*, 3, 3)
        quat = matrix_to_quaternion(rot)  # (*, 4)
        x_vec = torch.zeros((quat.shape[:-1] + (3,)), device=quat.device)
        x_vec[..., 0] = 1.0
        x_vec_sum = torch.zeros_like(x_vec)
        y_vec = torch.zeros((quat.shape[:-1] + (3,)), device=quat.device)
        y_vec[..., 1] = 1.0
        y_vec_sum = torch.zeros_like(y_vec)

        count = 0

        for target_i, j in enumerate(self.target_ind):
            if i >= j:
                # target과 같은 joint 또는 target의 자식 joint는 최적화하지 않는다.
                continue
            end_pos = matrix.get_position(self.global_mat)[..., j, :]  # (*, 3)
            end_rot = matrix.get_rotation(self.global_mat)[..., j, :, :]  # (*, 3, 3)
            end_quat = matrix_to_quaternion(end_rot)  # (*, 4)

            if self.target_pos is not None:
                target_pos = self.target_pos[..., target_i, :]  # (*, 3)
                # 목표 위치를 푼다.
                solved_pos_target_quat = qslerp(
                    quat,
                    qmul(qbetween(end_pos - pos, target_pos - pos), quat),
                    self.get_weight(i),
                )

                x_vec_sum += qrot(solved_pos_target_quat, x_vec)
                y_vec_sum += qrot(solved_pos_target_quat, y_vec)
                if self.pos_weight > 0:
                    count += 1

            if self.target_q is not None:
                if target_i < self.target_N - 1:
                    # 회전 target이 여러 개면 더 불안정하므로 마지막 것만 사용한다.
                    continue
                # 회전 target 최적화는 안정적이지 않다.
                target_q = self.target_q[..., target_i, :]  # (*, 4)
                # 목표 회전을 푼다.
                solved_q_target_quat = qslerp(
                    quat,
                    qmul(qmul(target_q, qinv(end_quat)), quat),
                    self.get_weight(i),
                )
                x_vec_sum += qrot(solved_q_target_quat, x_vec) * self.rot_weight
                y_vec_sum += qrot(solved_q_target_quat, y_vec) * self.rot_weight
                if self.rot_weight > 0:
                    count += 1

        if count > 0:
            x_vec_avg = matrix.normalize(x_vec_sum / count)
            y_vec_avg = matrix.normalize(y_vec_sum / count)
            z_vec_avg = torch.cross(x_vec_avg, y_vec_avg, dim=-1)
            solved_rot = torch.stack([x_vec_avg, y_vec_avg, z_vec_avg], dim=-1)  # 열 방향

            parent_rot = matrix.get_rotation(self.global_mat)[..., self.parent[i], :, :]
            solved_local_rot = matrix.get_mat_BtoA(parent_rot, solved_rot)
            self.local_mat[..., i, :-1, :-1] = solved_local_rot
            self.global_mat = matrix.forward_kinematics(self.local_mat, self.parent)
        self.optimize(i + 1)

    def get_weight(self, i):
        weight = (i + 1) / self.J_N
        return weight
