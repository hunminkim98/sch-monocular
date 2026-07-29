import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import List, Optional

import numpy as np

import math


def identity_mat(x=None, device="cpu", is_numpy=False):
    if x is not None:
        if isinstance(x, torch.Tensor):
            mat = torch.eye(4, device=device)
            mat = mat.repeat(x.shape[:-2] + (1, 1))
        elif isinstance(x, np.ndarray):
            mat = np.eye(4, dtype=np.float32)
            if x is not None:
                for _ in range(len(x.shape) - 2):
                    mat = mat[None]
            mat = np.tile(mat, x.shape[:-2] + (1, 1))
        else:
            raise ValueError
    else:
        # (4, 4)
        if is_numpy:
            mat = np.eye(4, dtype=np.float32)
        else:
            mat = torch.eye(4, device=device)

    return mat


def vec2mat(vec):
    """위치와 방향을 담은 vec [12]를 world matrix [4, 4]로 변환한다."""
    # batch 크기가 1이라고 가정한다.
    v = np.tile(np.array([[0, 0, 0, 1]]), (1, 1))
    if isinstance(vec, torch.Tensor):
        v = torch.tensor(
            v,
            device=vec.device,
            dtype=vec.dtype,
        )
    pos = vec[:3]
    forward = vec[3:6]
    up = vec[6:9]
    right = vec[9:12]

    if isinstance(vec, torch.Tensor):
        mat_world = torch.stack([right, up, forward, pos], dim=-1)
        mat_world = torch.cat([mat_world, v], dim=-2)
    elif isinstance(vec, np.ndarray):
        mat_world = np.stack([right, up, forward, pos], axis=-1)
        mat_world = np.concatenate([mat_world, v], axis=-2)
    else:
        raise ValueError
    mat_world = normalized_matrix(mat_world)
    return mat_world


def mat2vec(mat):
    """matrix [4, 4]를 위치와 방향을 담은 vec [12]로 변환한다."""
    # batch 크기가 1이라고 가정한다.
    pos = mat[:-1, 3]
    forward = normalized(mat[:-1, 2])
    up = normalized(mat[:-1, 1])
    right = normalized(mat[:-1, 0])
    if isinstance(mat, torch.Tensor):
        vec = torch.cat((pos, forward, up, right))
    elif isinstance(mat, np.ndarray):
        vec = np.concatenate((pos, forward, up, right))
    else:
        raise ValueError

    return vec


def vec2mat_batch(vec):
    """vec batch [B, 12]를 world matrix batch [B, 4, 4]로 변환한다."""

    v = np.tile(np.array([[0, 0, 0, 1]], dtype=np.float32), (vec.shape[0], 1, 1))
    if isinstance(vec, torch.Tensor):
        v = torch.tensor(
            v,
            device=vec.device,
            dtype=vec.dtype,
        )
    pos = vec[..., :3]
    forward = vec[..., 3:6]
    up = vec[..., 6:9]
    right = vec[..., 9:12]
    if isinstance(vec, torch.Tensor):
        mat_world = torch.stack([right, up, forward, pos], dim=-1)
        mat_world = torch.cat([mat_world, v], dim=-2)
    elif isinstance(vec, np.ndarray):
        mat_world = np.stack([right, up, forward, pos], axis=-1)
        mat_world = np.concatenate([mat_world, v], axis=-2)
    else:
        raise ValueError

    mat_world = normalized_matrix(mat_world)
    return mat_world


def rotmat2tan_norm(mat):
    """회전 행렬 [B, 3, 3]을 tangent-normal 표현 [B, 6]으로 변환한다."""
    if isinstance(mat, np.ndarray):
        tan = np.zeros_like(mat[..., 2])
        norm = np.zeros_like(mat[..., 0])
    elif isinstance(mat, torch.Tensor):
        tan = torch.zeros_like(mat[..., 2])
        norm = torch.zeros_like(mat[..., 0])
    else:
        raise ValueError
    tan[...] = mat[..., 2, ::-1]
    tan[..., -1] *= -1
    norm[...] = mat[..., 0, ::-1]
    norm[..., -1] *= -1
    if isinstance(mat, np.ndarray):
        tan_norm = np.concatenate((tan, norm), axis=-1)
    elif isinstance(mat, torch.Tensor):
        tan_norm = torch.cat((tan, norm), dim=-1)
    else:
        raise ValueError
    return tan_norm


def mat2tan_norm(mat):
    """transformation matrix [B, 4, 4]를 tangent-normal 표현 [B, 6]으로 변환한다."""
    rot_mat = mat[..., :-1, :-1]
    return rotmat2tan_norm(rot_mat)


def rotmat2tan_norm(mat):
    """회전 행렬 [B, 3, 3]을 tangent-normal 표현 [B, 6]으로 변환한다."""
    if isinstance(mat, np.ndarray):
        tan = np.zeros_like(mat[..., 2])
        norm = np.zeros_like(mat[..., 0])
        tan[...] = mat[..., 2, ::-1]
        norm[...] = mat[..., 0, ::-1]
    elif isinstance(mat, torch.Tensor):
        tan = torch.zeros_like(mat[..., 2])
        norm = torch.zeros_like(mat[..., 0])
        tan[...] = torch.flip(mat[..., 2], dims=[-1])
        norm[...] = torch.flip(mat[..., 0], dims=[-1])
    else:
        raise ValueError
    tan[..., -1] *= -1
    norm[..., -1] *= -1
    if isinstance(mat, np.ndarray):
        tan_norm = np.concatenate((tan, norm), axis=-1)
    elif isinstance(mat, torch.Tensor):
        tan_norm = torch.cat((tan, norm), dim=-1)
    else:
        raise ValueError
    return tan_norm


def tan_norm2rotmat(tan_norm):
    """tangent-normal 표현 [B, 6]을 회전 행렬 [B, 3, 3]으로 변환한다."""
    tan = copy.deepcopy(tan_norm[..., :3])
    norm = copy.deepcopy(tan_norm[..., 3:])
    tan[..., -1] *= -1
    norm[..., -1] *= -1
    if isinstance(tan_norm, np.ndarray):
        rotmat = np.zeros(tan_norm.shape[:-1] + (3, 3))
        tan = tan[..., ::-1]
        norm = norm[..., ::-1]
        other = np.cross(tan, norm)
    elif isinstance(tan_norm, torch.Tensor):
        rotmat = torch.zeros(tan_norm.shape[:-1] + (3, 3), device=tan_norm.device)
        tan = torch.flip(tan, dims=[-1])
        norm = torch.flip(norm, dims=[-1])
        other = torch.cross(tan, norm)
    else:
        raise ValueError
    rotmat[..., 2, :] = tan
    rotmat[..., 0, :] = norm
    rotmat[..., 1, :] = other
    return rotmat


def rotmat332vec_batch(mat):
    """회전 행렬 [B, 3, 3]을 방향 vector로 변환한다."""
    mat = normalized_matrix(mat)
    forward = mat[..., :, 2]
    up = mat[..., :, 1]
    right = mat[..., :, 0]
    if isinstance(mat, torch.Tensor):
        vec = torch.cat((forward, up, right), dim=-1)
    elif isinstance(mat, np.ndarray):
        vec = np.concatenate((forward, up, right), axis=-1)
    else:
        raise ValueError
    return vec


def rotmat2vec_batch(mat):
    """transformation matrix [B, 4, 4]를 방향 vector [B, 9]로 변환한다."""
    mat = normalized_matrix(mat)
    forward = mat[..., :-1, 2]
    up = mat[..., :-1, 1]
    right = mat[..., :-1, 0]
    if isinstance(mat, torch.Tensor):
        vec = torch.cat((forward, up, right), dim=-1)
    elif isinstance(mat, np.ndarray):
        vec = np.concatenate((forward, up, right), axis=-1)
    else:
        raise ValueError
    return vec


def mat2vec_batch(mat):
    """matrix [B, 4, 4]를 위치와 방향 vector [B, 12]로 변환한다."""
    mat = normalized_matrix(mat)
    pos = mat[..., :-1, 3]
    forward = mat[..., :-1, 2]
    up = mat[..., :-1, 1]
    right = mat[..., :-1, 0]
    if isinstance(mat, torch.Tensor):
        vec = torch.cat((pos, forward, up, right), dim=-1)
    elif isinstance(mat, np.ndarray):
        vec = np.concatenate((pos, forward, up, right), axis=-1)
    else:
        raise ValueError
    return vec


def mat2pose_batch(mat, returnvel=True):
    """matrix [B, 4, 4]를 위치, 방향, 속도 항으로 구성된 pose vector로 변환한다."""
    mat = normalized_matrix(mat)
    pos = mat[..., :-1, 3]
    forward = mat[..., :-1, 2]
    up = mat[..., :-1, 1]
    if isinstance(mat, torch.Tensor):
        if returnvel:
            vel = torch.zeros_like(up)
            vec = torch.cat((pos, forward, up, vel), dim=-1)
        else:
            vec = torch.cat((pos, forward, up), dim=-1)
    elif isinstance(mat, np.ndarray):
        if returnvel:
            vel = np.zeros_like(up)
            vec = np.concatenate((pos, forward, up, vel), axis=-1)
        else:
            vec = np.concatenate((pos, forward, up), axis=-1)
    else:
        raise ValueError
    return vec


def get_mat_BinA(matCtoA, matCtoB):
    """
        같은 객체를 좌표계 A와 B에서 나타낸 matrix로부터
        A 좌표계에서의 B matrix를 반환한다.

    Args:
        matCtoA (tensor): [4, 4] world matrix
        matCtoB (tensor): [4, 4] world matrix
    """
    if isinstance(matCtoA, torch.Tensor):
        matCtoB_inv = torch.inverse(matCtoB)
    elif isinstance(matCtoA, np.ndarray):
        matCtoB_inv = np.linalg.inv(matCtoB)
    else:
        raise ValueError
    matCtoB_inv = normalized_matrix(matCtoB_inv)
    if isinstance(matCtoA, torch.Tensor):
        mat_BtoA = torch.matmul(matCtoA, matCtoB_inv)
    elif isinstance(matCtoA, np.ndarray):
        mat_BtoA = np.matmul(matCtoA, matCtoB_inv)
    mat_BtoA = normalized_matrix(mat_BtoA)
    return mat_BtoA


def get_mat_BtoA(matA, matB):
    """
        A 좌표계에서의 B matrix를 반환한다.

    Args:
        matA (tensor): [4, 4] world matrix
        matB (tensor): [4, 4] world matrix
    """
    if isinstance(matA, torch.Tensor):
        matA_inv = torch.inverse(matA)
    elif isinstance(matA, np.ndarray):
        matA_inv = np.linalg.inv(matA)
    else:
        raise ValueError
    matA_inv = normalized_matrix(matA_inv)
    if isinstance(matA, torch.Tensor):
        mat_BtoA = torch.matmul(matA_inv, matB)
    elif isinstance(matA, np.ndarray):
        mat_BtoA = np.matmul(matA_inv, matB)
    mat_BtoA = normalized_matrix(mat_BtoA)
    return mat_BtoA


def get_mat_BfromA(matA, matBtoA):
    """
        matrix A와 A 기준의 상대 matrix B를 이용해 B의 world matrix를 반환한다.

    Args:
        matA: [4, 4] world matrix
        matBtoA: [4, 4] A 기준의 B matrix
    """
    if isinstance(matA, torch.Tensor):
        matB = torch.matmul(matA, matBtoA)
    if isinstance(matA, np.ndarray):
        matB = np.matmul(matA, matBtoA)
    matB = normalized_matrix(matB)
    return matB


def get_relative_position_to(pos, mat):
    """world 위치를 주어진 matrix 기준의 상대 위치로 변환한다."""
    if isinstance(mat, torch.Tensor):
        mat_inv = torch.inverse(mat)
    elif isinstance(mat, np.ndarray):
        mat_inv = np.linalg.inv(mat)
    else:
        raise ValueError
    mat_inv = normalized_matrix(mat_inv)
    if isinstance(mat, torch.Tensor):
        rot_pos = torch.matmul(mat_inv[..., :-1, :-1], pos.transpose(-1, -2)).transpose(-1, -2)
    elif isinstance(mat, np.ndarray):
        rot_pos = np.matmul(mat_inv[..., :-1, :-1], pos.swapaxes(-1, -2)).swapaxes(-1, -2)
    world_pos = rot_pos + mat_inv[..., None, :-1, 3]
    return world_pos


def get_rotation(mat):
    """transformation matrix에서 rotation을 가져온다."""
    return mat[..., :-1, :-1]


def set_rotation(mat, rotmat):
    """transformation matrix의 rotation을 설정한다."""
    mat[..., :-1, :-1] = rotmat
    return mat


def set_position(mat, pos):
    """transformation matrix의 위치를 설정한다."""
    mat[..., :-1, 3] = pos
    return mat


def get_position(mat):
    """transformation matrix에서 위치를 가져온다."""
    return mat[..., :-1, 3]


def get_position_from(pos, mat):
    """상대 위치를 주어진 matrix 기준의 world 위치로 변환한다."""
    if isinstance(mat, torch.Tensor):
        rot_pos = torch.matmul(mat[..., :-1, :-1], pos.transpose(-1, -2)).transpose(-1, -2)
    elif isinstance(mat, np.ndarray):
        rot_pos = np.matmul(mat[..., :-1, :-1], pos.swapaxes(-1, -2)).swapaxes(-1, -2)
    else:
        raise ValueError

    world_pos = rot_pos + mat[..., None, :-1, 3]
    return world_pos


def get_position_from_rotmat(pos, mat):
    """회전 행렬을 위치 vector에 적용한다."""
    if isinstance(mat, torch.Tensor):
        rot_pos = torch.matmul(mat, pos.transpose(-1, -2)).transpose(-1, -2)
    elif isinstance(mat, np.ndarray):
        rot_pos = np.matmul(mat, pos.swapaxes(-1, -2)).swapaxes(-1, -2)
    else:
        raise ValueError
    return rot_pos


def get_relative_direction_to(dir, mat):
    """world 방향을 주어진 matrix 기준의 상대 방향으로 변환한다."""
    if isinstance(mat, torch.Tensor):
        mat_inv = torch.inverse(mat)
    elif isinstance(mat, np.ndarray):
        mat_inv = np.linalg.inv(mat)
    else:
        raise ValueError
    mat_inv = normalized_matrix(mat_inv)
    rot_mat_inv = mat_inv[..., :3, :3]
    if isinstance(mat, torch.Tensor):
        rel_dir = torch.matmul(rot_mat_inv, dir.transpose(-1, -2))
        return rel_dir.transpose(-1, -2)
    elif isinstance(mat, np.ndarray):
        rel_dir = np.matmul(rot_mat_inv, dir.swapaxes(-1, -2))
        return rel_dir.swapaxes(-1, -2)
    else:
        raise ValueError
    return


def get_direction_from(dir, mat):
    """상대 방향을 주어진 matrix 기준의 world 방향으로 변환한다."""
    rot_mat = mat[..., :3, :3]
    if isinstance(mat, torch.Tensor):
        world_dir = torch.matmul(rot_mat, dir.transpose(-1, -2))
        return world_dir.transpose(-1, -2)
    elif isinstance(mat, np.ndarray):
        world_dir = np.matmul(rot_mat, dir.swapaxes(-1, -2))
        return world_dir.swapaxes(-1, -2)
    else:
        raise ValueError
    return


def get_coord_vis(pos, rot_mat, scale=1.0):
    forward = rot_mat[..., :, 2]
    up = rot_mat[..., :, 1]
    right = rot_mat[..., :, 0]
    return pos + right * scale, pos + up * scale, pos + forward * scale


def project_vec(vec):
    """위치·방향 vector를 xz 평면의 위치와 정면 방향으로 투영한다."""
    posx = vec[..., 0:1]
    posz = vec[..., 2:3]
    forwardx = vec[..., 3:4]
    forwardz = vec[..., 5:6]
    if isinstance(vec, torch.Tensor):
        proj_vec = torch.cat((posx, posz, forwardx, forwardz), dim=-1)
    elif isinstance(vec, np.ndarray):
        proj_vec = np.concatenate((posx, posz, forwardx, forwardz), axis=-1)
    else:
        raise ValueError

    return proj_vec


def xz2xyz(vec):
    x = vec[..., 0:1]
    z = vec[..., 1:2]
    if isinstance(vec, torch.Tensor):
        y = torch.zeros(vec.shape[:-1] + (1,), device=vec.device)
        xyz_vec = torch.cat((x, y, z), dim=-1)
    elif isinstance(vec, np.ndarray):
        y = np.zeros(vec.shape[:-1] + (1,))
        xyz_vec = np.concatenate((x, y, z), axis=-1)
    else:
        raise ValueError

    return xyz_vec


def normalized(vec):
    if isinstance(vec, torch.Tensor):
        norm_vec = vec / (vec.norm(2, dim=-1, keepdim=True) + 1e-9)
    elif isinstance(vec, np.ndarray):
        norm_vec = vec / (np.linalg.norm(vec, ord=2, axis=-1, keepdims=True) + 1e-9)
    else:
        raise ValueError

    return norm_vec


def normalized_matrix(mat):
    if mat.shape[-1] == 4:
        rot_mat = mat[..., :-1, :-1]
    else:
        rot_mat = mat
    if isinstance(mat, torch.Tensor):
        rot_mat_norm = rot_mat / (rot_mat.norm(2, dim=-2, keepdim=True) + 1e-9)
        norm_mat = torch.zeros_like(mat)
    elif isinstance(mat, np.ndarray):
        rot_mat_norm = rot_mat / (np.linalg.norm(rot_mat, ord=2, axis=-2, keepdims=True) + 1e-9)
        norm_mat = np.zeros_like(mat)
    else:
        raise ValueError
    if mat.shape[-1] == 4:
        norm_mat[..., :-1, :-1] = rot_mat_norm
        norm_mat[..., :-1, -1] = mat[..., :-1, -1]
        norm_mat[..., -1, -1] = 1.0
    else:
        norm_mat = rot_mat_norm
    return norm_mat


def get_rot_mat_from_forward(forward):
    """정면 방향 vector에서 회전 행렬을 구한다."""
    if isinstance(forward, torch.Tensor):
        mat = torch.eye(3, device=forward.device).repeat(forward.shape[:-1] + (1, 1))
        right = torch.zeros_like(forward)
    elif isinstance(forward, np.ndarray):
        mat = np.eye(3, dtype=np.float32)
        for _ in range(len(forward.shape) - 1):
            mat = mat[None]
        mat = np.tile(mat, forward.shape[:-1] + (1, 1))
        right = np.zeros_like(forward)
    else:
        raise ValueError

    right[..., 0] = forward[..., 2]
    right[..., 1] = 0.0
    right[..., 2] = -forward[..., 0]
    # right = torch.cross(mat[..., 1], forward)  # backward를 수행할 수 없다.

    mat[..., 2] = normalized(forward)
    right = normalized(right)
    mat[..., 0] = right
    return mat


def get_rot_mat_from_forward_up(forward, up):
    """정면과 위쪽 방향 vector에서 회전 행렬을 구한다."""
    if isinstance(forward, torch.Tensor):
        mat = torch.eye(3, device=forward.device).repeat(forward.shape[:-1] + (1, 1))
        right = torch.cross(up, forward)
    elif isinstance(forward, np.ndarray):
        mat = np.eye(3, dtype=np.float32)
        for _ in range(len(forward.shape) - 1):
            mat = mat[None]
        mat = np.tile(mat, forward.shape[:-1] + (1, 1))
        right = np.cross(up, forward)
    else:
        raise ValueError

    right = normalized(right)
    mat[..., 2] = normalized(forward)
    mat[..., 1] = normalized(up)
    mat[..., 0] = right
    return mat


def get_rot_mat_from_pose_vec(vec):
    """pose vector [N, M, 6]에서 회전 행렬을 구한다."""
    forward = vec[..., :3]
    up = vec[..., 3:6]
    return get_rot_mat_from_forward_up(forward, up)


def get_TRS(rot_mat, pos):
    """rotation과 위치에서 TRS matrix를 만든다."""
    if isinstance(rot_mat, torch.Tensor):
        mat = torch.eye(4, device=pos.device).repeat(pos.shape[:-1] + (1, 1))
    elif isinstance(rot_mat, np.ndarray):
        mat = np.eye(4, dtype=np.float32)
        for _ in range(len(pos.shape) - 1):
            mat = mat[None]
        mat = np.tile(mat, pos.shape[:-1] + (1, 1))
    else:
        raise ValueError
    mat[..., :3, :3] = rot_mat
    mat[..., :3, 3] = pos
    mat = normalized_matrix(mat)
    return mat


def xzvec2mat(vec):
    """xz 평면의 위치·방향 vector를 transformation matrix로 변환한다."""
    vec_shape = vec.shape[:-1]
    if isinstance(vec, torch.Tensor):
        pos = torch.zeros(vec_shape + (3,))
        forward = torch.zeros(vec_shape + (3,))
    elif isinstance(vec, np.ndarray):
        pos = np.zeros(vec_shape + (3,))
        forward = np.zeros(vec_shape + (3,))
    else:
        raise ValueError

    pos[..., 0] = vec[..., 0]
    pos[..., 2] = vec[..., 1]
    forward[..., 0] = vec[..., 2]
    forward[..., 2] = vec[..., 3]
    rot_mat = get_rot_mat_from_forward(forward)
    mat = get_TRS(rot_mat, pos)
    return mat


def distance(vec1, vec2):
    return ((vec1 - vec2) ** 2).sum() ** 0.5


def get_relative_pose_from_vec(pose, root, N):
    root_p_mat = xzvec2mat(root)
    pose = pose.reshape(-1, N, 12)
    pose[..., :3] = get_position_from(pose[..., :3], root_p_mat)
    pose[..., 3:6] = get_direction_from(pose[..., 3:6], root_p_mat)
    pose[..., 6:9] = get_direction_from(pose[..., 6:9], root_p_mat)
    pose[..., 9:] = get_direction_from(pose[..., 9:], root_p_mat)
    pos = pose[..., 0, :3]
    rot = pose[..., 3:9].reshape(-1, N * 6)
    pose = np.concatenate((pos, rot), axis=-1)
    return pose


def get_forward_from_pos(pos):
    """각 frame의 joint 위치 (N, J, 3)에서 정면 방향을 구한다."""

    pos_y_vec = torch.tensor([0, 1, 0], dtype=torch.float32).to(pos.device)
    face_joint_indx = [2, 1, 17, 16]
    r_hip, l_hip, r_sdr, l_sdr = face_joint_indx  # hip과 shoulder로 교차 vector를 구한다.
    cross_hip = pos[..., 0, r_hip, :] - pos[..., 0, l_hip, :]
    cross_sdr = pos[..., 0, r_sdr, :] - pos[..., 0, l_sdr, :]
    cross_vec = cross_hip + cross_sdr  # (3, )
    forward_vec = torch.cross(pos_y_vec, cross_vec, dim=-1)
    forward_vec = normalized(forward_vec)
    return forward_vec


def project_point_along_ray(p, ray, keepnorm=False):
    """point를 ray에 투영하며 keepnorm이면 원래 point 길이를 유지한다."""
    ray = normalized(ray)
    if keepnorm:
        new_p = ray * p.norm(dim=-1, keepdim=True)
    else:
        dot_product = torch.sum(p * ray, dim=-1, keepdim=True)
        new_p = dot_product * ray
    return new_p


def solve_point_along_ray_with_constraint(c, ray, p, constraint="x"):
    """지정 축의 제약값을 만족하는 ray 위 point를 구한다."""
    ray = normalized(ray)
    if constraint == "x":
        ind = 0
    elif constraint == "y":
        ind = 1
    elif constraint == "z":
        ind = 2
    else:
        raise ValueError
    t = (c - p[..., ind]) / ray[..., ind]
    out_p = ray * t[..., None] + p

    return out_p


def calc_cosine(vec1, vec2, return_angle=False):
    """두 vector의 cosine을 구하며 return_angle이면 각도를 반환한다."""
    vec1 = normalized(vec1)
    vec2 = normalized(vec2)
    cosine = torch.sum(vec1 * vec2, dim=-1)
    if return_angle:
        return torch.acos(cosine)
    return cosine


############################################
#
# quaternion은 xyzw 순서를 사용한다.
#
############################################


def quat_xyzw2wxyz(quat):
    new_quat = torch.cat([quat[..., 3:4], quat[..., :3]], dim=-1)
    return new_quat


def quat_wxyz2xyzw(quat):
    new_quat = torch.cat([quat[..., 1:4], quat[..., :1]], dim=-1)
    return new_quat


def quat_mul(a, b):
    """
    quaternion을 곱한다.
    """
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([x, y, z, w], dim=-1)


def quat_pos(x):
    """
    quaternion의 실수부를 양수로 만든다.
    """
    q = x
    z = (q[..., 3:] < 0).float()
    q = (1 - 2 * z) * q
    return q


def quat_abs(x):
    """
    quaternion norm을 구한다. 3D 회전을 나타내는 unit quaternion의 norm은 1이다.
    """
    x = x.norm(p=2, dim=-1)
    return x


def quat_unit(x):
    """
    norm이 1인 정규화 quaternion을 구한다.
    """
    norm = quat_abs(x).unsqueeze(-1)
    return x / (norm.clamp(min=1e-4))


def quat_conjugate(x):
    """
    허수부의 부호를 반전한 quaternion을 구한다.
    """
    return torch.cat([-x[..., :3], x[..., 3:]], dim=-1)


def quat_real(x):
    """
    quaternion의 실수부를 가져온다.
    """
    return x[..., 3]


def quat_imaginary(x):
    """
    quaternion의 허수부를 가져온다.
    """
    return x[..., :3]


def quat_norm_check(x):
    """
    quaternion의 norm이 1인지 확인한다.
    """
    assert bool((abs(x.norm(p=2, dim=-1) - 1) < 1e-3).all()), "the quaternion is has non-1 norm: {}".format(
        abs(x.norm(p=2, dim=-1) - 1)
    )
    assert bool((x[..., 3] >= 0).all()), "the quaternion has negative real part"


def quat_normalize(q):
    """
    정규화되지 않은 quaternion을 정규화해 3D 회전으로 만든다.
    """
    q = quat_unit(quat_pos(q))  # 실수부가 양수인 unit quaternion으로 정규화한다.
    return q


def quat_from_xyz(xyz):
    """
    허수부에서 3D 회전을 구성한다.
    """
    w = (1.0 - xyz.norm()).unsqueeze(-1)
    assert bool((w >= 0).all()), "xyz has its norm greater than 1"
    return torch.cat([xyz, w], dim=-1)


def quat_identity(shape: List[int]):
    """
    주어진 shape의 3D identity 회전을 만든다.
    """
    w = torch.ones(shape + (1,))
    xyz = torch.zeros(shape + (3,))
    q = torch.cat([xyz, w], dim=-1)
    return quat_normalize(q)


def tgm_quat_from_angle_axis(angle, axis, degree: bool = False):
    """회전 각도와 축에서 3D 회전을 만든다. 회전은 축을 따라 반시계 방향이다.

    이 회전은 frame "a"에서 축을 따라 반시계 방향으로 회전해 새 frame "b"가 된
    a_R_b로 해석할 수 있다.

    :param angle: 회전 각도
    :type angle: Tensor
    :param axis: 회전 축
    :type axis: Tensor
    :param degree: 각도가 degree 단위이면 True
    :type degree: bool, 선택, 기본값=False
    """
    if degree:
        angle = angle / 180.0 * math.pi
    theta = (angle / 2).unsqueeze(-1)
    axis = axis / (axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-4))
    xyz = axis * theta.sin()
    w = theta.cos()
    return quat_normalize(torch.cat([w, xyz], dim=-1))


def quat_from_rotation_matrix(m):
    """
    유효한 3x3 회전 행렬에서 3D 회전을 구성한다.
    참고 자료:
    http://www.cg.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche52.html

    :param m: 3x3 직교 회전 행렬
    :type m: Tensor

    :rtype: Tensor
    """
    m = m.unsqueeze(0)
    diag0 = m[..., 0, 0]
    diag1 = m[..., 1, 1]
    diag2 = m[..., 2, 2]

    # 수학적 계산
    w = (((diag0 + diag1 + diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    x = (((diag0 - diag1 - diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    y = (((-diag0 + diag1 - diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    z = (((-diag0 - diag1 + diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5

    # w가 x, y, z보다 큰 quaternion만 수정한다.
    c0 = (w >= x) & (w >= y) & (w >= z)
    x[c0] *= (m[..., 2, 1][c0] - m[..., 1, 2][c0]).sign()
    y[c0] *= (m[..., 0, 2][c0] - m[..., 2, 0][c0]).sign()
    z[c0] *= (m[..., 1, 0][c0] - m[..., 0, 1][c0]).sign()

    # x가 w, y, z보다 큰 quaternion만 수정한다.
    c1 = (x >= w) & (x >= y) & (x >= z)
    w[c1] *= (m[..., 2, 1][c1] - m[..., 1, 2][c1]).sign()
    y[c1] *= (m[..., 1, 0][c1] + m[..., 0, 1][c1]).sign()
    z[c1] *= (m[..., 0, 2][c1] + m[..., 2, 0][c1]).sign()

    # y가 w, x, z보다 큰 quaternion만 수정한다.
    c2 = (y >= w) & (y >= x) & (y >= z)
    w[c2] *= (m[..., 0, 2][c2] - m[..., 2, 0][c2]).sign()
    x[c2] *= (m[..., 1, 0][c2] + m[..., 0, 1][c2]).sign()
    z[c2] *= (m[..., 2, 1][c2] + m[..., 1, 2][c2]).sign()

    # z가 w, x, y보다 큰 quaternion만 수정한다.
    c3 = (z >= w) & (z >= x) & (z >= y)
    w[c3] *= (m[..., 1, 0][c3] - m[..., 0, 1][c3]).sign()
    x[c3] *= (m[..., 2, 0][c3] + m[..., 0, 2][c3]).sign()
    y[c3] *= (m[..., 2, 1][c3] + m[..., 1, 2][c3]).sign()

    return quat_normalize(torch.stack([x, y, z, w], dim=-1)).squeeze(0)


def quat_mul_norm(x, y):
    """
    두 3D 회전 집합을 곱한다. shape은 broadcast가 가능해야 한다.
    """
    return quat_normalize(quat_mul(x, y))


def quat_rotate(rot, vec):
    """
    3D 회전을 3D vector에 적용한다.
    """
    other_q = torch.cat([vec, torch.zeros_like(vec[..., :1])], dim=-1)
    return quat_imaginary(quat_mul(quat_mul(rot, other_q), quat_conjugate(rot)))


def quat_inverse(x):
    """
    회전의 역을 구한다.
    """
    return quat_conjugate(x)


def quat_identity_like(x):
    """
    같은 shape의 3D identity 회전을 만든다.
    """
    return quat_identity(x.shape[:-1])


def quat_angle_axis(x):
    """
    회전을 (angle, axis)로 표현한다. 축은 unit length로 정규화되며 각도는 [0, pi] 범위이다.
    """
    s = 2 * (x[..., 3] ** 2) - 1
    angle = s.clamp(-1, 1).arccos()  # 수치 안정성을 위해 범위를 제한한다.
    axis = x[..., :3]
    axis /= axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-4)
    return angle, axis


def quat_yaw_rotation(x, z_up: bool = True):
    """
    yaw 회전, 즉 z축을 중심으로 한 회전을 구한다.
    """
    q = x
    if z_up:
        q = torch.cat([torch.zeros_like(q[..., 0:2]), q[..., 2:3], q[..., 3:]], dim=-1)
    else:
        q = torch.cat(
            [
                torch.zeros_like(q[..., 0:1]),
                q[..., 1:2],
                torch.zeros_like(q[..., 2:3]),
                q[..., 3:4],
            ],
            dim=-1,
        )
    return quat_normalize(q)


def transform_from_rotation_translation(r: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None):
    """
    quaternion과 3D translation으로 transform을 만든다. 둘 중 하나만 None일 수 있다.
    """
    assert r is not None or t is not None, "rotation and translation can't be all None"
    if r is None:
        assert t is not None
        r = quat_identity(list(t.shape))
    if t is None:
        t = torch.zeros(list(r.shape) + [3])
    return torch.cat([r, t], dim=-1)


def transform_identity(shape: List[int]):
    """
    주어진 shape의 identity transformation을 만든다.
    """
    r = quat_identity(shape)
    t = torch.zeros(shape + [3])
    return transform_from_rotation_translation(r, t)


def transform_rotation(x):
    """transform에서 rotation을 가져온다."""
    return x[..., :4]


def transform_translation(x):
    """transform에서 translation을 가져온다."""
    return x[..., 4:]


def transform_inverse(x):
    """
    transformation의 역을 구한다.
    """
    inv_so3 = quat_inverse(transform_rotation(x))
    return transform_from_rotation_translation(r=inv_so3, t=quat_rotate(inv_so3, -transform_translation(x)))


def transform_identity_like(x):
    """
    같은 shape의 identity transformation을 만든다.
    """
    return transform_identity(x.shape)


def transform_mul(x, y):
    """
    두 transformation을 합성한다.
    """
    z = transform_from_rotation_translation(
        r=quat_mul_norm(transform_rotation(x), transform_rotation(y)),
        t=quat_rotate(transform_rotation(x), transform_translation(y)) + transform_translation(x),
    )
    return z


def transform_apply(rot, vec):
    """
    transformation을 3D vector에 적용한다.
    """
    assert isinstance(vec, torch.Tensor)
    return quat_rotate(transform_rotation(rot), vec) + transform_translation(rot)


def rot_matrix_det(x):
    """
    3x3 matrix의 determinant를 반환한다. tensor shape은 matrix의 batch shape을 따른다.
    """
    a, b, c = x[..., 0, 0], x[..., 0, 1], x[..., 0, 2]
    d, e, f = x[..., 1, 0], x[..., 1, 1], x[..., 1, 2]
    g, h, i = x[..., 2, 0], x[..., 2, 1], x[..., 2, 2]
    t1 = a * (e * i - f * h)
    t2 = b * (d * i - f * g)
    t3 = c * (d * h - e * g)
    return t1 - t2 + t3


def rot_matrix_integrity_check(x):
    """
    회전 행렬의 determinant가 1이고 직교하는지 확인한다.
    """
    det = rot_matrix_det(x)
    assert bool((abs(det - 1) < 1e-3).all()), "the matrix has non-one determinant"
    rtr = x @ x.permute(torch.arange(x.dim() - 2), -1, -2)
    rtr_gt = rtr.zeros_like()
    rtr_gt[..., 0, 0] = 1
    rtr_gt[..., 1, 1] = 1
    rtr_gt[..., 2, 2] = 1
    assert bool(((rtr - rtr_gt) < 1e-3).all()), "the matrix is not orthogonal"


def rot_matrix_from_quaternion(q):
    """
    quaternion에서 회전 행렬을 만든다.
    """
    # Wikipedia 표기법에 따른 각 원소의 축약 표현
    qi, qj, qk, qr = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # 각 원소를 설정한다.
    R00 = 1.0 - 2.0 * (qj**2 + qk**2)
    R01 = 2 * (qi * qj - qk * qr)
    R02 = 2 * (qi * qk + qj * qr)
    R10 = 2 * (qi * qj + qk * qr)
    R11 = 1.0 - 2.0 * (qi**2 + qk**2)
    R12 = 2 * (qj * qk - qi * qr)
    R20 = 2 * (qi * qk - qj * qr)
    R21 = 2 * (qj * qk + qi * qr)
    R22 = 1.0 - 2.0 * (qi**2 + qj**2)

    R0 = torch.stack([R00, R01, R02], dim=-1)
    R1 = torch.stack([R10, R11, R12], dim=-1)
    R2 = torch.stack([R20, R21, R22], dim=-1)

    R = torch.stack([R0, R1, R2], dim=-2)

    return R


def euclidean_to_rotation_matrix(x):
    """
    Euclidean transformation matrix의 왼쪽 위에서 회전 행렬을 가져온다.
    """
    return x[..., :3, :3]


def euclidean_integrity_check(x):
    euclidean_to_rotation_matrix(x)  # 3D 회전 행렬을 확인한다.
    assert bool((x[..., 3, :3] == 0).all()), "the last row is illegal"
    assert bool((x[..., 3, 3] == 1).all()), "the last row is illegal"


def euclidean_translation(x):
    """
    matrix 마지막 열의 translation vector를 가져온다.
    """
    return x[..., :3, 3]


def euclidean_inverse(x):
    """
    역회전을 나타내는 matrix를 계산한다.
    """
    s = x.zeros_like()
    irot = quat_inverse(quat_from_rotation_matrix(x))
    s[..., :3, :3] = irot
    s[..., :3, 4] = quat_rotate(irot, -euclidean_translation(x))
    return s


def euclidean_to_transform(transformation_matrix):
    """
    Euclidean transformation matrix에서 transform을 만든다.
    """
    return transform_from_rotation_translation(
        r=quat_from_rotation_matrix(m=euclidean_to_rotation_matrix(transformation_matrix)),
        t=euclidean_translation(transformation_matrix),
    )


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


def quat_unit(a):
    return normalize(a)


def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * torch.sin(theta.clone())
    w = torch.cos(theta.clone())
    return quat_unit(torch.cat([xyz, w], dim=-1))


def normalize_angle(x):
    return torch.atan2(torch.sin(x.clone()), torch.cos(x.clone()))


def tf_inverse(q, t):
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


def tf_apply(q, t, v):
    return quat_apply(q, v) + t


def tf_vector(q, v):
    return quat_apply(q, v)


def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


def get_basis_vector(q, v):
    return quat_rotate(q, v)


def get_axis_params(value, axis_idx, x_value=0.0, dtype=float, n_dims=3):
    """축 index에 맞춰 `Vec` 인자를 구성한다."""
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = np.where(zs == 1.0, value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll(x축 회전)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch(y축 회전)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw(z축 회전)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)


def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def torch_random_dir_2(shape, device):
    # type: (Tuple[int, int], str) -> Tensor
    angle = torch_rand_float(-np.pi, np.pi, shape, device).squeeze(-1)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # quaternion q에서 axis-angle 표현을 계산한다.
    # q는 정규화되어 있어야 한다.
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


def angle_axis_to_exp_map(angle, axis):
    # type: (Tensor, Tensor) -> Tensor
    # axis-angle에서 exponential map을 계산한다.
    angle_expand = angle.unsqueeze(-1)
    exp_map = angle_expand * axis
    return exp_map


def quat_to_exp_map(q):
    # type: (Tensor) -> Tensor
    # quaternion에서 exponential map을 계산한다.
    # q는 정규화되어 있어야 한다.
    angle, axis = quat_to_angle_axis(q)
    exp_map = angle_axis_to_exp_map(angle, axis)
    return exp_map


def quat_to_tan_norm(q):
    # type: (Tensor) -> Tensor
    # 회전을 tangent와 normal vector로 표현한다.
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


def euler_xyz_to_exp_map(roll, pitch, yaw):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    q = quat_from_euler_xyz(roll, pitch, yaw)
    exp_map = quat_to_exp_map(q)
    return exp_map


def exp_map_to_angle_axis(exp_map):
    min_theta = 1e-5

    angle = torch.norm(exp_map.clone(), dim=-1) + 1e-6
    angle_exp = torch.unsqueeze(angle, dim=-1)
    axis = exp_map.clone() / angle_exp.clone()
    angle = normalize_angle(angle)

    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = torch.abs(angle) > min_theta
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)

    return angle, axis


def exp_map_to_quat(exp_map):
    angle, axis = exp_map_to_angle_axis(exp_map)
    q = quat_from_angle_axis(angle, axis)
    return q


def slerp(q0, q1, t):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q


def calc_heading_vec(q, head_ind=0):
    # type: (Tensor, int) -> Tensor
    # quaternion에서 heading 방향을 계산한다.
    # heading은 방향 vector이다.
    # q는 정규화되어 있어야 한다.
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., head_ind] = 1
    rot_dir = quat_rotate(q, ref_dir)

    return rot_dir


def calc_heading(q, head_ind=0, gravity_axis="z"):
    # type: (Tensor, int, str) -> Tensor
    # quaternion에서 heading 방향을 계산한다.
    # heading은 xy 평면의 방향이다.
    # q는 정규화되어 있어야 한다.
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., head_ind] = 1
    # ref_dir[..., 0] = 1
    shape = ref_dir.shape[:-1]
    q = q.reshape((-1, 4))
    ref_dir = ref_dir.reshape(-1, 3)
    rot_dir = quat_rotate(q, ref_dir)
    rot_dir = rot_dir.reshape(shape + (3,))
    if gravity_axis == "z":
        heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    elif gravity_axis == "y":
        heading = torch.atan2(rot_dir[..., 0], rot_dir[..., 2])
    elif gravity_axis == "x":
        heading = torch.atan2(rot_dir[..., 2], rot_dir[..., 1])
    return heading


def calc_heading_quat(q, head_ind=0, gravity_axis="z"):
    # type: (Tensor, int, str) -> Tensor
    # quaternion에서 heading 회전을 계산한다.
    # heading은 xy 평면의 방향이다.
    # q는 정규화되어 있어야 한다.
    heading = calc_heading(q, head_ind, gravity_axis=gravity_axis)
    axis = torch.zeros_like(q[..., 0:3])
    if gravity_axis == "z":
        g_axis = 2
    elif gravity_axis == "y":
        g_axis = 1
    elif gravity_axis == "x":
        g_axis = 0
    axis[..., g_axis] = 1

    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q


def calc_heading_quat_inv(q, head_ind=0):
    # type: (Tensor, int) -> Tensor
    # quaternion에서 heading 역회전을 계산한다.
    # heading은 xy 평면의 방향이다.
    # q는 정규화되어 있어야 한다.
    heading = calc_heading(q, head_ind)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis)
    return heading_q


def forward_kinematics(mat, parent):
    """local matrix와 parent index로 forward kinematics를 계산한다."""
    if isinstance(mat, torch.Tensor):
        rotations = torch.eye(mat.shape[-1], device=mat.device)
        rotations = rotations.repeat(mat.shape[:-2] + (1, 1))
    else:
        rotations = np.eye(mat.shape[-1], dtype=np.float32)
        rotations = np.tile(rotations, mat.shape[:-2] + (1, 1))
    for i in range(mat.shape[-3]):
        if parent[i] != -1:
            if isinstance(mat, torch.Tensor):
                # gradient가 흐르도록 새 tensor를 조합한다.
                new_mat = get_mat_BfromA(rotations[..., parent[i], :, :], mat[..., i, :, :])
                rotations = torch.cat(
                    (
                        rotations[..., :i, :, :],
                        new_mat[..., None, :, :],
                        rotations[..., i + 1 :, :, :],
                    ),
                    dim=-3,
                )
            else:
                rotations[..., i, :, :] = get_mat_BfromA(rotations[..., parent[i], :, :], mat[..., i, :, :])
        else:
            if isinstance(mat, torch.Tensor):
                # gradient가 흐르도록 새 tensor를 조합한다.
                rotations = torch.cat((mat[..., : i + 1, :, :], rotations[..., i + 1 :, :, :]), dim=-3)
            else:
                rotations[..., i, :, :] = mat[..., i, :, :]
    return rotations
