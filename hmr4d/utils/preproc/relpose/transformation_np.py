import numpy as np


def rotation_matrix_to_quaternion(R):
    """3x3 rotation matrix R을 quaternion [w, x, y, z]로 변환합니다."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S = 4 * qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S = 4 * qx
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S = 4 * qy
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S = 4 * qz
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


def quaternion_to_rotation_matrix(q):
    """quaternion [w, x, y, z]를 3x3 rotation matrix로 변환합니다."""
    qw, qx, qy, qz = q
    R = np.array(
        [
            [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
        ]
    )
    return R


def slerp(q0, q1, t):
    """
    두 quaternion q0와 q1을 구면 선형 보간(SLERP)합니다.

    인자:
        q0, q1: quaternion [w, x, y, z]를 나타내는 (4,) numpy array
        t: 0 <= t <= 1인 보간 계수

    반환:
        보간된 (4,) quaternion
    """
    dot = np.dot(q0, q1)
    # dot product가 음수이면 부호를 바꿔 짧은 경로를 선택합니다.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # 두 quaternion이 매우 가까우면 선형 보간 후 정규화합니다.
        result = q0 + t * (q1 - q0)
        result = result / np.linalg.norm(result)
        return result

    theta_0 = np.arccos(dot)  # 두 quaternion 사이의 각도
    theta = theta_0 * t  # 보간된 각도
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return (s0 * q0) + (s1 * q1)


def lerp_missing_frames(T_w2c_list, sample_idxs):
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
