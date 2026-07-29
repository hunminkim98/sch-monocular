import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix
from smplx.utils import Struct, to_np, to_tensor
from hmr4d.utils.smplx_utils import forward_kinematics_motion


class MinimalLBS(nn.Module):
    def __init__(self, sp_ids, bm_dir='models/smplh', num_betas=16, model_type='smplh', **kwargs):
        super().__init__()
        self.num_betas = num_betas
        self.sensor_point_vid = torch.tensor(sp_ids)

        # 미리 정의한 sensor point에 해당하는 구조 데이터를 불러온다.
        self.load_struct_on_sp(f'{bm_dir}/male/model.npz', prefix='male')
        self.load_struct_on_sp(f'{bm_dir}/female/model.npz', prefix='female')

    def load_struct_on_sp(self, bm_path, prefix='m'):
        """
        body model 구조에서 네 종류의 weight를 불러온다.
        sensor point만 남기고 prefix로 body model을 구분한다.
        """
        num_betas = self.num_betas
        sp_vid = self.sensor_point_vid
        # 데이터를 불러온다.
        data_struct = Struct(**np.load(bm_path, encoding='latin1'))

        # vertex template
        v_template = to_tensor(to_np(data_struct.v_template))  # (V, 3)
        v_template_sp = v_template[sp_vid]  # (N, 3)
        self.register_buffer(f'{prefix}_v_template_sp', v_template_sp, False)

        # shape direction
        shapedirs = to_tensor(to_np(data_struct.shapedirs[:, :, :num_betas]))  # (V, 3, NB)
        shapedirs_sp = shapedirs[sp_vid]
        self.register_buffer(f'{prefix}_shapedirs_sp', shapedirs_sp, False)

        # pose direction
        posedirs = to_tensor(to_np(data_struct.posedirs))  # (V, 3, 51*9)
        posedirs_sp = posedirs[sp_vid]
        posedirs_sp = posedirs_sp.reshape(len(sp_vid)*3, -1).T  # (51*9, N*3)
        self.register_buffer(f'{prefix}_posedirs_sp', posedirs_sp, False)

        # LBS weight
        lbs_weights = to_tensor(to_np(data_struct.weights))  # (V, J+1)
        lbs_weights_sp = lbs_weights[sp_vid]
        self.register_buffer(f'{prefix}_lbs_weights_sp', lbs_weights_sp, False)

    def forward(self, root_orient=None, pose_body=None, trans=None, betas=None, A=None, recompute_A=False, genders=None,
                joints_zero=None):
        """
        인자:
            root_orient, Optional: (B, T, 3)
            pose_body: (B, T, J*3)
            trans: (B, T, 3)
            betas: (B, T, 16)
            A, Optional: (B, T, J+1, 4, 4)
            recompute_A: True이면 root_orient가 필요하고, 아니면 A를 사용한다.
            genders, List: ['male', 'female', ...]
            joints_zero: (B, J+1, 3), recompute_A가 True일 때 필요하다.
        반환:
            sensor_verts: (B, T, N, 3)
        """
        B, T = pose_body.shape[:2]

        v_template = torch.stack([getattr(self, f'{g}_v_template_sp') for g in genders])  # (B, N, 3)
        shapedirs = torch.stack([getattr(self, f'{g}_shapedirs_sp') for g in genders])  # (B, N, 3, NB)
        posedirs = torch.stack([getattr(self, f'{g}_posedirs_sp') for g in genders])  # (B, 51*9, N*3)
        lbs_weights = torch.stack([getattr(self, f'{g}_lbs_weights_sp') for g in genders])  # (B, N, J+1)

        # ===== LBS에서 T 차원 처리 ===== #
        # 2. shape 기여분을 더한다.
        if betas.shape[1] == 1:
            betas = betas.expand(-1, T, -1)
        blend_shape = torch.einsum('btl,bmkl->btmk', [betas, shapedirs])
        v_shaped = v_template[:, None] + blend_shape

        # 3. pose blend shape을 더한다.
        ident = torch.eye(3).to(pose_body)
        aa = pose_body.reshape(B, T, -1, 3)
        R = axis_angle_to_matrix(aa)
        pose_feature = (R - ident).view(B, T, -1)
        dim_pf = pose_feature.shape[-1]
        # (B, T, P) @ (B, P, N*3) -> (B, T, N, 3)
        pose_offsets = torch.matmul(pose_feature, posedirs[:, :dim_pf]).view(B, T, -1, 3)
        v_posed = pose_offsets + v_shaped

        # 4. A를 계산한다.
        if recompute_A:
            _, _, A = forward_kinematics_motion(root_orient, pose_body, trans, joints_zero)

        # 5. Skinning을 적용한다.
        W = lbs_weights
        # (B, 1, N, J+1)) @ (B, T, J+1, 16)
        num_joints = A.shape[-3]  # 22
        Ts = torch.matmul(W[:, None, :, :num_joints], A.view(B, T, num_joints, 16))
        Ts = Ts.view(B, T, -1, 4, 4)  # (B, T, N, 4, 4)
        v_posed_homo = F.pad(v_posed, (0, 1), value=1)  # (B, T, N, 4)
        v_homo = torch.matmul(Ts, torch.unsqueeze(v_posed_homo, dim=-1))

        # 6. Translation을 적용한다.
        sensor_verts = v_homo[:, :, :, :3, 0] + trans[:, :, None]

        return sensor_verts
