from turtle import forward
import numpy as np

import torch
import torch.nn as nn

from smplx import SMPL, SMPLH, SMPLX
from smplx.vertex_ids import vertex_ids
from smplx.utils import Struct


class BodyModel(nn.Module):
    """ 
    SMPL-X body model class의 wrapper.
    Zehong Shen이 수정했다.
    """

    def __init__(self,
                 bm_path,
                 num_betas=16,
                 use_vtx_selector=False,
                 model_type='smplh'):
        super().__init__()
        '''
        주어진 경로에서 body model 객체를 만든다.

        :param bm_path: body model pkl 파일 경로
        :param model_type: [smpl, smplh, smplx] 중 하나
        :param use_vtx_selector: True이면 OpenPose joint에 해당하는 vertex를 추가 joint로 반환한다.
        '''
        self.use_vtx_selector = use_vtx_selector
        cur_vertex_ids = None
        if self.use_vtx_selector:
            cur_vertex_ids = vertex_ids[model_type]
        data_struct = None
        if '.npz' in bm_path:
            # smplx는 기본적으로 .npz를 지원하지 않으므로 직접 불러온다.
            smpl_dict = np.load(bm_path, encoding='latin1')
            data_struct = Struct(**smpl_dict)
            # print(smpl_dict.files)
            if model_type == 'smplh':
                data_struct.hands_componentsl = np.zeros((0))
                data_struct.hands_componentsr = np.zeros((0))
                data_struct.hands_meanl = np.zeros((15 * 3))
                data_struct.hands_meanr = np.zeros((15 * 3))
                V, D, B = data_struct.shapedirs.shape
                data_struct.shapedirs = np.concatenate([data_struct.shapedirs, np.zeros(
                    (V, D, SMPL.SHAPE_SPACE_DIM-B))], axis=-1)  # SMPL-H가 16차원 beta를 쓰도록 맞춘 임시 처리
        kwargs = {
            'model_type': model_type,
            'data_struct': data_struct,
            'num_betas': num_betas,
            'vertex_ids': cur_vertex_ids,
            'use_pca': False,
            'flat_hand_mean': True,
            # module 변수가 필요 없으므로 가변 batch 크기를 허용한다.
            'create_body_pose': False,
            'create_betas': False,
            'create_global_orient': False,
            'create_transl': False,
            'create_left_hand_pose': False,
            'create_right_hand_pose': False,
        }
        assert(model_type in ['smpl', 'smplh', 'smplx'])
        if model_type == 'smpl':
            self.bm = SMPL(bm_path, **kwargs)
            self.num_joints = SMPL.NUM_JOINTS
        elif model_type == 'smplh':
            self.bm = SMPLH(bm_path, **kwargs)
            self.num_joints = SMPLH.NUM_JOINTS
        elif model_type == 'smplx':
            self.bm = SMPLX(bm_path, **kwargs)
            self.num_joints = SMPLX.NUM_JOINTS

        self.model_type = model_type

    def forward(self, root_orient=None, pose_body=None, pose_hand=None, pose_jaw=None, pose_eye=None, betas=None,
                trans=None, dmpls=None, expression=None, return_dict=False, **kwargs):
        '''
        dmpls는 지원하지 않는다.
        '''
        assert(dmpls is None)
        B = pose_body.shape[0]
        if pose_hand is None:
            pose_hand = torch.zeros((B, 2*SMPLH.NUM_HAND_JOINTS*3), device=pose_body.device)
        if len(betas.shape) == 1:
            betas = betas.reshape((1, -1)).expand(B, -1)

        out_obj = self.bm(
            betas=betas,
            global_orient=root_orient,
            body_pose=pose_body,
            left_hand_pose=pose_hand[:, :(SMPLH.NUM_HAND_JOINTS*3)],
            right_hand_pose=pose_hand[:, (SMPLH.NUM_HAND_JOINTS*3):],
            transl=trans,
            expression=expression,
            jaw_pose=pose_jaw,
            leye_pose=None if pose_eye is None else pose_eye[:, :3],
            reye_pose=None if pose_eye is None else pose_eye[:, 3:],
            return_full_pose=True,
            **kwargs
        )

        out = {
            'v': out_obj.vertices,
            'f': self.bm.faces_tensor,
            'Jtr': out_obj.joints,
        }

        if not self.use_vtx_selector:
            # 추가 joint는 필요 없다.
            out['Jtr'] = out['Jtr'][:, :self.num_joints+1]  # root를 위해 하나를 추가한다.

        if not return_dict:
            out = Struct(**out)

        return out

    def forward_motion(self, **kwargs):
        B, W, _ = kwargs['pose_body'].shape
        kwargs = {k: v.reshape(B*W, v.shape[-1]) for k, v in kwargs.items()}

        smpl_opt = self.forward(**kwargs)
        smpl_opt.v = smpl_opt.v.reshape(B, W, -1, 3)
        smpl_opt.Jtr = smpl_opt.Jtr.reshape(B, W, -1, 3)

        return smpl_opt
