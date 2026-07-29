import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle
import torch
import joblib

RESOURCE_FOLDER = Path(__file__).resolve().parent / "resource"


def read_raw_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")

    num_subjects = len(data[b"poses"])
    F = data[b"poses"][0].shape[0]
    smpl_params = []
    for i in range(num_subjects):
        smpl_params.append(
            {
                "body_pose": torch.from_numpy(data[b"poses"][i][:, 3:72]).float(),  # (F, 69)
                "betas": torch.from_numpy(data[b"betas"][i][:10]).repeat(F, 1).float(),  # (F, 10)
                "global_orient": torch.from_numpy(data[b"poses"][i][:, :3]).float(),  # (F, 3)
                "transl": torch.from_numpy(data[b"trans"][i]).float(),  # (F, 3)
            }
        )
    genders = ["male" if g == "m" else "female" for g in data[b"genders"]]
    campose_valid = [torch.from_numpy(v).bool() for v in data[b"campose_valid"]]

    seq_name = data[b"sequence"]
    K_fullimg = torch.from_numpy(data[b"cam_intrinsics"]).float()
    T_w2c = torch.from_numpy(data[b"cam_poses"]).float()

    return_data = {
        "sequence": seq_name,  # 'courtyard_bodyScannerMotions_00'
        "K_fullimg": K_fullimg,  # (3, 3), 55도 FoV가 아닙니다.
        "T_w2c": T_w2c,  # (F, 4, 4)
        "smpl_params": smpl_params,  # dictionary 목록
        "genders": genders,  # 문자열 목록
        "campose_valid": campose_valid,  # bool 배열 목록
        # "jointPositions": data[b'jointPositions'],  # SMPL, 24x3
        # "poses2d": data[b"poses2d"],  # COCO, 3x18(?)
    }
    return return_data


def load_and_convert_wham_pth(pth):
    """
    ``{vid: DataDict}`` 형식으로 변환하고 smpl_params_incam을 추가합니다.
    """
    # 원본 label을 불러옵니다.
    wham_labels_raw = joblib.load(pth)
    # {vid: DataDict} 형식으로 변환합니다.
    wham_labels = {}
    for i, vid in enumerate(wham_labels_raw["vid"]):
        wham_labels[vid] = {k: wham_labels_raw[k][i] for k in wham_labels_raw}

    # pose와 beta를 translation 없는 smpl_params_incam으로 변환합니다.
    for vid in wham_labels:
        pose = wham_labels[vid]["pose"]
        global_orient = pose[:, :3]  # (F, 3)
        body_pose = pose[:, 3:]  # (F, 69)
        betas = wham_labels[vid]["betas"]  # (F, 10), 모든 frame에서 같습니다.
        wham_labels[vid]["smpl_params_incam"] = {
            "body_pose": body_pose.float(),  # (F, 69)
            "betas": betas.float(),  # (F, 10)
            "global_orient": global_orient.float(),  # (F, 3)
        }

    return wham_labels


# Neural-Annot 유틸리티


def na_cam_param_to_K_fullimg(cam_param):
    K = torch.eye(3)
    K[[0, 1], [0, 1]] = torch.tensor(cam_param["focal"])
    K[[0, 1], [2, 2]] = torch.tensor(cam_param["princpt"])
    return K
