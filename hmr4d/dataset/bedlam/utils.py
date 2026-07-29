import torch
from pathlib import Path

resource_dir = Path(__file__).parent / "resource"


def mid2vname(mid):
    """``vname = {scene}/{seq}`` 형식으로 바꿉니다. 확장자는 .mp4입니다."""
    # mid 예: "inputs/bedlam/bedlam_download/20221011_1_250_batch01hand_closeup_suburb_a/mp4/seq_000001.mp4-rp_emma_posed_008"
    # -> vname: 20221011_1_250_batch01hand_closeup_suburb_a/seq_000001.mp4
    scene = mid.split("/")[-3]
    seq = mid.split("/")[-1].split("-")[0]
    vname = f"{scene}/{seq}"
    return vname


def mid2featname(mid):
    """``featname = {scene}/{seqsubj}`` 형식으로 바꿉니다. 확장자는 .pt입니다."""
    # mid 예: "inputs/bedlam/bedlam_download/20221011_1_250_batch01hand_closeup_suburb_a/mp4/seq_000001.mp4-rp_emma_posed_008"
    # -> featname: 20221011_1_250_batch01hand_closeup_suburb_a/seq_000001.mp4-rp_emma_posed_008.pt
    scene = mid.split("/")[-3]
    seqsubj = mid.split("/")[-1]
    featname = f"{scene}/{seqsubj}.pt"
    return featname


def featname2mid(featname):
    """mid2featname의 역변환을 수행하며 .pt 확장자를 제거합니다."""
    # featname 예: 20221011_1_250_batch01hand_closeup_suburb_a/seq_000001.mp4-rp_emma_posed_008.pt
    # -> mid: inputs/bedlam/bedlam_download/20221011_1_250_batch01hand_closeup_suburb_a/mp4/seq_000001.mp4-rp_emma_posed_008
    scene = featname.split("/")[0]
    seqsubj = featname.split("/")[1].strip(".pt")
    mid = f"inputs/bedlam/bedlam_download/{scene}/mp4/{seqsubj}"
    return mid


def load_vname2lwh():
    return torch.load(resource_dir / "vname2lwh.pt")
