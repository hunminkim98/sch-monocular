import torch
import numpy as np

# def get_frame_id_list_from_mask(mask):
#     """
#     Args:
#         mask (F,), bool.
#     Return:
#         frame_id_list: List of frame_ids.
#     """
#     frame_id_list = []
#     i = 0
#     while i < len(mask):
#         if not mask[i]:
#             i += 1
#         else:
#             j = i
#             while j < len(mask) and mask[j]:
#                 j += 1
#             frame_id_list.append(torch.arange(i, j))
#             i = j

#     return frame_id_list


# GPT가 생성한 vectorized 구현
def get_frame_id_list_from_mask(mask):
    # batch=64, 0.13s
    """
    boolean mask에서 연속된 frame ID list를 vectorized 방식으로 구합니다.

    인자:
        mask (F,), bool tensor: True인 위치의 frame을 처리합니다.

    반환:
        frame_id_list: mask가 True인 연속 index를 담은 torch.Tensor list
    """
    # mask가 False에서 True로, 또는 True에서 False로 바뀌는 index를 찾습니다.
    padded_mask = torch.cat(
        [torch.tensor([False], device=mask.device), mask, torch.tensor([False], device=mask.device)]
    )
    diffs = torch.diff(padded_mask.int())
    starts = (diffs == 1).nonzero(as_tuple=False).squeeze()
    ends = (diffs == -1).nonzero(as_tuple=False).squeeze()
    if starts.numel() == 0:
        return []
    if starts.numel() == 1:
        starts = starts.reshape(-1)
        ends = ends.reshape(-1)

    # 연속 구간 list를 만듭니다.
    frame_id_list = [torch.arange(start, end) for start, end in zip(starts, ends)]
    return frame_id_list


def get_batch_frame_id_lists_from_mask_BLC(masks):
    # batch=64, 0.10s
    """
    3D mask에서 batch와 channel별로 연속된 True 구간의 index list를 구합니다.

    인자:
        masks (B, L, C), bool tensor: True인 위치의 frame을 처리합니다.

    반환:
        batch_frame_id_lists: batch와 channel별 frame ID를 담은 중첩 list
    """
    B, L, C = masks.size()
    # sequence 양 끝에 False를 하나씩 추가합니다.
    padded_masks = torch.cat(
        [
            torch.zeros((B, 1, C), dtype=torch.bool, device=masks.device),
            masks,
            torch.zeros((B, 1, C), dtype=torch.bool, device=masks.device),
        ],
        dim=1,
    )
    # 차분으로 True 구간의 시작점과 끝점을 찾습니다.
    diffs = torch.diff(padded_masks.int(), dim=1)
    starts = (diffs == 1).nonzero(as_tuple=True)
    ends = (diffs == -1).nonzero(as_tuple=True)

    # 반환 list를 초기화합니다.
    batch_frame_id_lists = [[[] for _ in range(C)] for _ in range(B)]
    for b in range(B):
        for c in range(C):
            batch_start = starts[0][(starts[0] == b) & (starts[2] == c)]
            batch_end = ends[0][(ends[0] == b) & (ends[2] == c)]
            # start와 end가 모두 1D tensor인지 확인합니다.
            batch_frame_id_lists[b][c] = [
                torch.arange(start.item(), end.item()) for start, end in zip(batch_start, batch_end)
            ]

    return batch_frame_id_lists


def get_frame_id_list_from_frame_id(frame_id):
    mask = torch.zeros(frame_id[-1] + 1, dtype=torch.bool)
    mask[frame_id] = True
    frame_id_list = get_frame_id_list_from_mask(mask)
    return frame_id_list


def rearrange_by_mask(x, mask):
    """
    x (L, *)
    mask (M,), M >= L
    """
    M = mask.size(0)
    L = x.size(0)
    if M == L:
        return x
    assert M > L
    assert mask.sum() == L
    x_rearranged = torch.zeros((M, *x.size()[1:]), dtype=x.dtype, device=x.device)
    x_rearranged[mask] = x
    return x_rearranged


def frame_id_to_mask(frame_id, max_len):
    mask = torch.zeros(max_len, dtype=torch.bool)
    mask[frame_id] = True
    return mask


def mask_to_frame_id(mask):
    frame_id = torch.where(mask)[0]
    return frame_id


def linear_interpolate_frame_ids(data, frame_id_list):
    data = data.clone()
    for i, invalid_frame_ids in enumerate(frame_id_list):
        # 이전 값과 다음 값 사이를 보간합니다.
        # 처음 또는 끝 구간이면 가까운 값을 그대로 사용합니다.
        if invalid_frame_ids[0] - 1 < 0 or invalid_frame_ids[-1] + 1 >= len(data):
            if invalid_frame_ids[0] - 1 < 0:
                data[invalid_frame_ids] = data[invalid_frame_ids[-1] + 1].clone()
            else:
                data[invalid_frame_ids] = data[invalid_frame_ids[0] - 1].clone()
        else:
            prev = data[invalid_frame_ids[0] - 1]
            next = data[invalid_frame_ids[-1] + 1]
            data[invalid_frame_ids] = (
                torch.linspace(0, 1, len(invalid_frame_ids) + 2)[1:-1][:, None] * (next - prev)[None] + prev[None]
            )
    return data


def linear_interpolate(data, N_middle_frames):
    """
    인자:
        data: (2, C)
    반환:
        data_interpolated: (1+N+1, C)
    """
    prev = data[0]
    next = data[1]
    middle = torch.linspace(0, 1, N_middle_frames + 2)[1:-1][:, None] * (next - prev)[None] + prev[None]  # (N, C)
    data_interpolated = torch.cat([data[0][None], middle, data[1][None]], dim=0)  # (1+N+1, C)
    return data_interpolated


def find_top_k_span(mask, k=3):
    """
    인자:
        mask: (L,)
    반환:
        topk_span: [start, end) 형식의 tuple list
    """
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if mask.sum() == 0:
        return []
    mask = mask.clone().float()
    mask = torch.cat([mask.new([0]), mask, mask.new([0])])
    diff = mask[1:] - mask[:-1]
    start = torch.where(diff == 1)[0]
    end = torch.where(diff == -1)[0]
    assert len(start) == len(end)
    span_lengths = end - start
    span_lengths, idx = span_lengths.sort(descending=True)
    start = start[idx]
    end = end[idx]
    return list(zip(start.tolist(), end.tolist()))[:k]
