"""
Matplotlib 기반의 2D 시각화 도구입니다.
1) ``plot_images``로 image를 그립니다.
2) ``plot_keypoints`` 또는 ``plot_matches``를 필요한 만큼 호출합니다.
3) 필요하면 ``save_plot``으로 .png 또는 .pdf를 저장합니다.
"""

import matplotlib
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch


def cm_RdGn(x):
    """red(0) -> yellow(0.5) -> green(1) custom colormap을 만듭니다."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0]]) + (2 - x) * np.array([[1.0, 0, 0]])
    return np.clip(c, 0, 1)


def cm_BlRdGn(x_):
    """blue(-1) -> red(0.0) -> green(1) custom colormap을 만듭니다."""
    x = np.clip(x_, 0, 1)[..., None] * 2
    c = x * np.array([[0, 1.0, 0, 1.0]]) + (2 - x) * np.array([[1.0, 0, 0, 1.0]])

    xn = -np.clip(x_, -1, 0)[..., None] * 2
    cn = xn * np.array([[0, 0.1, 1, 1.0]]) + (2 - xn) * np.array([[1.0, 0, 0, 1.0]])
    out = np.clip(np.where(x_[..., None] < 0, cn, c), 0, 1)
    return out


def cm_prune(x_):
    """pruning 시각화용 custom colormap을 만듭니다."""
    if isinstance(x_, torch.Tensor):
        x_ = x_.cpu().numpy()
    max_i = max(x_)
    norm_x = np.where(x_ == max_i, -1, (x_ - 1) / 9)
    return cm_BlRdGn(norm_x)


def plot_images(imgs, titles=None, cmaps="gray", dpi=100, pad=0.5, adaptive=True):
    """여러 image를 가로로 배치해 그립니다.

    인자:
        imgs: NumPy RGB (H, W, 3), PyTorch RGB (3, H, W) 또는 mono (H, W)의 list
        titles: 각 image의 title string list
        cmaps: monochrome image용 colormap
        adaptive: figure 크기를 image aspect ratio에 맞출지 여부
    """
    # torch.Tensor를 (H, W, 3) 형식으로 변환합니다.
    imgs = [
        img.permute(1, 2, 0).cpu().numpy() if (isinstance(img, torch.Tensor) and img.dim() == 3) else img
        for img in imgs
    ]

    n = len(imgs)
    if not isinstance(cmaps, (list, tuple)):
        cmaps = [cmaps] * n

    if adaptive:
        ratios = [i.shape[1] / i.shape[0] for i in imgs]  # W / H
    else:
        ratios = [4 / 3] * n
    figsize = [sum(ratios) * 4.5, 4.5]
    fig, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi, gridspec_kw={"width_ratios": ratios})
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmaps[i]))
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        ax[i].set_axis_off()
        for spine in ax[i].spines.values():  # frame 테두리를 제거합니다.
            spine.set_visible(False)
        if titles:
            ax[i].set_title(titles[i])
    fig.tight_layout(pad=pad)


def plot_keypoints(kpts, colors="lime", ps=4, axes=None, a=1.0):
    """기존 image 위에 keypoint를 그립니다.

    인자:
        kpts: (N, 2) ndarray list
        colors: string 또는 keypoint별 tuple list의 list
        ps: keypoint 크기
    """
    if not isinstance(colors, list):
        colors = [colors] * len(kpts)
    if not isinstance(a, list):
        a = [a] * len(kpts)
    if axes is None:
        axes = plt.gcf().axes
    for ax, k, c, alpha in zip(axes, kpts, colors, a):
        if isinstance(k, torch.Tensor):
            k = k.cpu().numpy()
        ax.scatter(k[:, 0], k[:, 1], c=c, s=ps, linewidths=0, alpha=alpha)


def plot_matches(kpts0, kpts1, color=None, lw=1.5, ps=4, a=1.0, labels=None, axes=None):
    """두 image 사이의 keypoint match를 그립니다.

    인자:
        kpts0, kpts1: 서로 대응하는 (N, 2) keypoint
        color: 각 match의 string 또는 RGB tuple 색상. 미지정 시 무작위로 정합니다.
        lw: line 두께
        ps: endpoint 크기. 0이면 endpoint를 그리지 않습니다.
        indices: match를 그릴 image index
        a: match line의 alpha opacity
    """
    fig = plt.gcf()
    if axes is None:
        ax = fig.axes
        ax0, ax1 = ax[0], ax[1]
    else:
        ax0, ax1 = axes
    if isinstance(kpts0, torch.Tensor):
        kpts0 = kpts0.cpu().numpy()
    if isinstance(kpts1, torch.Tensor):
        kpts1 = kpts1.cpu().numpy()
    assert len(kpts0) == len(kpts1)
    if color is None:
        color = matplotlib.cm.hsv(np.random.rand(len(kpts0))).tolist()
    elif len(color) > 0 and not isinstance(color[0], (tuple, list)):
        color = [color] * len(kpts0)

    if lw > 0:
        for i in range(len(kpts0)):
            line = matplotlib.patches.ConnectionPatch(
                xyA=(kpts0[i, 0], kpts0[i, 1]),
                xyB=(kpts1[i, 0], kpts1[i, 1]),
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                axesA=ax0,
                axesB=ax1,
                zorder=1,
                color=color[i],
                linewidth=lw,
                clip_on=True,
                alpha=a,
                label=None if labels is None else labels[i],
                picker=5.0,
            )
            line.set_annotation_clip(True)
            fig.add_artist(line)

    # transformation이 바뀌지 않도록 axis를 고정합니다.
    ax0.autoscale(enable=False)
    ax1.autoscale(enable=False)

    if ps > 0:
        ax0.scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
        ax1.scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def add_text(
    idx,
    text,
    pos=(0.01, 0.99),
    fs=15,
    color="w",
    lcolor="k",
    lwidth=2,
    ha="left",
    va="top",
):
    ax = plt.gcf().axes[idx]
    t = ax.text(*pos, text, fontsize=fs, ha=ha, va=va, color=color, transform=ax.transAxes)
    if lcolor is not None:
        t.set_path_effects(
            [
                path_effects.Stroke(linewidth=lwidth, foreground=lcolor),
                path_effects.Normal(),
            ]
        )


def save_plot(path, **kw):
    """현재 figure를 흰 여백 없이 저장합니다."""
    plt.savefig(path, bbox_inches="tight", pad_inches=0, **kw)
