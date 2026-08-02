import argparse
from pathlib import Path

import hydra
import torch
from einops import einsum
from hydra import compose, initialize_config_module
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm

from hmr4d.configs import register_store_footmr
from hmr4d.model.footmr.footmr_pl_demo import DemoPL
from hmr4d.model.footmr.utils.contact_grounding import apply_contact_linear_grounding
from hmr4d.utils.geo.hmr_cam import (
    convert_K_to_K4,
    create_camera_sensor,
    estimate_K,
    get_bbx_xys_from_xyxy,
)
from hmr4d.utils.geo_transform import apply_T_on_points, compute_cam_angvel, compute_T_ayfz2ay
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.preproc import Extractor, SimpleVO, Tracker, VitPoseExtractor
from hmr4d.utils.pylogger import Log
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    get_video_reader,
    get_writer,
    merge_videos_horizontal,
    normalize_video_to_30fps,
    read_video_np,
    resolve_focal_length_35mm,
    save_video,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco_skeleton_batch
from hmr4d.utils.vis.renderer import (
    Renderer,
    get_global_cameras_static,
    get_global_render_y_offset,
    get_ground_params_from_points,
)

CRF = 23  # 17은 무손실이며, 6이 증가할 때마다 mp4 크기가 절반으로 줄어듭니다.
DEFAULT_FOCAL_MM = (24**2 + 36**2) ** 0.5


def parse_args_to_cfg():
    # 모든 인자를 cfg에 반영합니다.
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str)
    parser.add_argument("--output_root", type=str, default=None, help="by default to outputs/demo")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument(
        "--use_dpvo", action="store_true", help="If true, use DPVO. By default not using DPVO."
    )
    parser.add_argument(
        "--use_sapiens",
        action="store_true",
        help="If true, use Sapiens 2D poses. By default use ViTPose because it's faster.",
    )
    parser.add_argument(
        "--no_postproc",
        action="store_true",
        help="If true, does not perform post processing of global human motion."
        "Post processing can lead to fine-grained motion being suppressed.",
    )
    parser.add_argument(
        "--f_mm",
        type=int,
        default=None,
        help="35mm-equivalent focal length in mm. When omitted, read verified original-video "
        "metadata and otherwise use the GVHMR default estimate.",
    )
    parser.add_argument(
        "--grounding",
        choices=("none", "contact-linear"),
        default="none",
        help="평지 보행용 선택형 grounding입니다. contact-linear는 Raw와 C_Temporal을 사용합니다.",
    )
    parser.add_argument(
        "--export-trc",
        action="store_true",
        help="저장된 global 결과를 추가 변환 없이 TRC로 내보냅니다.",
    )
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    args = parser.parse_args()

    # 입력
    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")

    selected_f_mm, focal_source = resolve_focal_length_35mm(video_path, args.f_mm)
    if focal_source == "cli":
        Log.info(f"[Focal] {selected_f_mm:g} mm equivalent from --f_mm")
    elif focal_source == "metadata":
        Log.info(f"[Focal] {selected_f_mm:g} mm equivalent from original video metadata")
    else:
        Log.info("[Focal] No verified metadata; using the GVHMR default estimate")

    focal_suffix = "default"
    if selected_f_mm is not None:
        focal_suffix = f"{selected_f_mm:g}".replace(".", "p")
    grounding_suffix = "" if args.grounding == "none" else "_gcontact"

    # 설정
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}_fps30_f{focal_suffix}{grounding_suffix}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"use_dpvo={args.use_dpvo}",
            f"no_postproc={args.no_postproc}",
            f"use_sapiens={args.use_sapiens}",
            f"grounding={args.grounding}",
            f"export_trc={args.export_trc}",
        ]
        if selected_f_mm is not None:
            overrides.append(f"f_mm={selected_f_mm}")

        # 출력 루트 변경을 허용합니다.
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")
        register_store_footmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # 출력
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # FootMR checkpoint의 시간축에 맞춰 canonical 30fps 입력을 준비합니다.
    Log.info(f"[Normalize Video] {video_path} -> {cfg.video_path}")
    normalize_video_to_30fps(video_path, cfg.video_path, crf=CRF)

    return cfg


@torch.no_grad()
def run_preprocess(cfg):
    Log.info("[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # bounding box tracking 결과를 구합니다.
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(
            bbx_xyxy, base_enlarge=1.2
        ).float()  # (L, 3) aspect ratio를 적용하고 확대합니다.
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

    # VitPose 결과를 구합니다.
    number_joints = 23
    if not Path(paths.vitpose).exists():
        if cfg.use_sapiens:
            from hmr4d.utils.preproc.sapiens import SapiensPoseExtractor

            sapiens_extractor = SapiensPoseExtractor()
            vitpose = sapiens_extractor.extract(video_path, paths.extracted_frames, bbx_xyxy)
            torch.save(vitpose, paths.vitpose)
            del sapiens_extractor
        else:
            vitpose_extractor = VitPoseExtractor(number_joints=number_joints)
            vitpose = vitpose_extractor.extract(video_path, bbx_xys)
            torch.save(vitpose, paths.vitpose)
            del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco_skeleton_batch(video, vitpose, number_joints, conf_thr=0.5)
        save_video(video_overlay, paths.vitpose_video_overlay)

    # ViT feature를 구합니다.
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # visual odometry 결과를 구합니다.
    if not static_cam:  # SLAM을 사용해 camera rotation을 구합니다.
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                vo_f_mm = cfg.f_mm if cfg.f_mm is not None else DEFAULT_FOCAL_MM
                simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=vo_f_mm)
                vo_results = simple_vo.compute()  # (L, 4, 4), numpy
                torch.save(vo_results, paths.slam)
            else:  # DPVO 사용
                from hmr4d.utils.preproc.slam import SLAMModel

                length, width, height = get_video_lwh(cfg.video_path)
                if cfg.f_mm is not None:
                    K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2]
                else:
                    K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
                bar = tqdm(total=length, desc="DPVO")
                while True:
                    ret = slam.track()
                    if ret:
                        bar.update()
                    else:
                        break
                slam_results = slam.process()  # (L, 7), numpy
                torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time() - tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        if cfg.use_dpvo:  # DPVO 사용
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        else:  # SimpleVO 사용
            R_w2c = torch.from_numpy(traj[:, :3, :3])
    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    kp2d = torch.load(paths.vitpose)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": kp2d,
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }
    return data


def render_incam(cfg):
    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    pred = torch.load(cfg.paths.hmr4d_results)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces

    # SMPL 변환
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    # -- rendering 코드 -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    # renderer 준비
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy

    # -- mesh rendering 실행 -- #
    verts_incam = pred_c_verts
    writer = get_writer(incam_video_path, fps=30, crf=CRF)
    for i, img_raw in tqdm(
        enumerate(reader), total=get_video_lwh(video_path)[0], desc="Rendering Incam"
    ):
        img = renderer.render_mesh(verts_incam[i].cuda(), img_raw, [0.8, 0.8, 0.8])
        writer.write_frame(img)
    writer.close()
    reader.close()


def render_global(cfg):
    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    pred = torch.load(cfg.paths.hmr4d_results)
    grounding_report = pred.get("grounding")
    if isinstance(grounding_report, dict) and grounding_report.get("applied") is True:
        Log.info("[Render Global] Preserving contact-linear ground at Y=0")
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

    # SMPL 변환
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    def move_to_start_point_face_z(verts):
        """XZ 원점으로 이동하고 지면에서 시작하며 Z축을 향하도록 정렬합니다."""
        # 위치 정렬
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = get_global_render_y_offset(verts, grounding_report)
        verts = verts - offset
        # 정면 방향 정렬
        T_ay2ayfz = compute_T_ayfz2ay(
            einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True
        )
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    # -- rendering 코드 -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)  # 24mm lens로 rendering합니다.

    # renderer 준비
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    # renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- mesh rendering 실행 -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    render_length = length if not debug_cam else 8
    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in tqdm(range(render_length), desc="Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()


def export_trc_if_requested(cfg):
    """요청된 경우에만 저장된 production global 결과를 TRC로 내보냅니다."""

    if not cfg.export_trc:
        return

    from hmr4d.utils.trc_export import export_global_trc

    Log.info("[TRC Export] Start")
    report = export_global_trc(
        cfg.paths.hmr4d_results,
        cfg.video_path,
        out_dir=cfg.paths.trc_dir,
        device="cpu",
    )
    Log.info(
        "[TRC Export] Complete: "
        f"frames={report['frames']}, markers={report['marker_count']}, "
        f"fps={report['timebase']['fps']:g}, trc={report['artifacts']['trc']}"
    )


if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    paths = cfg.paths
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
    Log.info(f"[GPU]: {torch.cuda.get_device_properties('cuda')}")

    # ===== 전처리 후 디스크에 저장 ===== #
    run_preprocess(cfg)
    data = load_data_dict(cfg)

    # ===== HMR4D 추론 ===== #
    if not Path(paths.hmr4d_results).exists():
        Log.info("[HMR4D] Predicting")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        tic = Log.sync_time()
        use_contact_grounding = cfg.grounding == "contact-linear"
        if use_contact_grounding:
            Log.info("[Grounding] Using Raw motion with C_Temporal contact-linear grounding")
        pred = model.predict(
            data,
            static_cam=cfg.static_cam,
            no_postproc=cfg.no_postproc or use_contact_grounding,
        )
        if use_contact_grounding:
            boxes_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
            grounded_global, grounding_report = apply_contact_linear_grounding(
                pred["smpl_params_global"],
                pred["smpl_params_incam"],
                data["kp2d"],
                boxes_xyxy,
                model.pipeline.endecoder.smplx_model,
                fps=30.0,
            )
            pred["smpl_params_global"] = grounded_global
            pred["grounding"] = grounding_report
            if grounding_report["applied"]:
                Log.info(
                    "[Grounding] Applied: "
                    f"slope={grounding_report['slope_mps']:.6f} m/s, "
                    f"stances={grounding_report['contact_event_count']}"
                )
            else:
                Log.warning(
                    "[Grounding] Quality check failed; returning Raw motion: "
                    f"{grounding_report['fallback_reason']}"
                )
        pred = detach_to_cpu(pred)
        data_time = data["length"] / 30
        Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
        torch.save(pred, paths.hmr4d_results)

    # ===== 선택형 TRC export ===== #
    export_trc_if_requested(cfg)

    # ===== 결과 rendering ===== #
    render_incam(cfg)
    render_global(cfg)
    if not Path(paths.incam_global_horiz_video).exists():
        Log.info("[Merge Videos]")
        merge_videos_horizontal(
            [paths.incam_video, paths.global_video], paths.incam_global_horiz_video
        )
