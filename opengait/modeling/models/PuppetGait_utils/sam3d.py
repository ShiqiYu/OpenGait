"""SAM 3D Body integration used by PuppetGait.

This module isolates the model-specific setup and the synthetic metadata that
SAM 3D Body expects when its image encoder and MHR decoder are invoked from an
OpenGait training pipeline.  None of the helpers below are learnable.
"""

import math
import sys
from pathlib import Path

import torch


def cast_floating_dtype(obj, dtype):
    """Recursively cast floating tensors while preserving integer metadata."""
    if torch.is_tensor(obj):
        return obj.to(dtype=dtype) if obj.is_floating_point() else obj
    if isinstance(obj, dict):
        return {key: cast_floating_dtype(value, dtype) for key, value in obj.items()}
    if isinstance(obj, list):
        return [cast_floating_dtype(value, dtype) for value in obj]
    if isinstance(obj, tuple):
        return tuple(cast_floating_dtype(value, dtype) for value in obj)
    return obj


def _module_float_dtype(module, fallback=torch.float32):
    for value in list(module.parameters()) + list(module.buffers()):
        if value.is_floating_point():
            return value.dtype
    return fallback


def cast_floating_to_module_dtype(obj, module):
    return cast_floating_dtype(obj, _module_float_dtype(module))


def load_sam3d_body(repo_path, hook_mask, feature_store, msg_mgr):
    """Load frozen SAM 3D Body and register the requested encoder hooks."""
    repo_path = str(Path(repo_path).expanduser().resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        from huggingface_hub import snapshot_download
        from sam_3d_body import load_sam_3d_body
    except ImportError as exc:
        raise ImportError(
            "Cannot import SAM 3D Body. Clone/install it at the path given by "
            f"model_cfg.pretrained_lvm (currently: {repo_path})."
        ) from exc

    msg_mgr.log_info("[PuppetGait] Loading frozen SAM 3D Body...")
    snapshot = Path(snapshot_download(
        repo_id="facebook/sam-3d-body-dinov3",
        revision="11aaa346c7204874a1cbafe3d39a979080b2c55a",
    ))
    # Load the same core model as the demo helper, without its unused detector
    # and FOV estimator. Loading once on CPU also avoids an implicit GPU-0 load.
    engine, _ = load_sam_3d_body(
        checkpoint_path=str(snapshot / "model.ckpt"),
        mhr_path=str(snapshot / "assets" / "mhr_model.pt"),
        device="cpu",
    )

    raw_backbone = getattr(engine, "backbone", None)
    if raw_backbone is None:
        raw_backbone = getattr(engine, "image_encoder", None)
    if raw_backbone is None:
        raise RuntimeError("SAM 3D Body exposes neither backbone nor image_encoder.")
    backbone = getattr(raw_backbone, "encoder", raw_backbone)

    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        blocks = getattr(backbone, "layers", None)
    if blocks is None:
        raise RuntimeError("Cannot find transformer blocks in the SAM3D backbone.")
    if len(hook_mask) > len(blocks):
        raise ValueError(
            f"hook_mask has {len(hook_mask)} entries, but the backbone has "
            f"only {len(blocks)} blocks."
        )

    hook_handles = []

    def capture(index):
        def hook(_module, _inputs, output):
            while isinstance(output, (list, tuple)):
                output = output[0]
            feature_store[index] = output
        return hook

    for layer_index, enabled in enumerate(hook_mask):
        if enabled:
            hook_handles.append(
                blocks[layer_index].register_forward_hook(capture(len(hook_handles)))
            )

    engine.cpu()
    engine.eval()
    engine.requires_grad_(False)
    msg_mgr.log_info(f"[PuppetGait] Hooked {len(hook_handles)} SAM3D layers.")
    return engine, backbone, hook_handles


def _dummy_batch(image_embeddings, target_h, target_w):
    """Construct the camera/image metadata required by the SAM3D decoder."""
    batch_size = image_embeddings.shape[0]
    device = image_embeddings.device
    focal_length = max(target_h, target_w) * 1.1
    center_x, center_y = target_w / 2.0, target_h / 2.0

    camera_intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(
        batch_size, 1, 1
    )
    camera_intrinsics[:, 0, 0] = focal_length
    camera_intrinsics[:, 1, 1] = focal_length
    camera_intrinsics[:, 0, 2] = center_x
    camera_intrinsics[:, 1, 2] = center_y

    grid_y, grid_x = torch.meshgrid(
        torch.arange(target_h, device=device),
        torch.arange(target_w, device=device),
        indexing="ij",
    )
    ray_condition = torch.stack(
        [
            (grid_x - center_x) / focal_length,
            (grid_y - center_y) / focal_length,
        ],
        dim=0,
    ).unsqueeze(0).expand(batch_size, 2, target_h, target_w)

    image_size = torch.tensor(
        [float(target_w), float(target_h)], device=device
    ).view(1, 1, 2).expand(batch_size, 1, 2)
    bbox_center = torch.tensor(
        [center_x, center_y], device=device
    ).view(1, 1, 2).expand(batch_size, 1, 2)
    bbox_scale = torch.tensor(
        [max(target_h, target_w)], device=device
    ).view(1, 1, 1).expand(batch_size, 1, 1)
    affine_transform = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device
    ).view(1, 1, 2, 3).expand(batch_size, 1, 2, 3)

    return {
        "img": torch.zeros(
            batch_size, 1, 3, target_h, target_w, device=device
        ),
        "ori_img_size": image_size,
        "img_size": image_size,
        "bbox_center": bbox_center,
        "bbox_scale": bbox_scale,
        "cam_int": camera_intrinsics,
        "affine_trans": affine_transform,
        "ray_cond": ray_condition,
    }


def decode_body(engine, image_embeddings, target_h, target_w):
    """Run the frozen MHR decoder on already-computed SAM3D image tokens."""
    batch_size = image_embeddings.shape[0]
    device = image_embeddings.device
    batch = _dummy_batch(image_embeddings, target_h, target_w)

    engine._batch_size = batch_size
    engine._max_num_person = 1
    engine.body_batch_idx = torch.arange(batch_size, device=device)
    engine.hand_batch_idx = []

    condition_info = torch.zeros(batch_size, 3, device=device)
    condition_info[:, 2] = 1.1
    dummy_keypoints = torch.zeros(batch_size, 1, 3, device=device)
    dummy_keypoints[..., -1] = -2

    with torch.amp.autocast(enabled=False, device_type=device.type):
        _, pose_outputs = engine.forward_decoder(
            image_embeddings=image_embeddings,
            keypoints=dummy_keypoints,
            condition_info=condition_info,
            batch=batch,
        )
    return cast_floating_dtype(pose_outputs[-1], torch.float32), batch


def visible_vertex_index_map(
    vertices,
    camera_translation,
    camera_intrinsics,
    feature_h,
    feature_w,
    image_h,
    image_w,
):
    """Rasterize the closest visible MHR vertex index at each feature cell."""
    batch_size, num_vertices, _ = vertices.shape
    device = vertices.device

    vertices_camera = vertices + camera_translation.unsqueeze(1)
    x, y, z = vertices_camera.unbind(-1)
    safe_z = z.clamp(min=1e-3)

    fx = camera_intrinsics[:, 0, 0].unsqueeze(1)
    fy = camera_intrinsics[:, 1, 1].unsqueeze(1)
    cx = camera_intrinsics[:, 0, 2].unsqueeze(1)
    cy = camera_intrinsics[:, 1, 2].unsqueeze(1)
    pixel_x = (x / safe_z) * fx + cx
    pixel_y = (y / safe_z) * fy + cy

    feature_x = (pixel_x / image_w * feature_w).long().clamp(0, feature_w - 1)
    feature_y = (pixel_y / image_h * feature_h).long().clamp(0, feature_h - 1)
    flat_pixels = feature_y * feature_w + feature_x

    depth = torch.full(
        (batch_size, feature_h * feature_w), 1e6, device=device
    )
    depth.scatter_reduce_(1, flat_pixels, z, reduce="amin", include_self=False)
    closest_depth = torch.gather(depth, 1, flat_pixels)
    visible = z < closest_depth + 1e-4

    index_map = torch.full(
        (batch_size, feature_h * feature_w),
        -1,
        dtype=torch.long,
        device=device,
    )
    vertex_indices = torch.arange(num_vertices, device=device).unsqueeze(0).expand(
        batch_size, -1
    )
    offsets = torch.arange(batch_size, device=device).unsqueeze(1) * (
        feature_h * feature_w
    )
    global_pixels = (flat_pixels + offsets).reshape(-1)
    visible_flat = visible.reshape(-1)
    index_map.reshape(-1)[global_pixels[visible_flat]] = vertex_indices.reshape(-1)[
        visible_flat
    ]

    return (
        index_map.reshape(batch_size, feature_h, feature_w),
        depth.reshape(batch_size, 1, feature_h, feature_w),
    )


def build_canonical_camera(batch_size, device, image_h, image_w):
    focal_length = max(image_h, image_w) * 1.1
    camera_intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(
        batch_size, 1, 1
    )
    camera_intrinsics[:, 0, 0] = focal_length
    camera_intrinsics[:, 1, 1] = focal_length
    camera_intrinsics[:, 0, 2] = image_w / 2.0
    camera_intrinsics[:, 1, 2] = image_h / 2.0

    camera_translation = torch.zeros(batch_size, 3, device=device)
    camera_translation[:, 2] = 2.2
    return camera_intrinsics, camera_translation


def generate_apose(engine, pose_output):
    """Generate the canonical MHR A-pose while preserving shape and scale."""
    device = pose_output["pred_vertices"].device
    batch_size = pose_output["pred_vertices"].shape[0]

    body_pose = torch.zeros_like(pose_output["body_pose"], dtype=torch.float32)
    arm_angle = math.radians(-20.0)
    body_pose[:, 25] = arm_angle
    body_pose[:, 35] = arm_angle

    with torch.no_grad(), torch.amp.autocast(
        enabled=False, device_type=device.type
    ):
        vertices, keypoints = engine.head_pose.mhr_forward(
            global_trans=torch.zeros(batch_size, 3, device=device),
            global_rot=torch.zeros_like(
                pose_output["global_rot"], dtype=torch.float32
            ),
            body_pose_params=body_pose,
            hand_pose_params=torch.zeros_like(
                pose_output["hand"], dtype=torch.float32
            ),
            scale_params=pose_output["scale"].float(),
            shape_params=pose_output["shape"].float(),
            expr_params=pose_output["face"].float(),
            return_keypoints=True,
        )

    vertices[..., [1, 2]] *= -1
    keypoints = keypoints[:, :70]
    keypoints[..., [1, 2]] *= -1
    return vertices, keypoints
