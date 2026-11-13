import copy
import math
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange
from torch.nn import functional as F

from ..base_model import BaseModel
from .PuppetGait_utils.gaitbase import Baseline_ShareTime_2B
from .PuppetGait_utils import (
    build_canonical_camera,
    cast_floating_to_module_dtype,
    decode_body,
    generate_apose,
    load_sam3d_body,
    visible_vertex_index_map,
)


class ResizeToHW(nn.Module):
    """Bilinearly resize a 2D feature map to a fixed spatial shape."""

    def __init__(self, target_size):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        return F.interpolate(
            x, size=self.target_size, mode="bilinear", align_corners=False
        )


class SparseGeometryTransport(nn.Module):
    """Sparse Sinkhorn transport on a geometry-restricted bipartite graph.

    Top-k pruning is target-wise: each canonical target cell retains at most
    ``topk_support`` geometrically closest source tokens before normalization.
    The module is parameter-free and the transport plan is not differentiated.
    """

    def __init__(
        self,
        temperature=0.01,
        dist_thresh=0.2,
        num_iters=8,
        topk_support=4,
        sparse_rebalance_iters=4,
    ):
        super().__init__()
        self.epsilon = temperature
        self.dist_thresh = dist_thresh
        self.num_iters = num_iters
        self.topk_support = int(topk_support) if topk_support else 0
        if sparse_rebalance_iters is None or int(sparse_rebalance_iters) <= 0:
            self.sparse_rebalance_iters = num_iters
        else:
            self.sparse_rebalance_iters = int(sparse_rebalance_iters)

    def forward(
        self,
        source_feats,
        source_locs,
        target_locs,
        source_valid_mask=None,
        target_valid_mask=None,
    ):
        batch_size = source_feats.shape[0]

        with torch.no_grad():
            diff = target_locs.unsqueeze(2) - source_locs.unsqueeze(1)
            dist_sq = torch.sum(diff ** 2, dim=-1)
            log_kernel = -dist_sq / (self.epsilon + 1e-8)
            valid = dist_sq < self.dist_thresh ** 2
            del diff, dist_sq

            if source_valid_mask is not None:
                valid &= source_valid_mask.unsqueeze(1)
            if target_valid_mask is not None:
                valid &= target_valid_mask.unsqueeze(2)

            if self.topk_support > 0:
                source_count = source_feats.shape[1]
                topk = min(self.topk_support, source_count)
                if topk < source_count:
                    masked_kernel = log_kernel.masked_fill(~valid, -1e9)
                    topk_indices = masked_kernel.topk(
                        k=topk, dim=2, largest=True
                    ).indices
                    sparse_support = torch.zeros_like(valid)
                    sparse_support.scatter_(2, topk_indices, True)
                    valid &= sparse_support

            log_kernel = log_kernel.masked_fill(~valid, -1e9)
            source_count = source_feats.shape[1]
            target_count = target_locs.shape[1]
            source_dual = torch.zeros(
                batch_size, 1, source_count, device=source_feats.device
            )
            target_dual = torch.zeros(
                batch_size, target_count, 1, device=source_feats.device
            )
            sinkhorn_iters = (
                self.sparse_rebalance_iters
                if self.topk_support > 0
                else self.num_iters
            )

            for _ in range(sinkhorn_iters):
                target_dual = -torch.logsumexp(
                    log_kernel + source_dual, dim=2, keepdim=True
                )
                source_dual = -torch.logsumexp(
                    log_kernel + target_dual, dim=1, keepdim=True
                )
                if source_valid_mask is not None:
                    source_dual = source_dual.masked_fill(
                        ~source_valid_mask.unsqueeze(1), 0.0
                    )

            attention = torch.exp(log_kernel + target_dual + source_dual)
            has_source = valid.any(dim=-1, keepdim=True)

        target_feats = torch.bmm(attention, source_feats)
        if target_valid_mask is not None:
            target_feats = target_feats * target_valid_mask.unsqueeze(-1).float()
        target_feats = target_feats * has_source.float()
        return target_feats


class PuppetGait(BaseModel):
    """Public PuppetGait model for OpenGait."""

    def build_network(self, model_cfg):
        self.pretrained_lvm = model_cfg["pretrained_lvm"]
        self.image_size = model_cfg["image_size"]
        self.sils_size = model_cfg["sils_size"]
        self.feature_dim = model_cfg["source_dim"]
        self.num_unknown = model_cfg["num_unknown"]
        self.num_FPN = model_cfg["num_FPN"]
        self.chunk_size = model_cfg.get("chunk_size", 96)
        self.crop_square_rgb = model_cfg.get("crop_square_rgb", True)

        layer_cfg = model_cfg.get("layer_config", {})
        self.hook_mask = layer_cfg.get(
            "hook_mask", [False] * 16 + [True] * 16
        )
        self.hook_sample_type = layer_cfg.get("hook_sample_type", "chunk")
        hooked_layers = sum(self.hook_mask)
        if len(self.hook_mask) != 32 or not self.hook_mask[-1]:
            raise ValueError("Select SAM3D's final layer with a 32-entry hook_mask.")
        if hooked_layers == 0:
            raise ValueError("layer_config.hook_mask selects no SAM3D layers.")
        if hooked_layers % self.num_FPN != 0:
            raise ValueError(
                f"Hooked layers ({hooked_layers}) must be divisible by "
                f"num_FPN ({self.num_FPN})."
            )
        self.layers_per_head = hooked_layers // self.num_FPN

        self.branch_configs = model_cfg["branch_configs"]
        self.num_branches = len(self.branch_configs)
        if self.num_branches == 0:
            raise ValueError("branch_configs must contain at least one branch.")
        for branch_cfg in self.branch_configs:
            if "yaw" not in branch_cfg:
                raise ValueError("Every branch configuration must define yaw.")

        self.Gait_Nets = nn.ModuleList(
            [
                Baseline_ShareTime_2B(copy.deepcopy(model_cfg))
                for _ in range(self.num_branches)
            ]
        )

        input_dim = self.feature_dim * self.layers_per_head
        self.HumanSpace_Conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm2d(input_dim, affine=False),
                    nn.Conv2d(input_dim, self.feature_dim // 2, kernel_size=1),
                    nn.BatchNorm2d(self.feature_dim // 2, affine=False),
                    nn.GELU(),
                    nn.Conv2d(
                        self.feature_dim // 2, self.num_unknown, kernel_size=1
                    ),
                    ResizeToHW((self.sils_size * 2, self.sils_size)),
                    nn.BatchNorm2d(self.num_unknown, affine=False),
                    nn.Sigmoid(),
                )
                for _ in range(self.num_FPN)
            ]
        )

        self.ot_solver = SparseGeometryTransport(
            temperature=model_cfg.get("ot_temperature", 0.01),
            dist_thresh=model_cfg.get("ot_dist_thresh", 0.2),
            num_iters=model_cfg.get("ot_iters", 8),
            topk_support=model_cfg.get("ot_topk_support", 4),
            sparse_rebalance_iters=model_cfg.get(
                "ot_sparse_rebalance_iters", 4
            ),
        )

        # SAM3D is loaded after the trainable modules are initialized.  This
        # avoids constructing the 1.3B-parameter frozen model twice.
        self.intermediate_features = {}
        self.SAM_Engine = None
        self.Backbone = None
        self.hook_handles = []

    def init_parameters(self):
        # Preserve OpenGait's initialization for every trainable module.
        super().init_parameters()
        (
            self.SAM_Engine,
            self.Backbone,
            self.hook_handles,
        ) = load_sam3d_body(
            self.pretrained_lvm,
            self.hook_mask,
            self.intermediate_features,
            self.msg_mgr,
        )

        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        self.msg_mgr.log_info(
            "All Model Count: {:.5f}M".format(parameter_count / 1e6)
        )

    def train(self, mode=True):
        """Keep the frozen SAM3D model in evaluation mode."""
        super().train(mode)
        if self.SAM_Engine is not None:
            self.SAM_Engine.eval()
        return self

    def get_loader(self, data_cfg, train=True):
        from .PuppetGait_utils.data import build_rgb_loader

        return build_rgb_loader(self.cfgs, data_cfg, train)

    @staticmethod
    def _is_original_branch(branch_cfg):
        if branch_cfg.get("original_view", False) or branch_cfg.get(
            "no_ot", False
        ):
            return True
        yaw = branch_cfg.get("yaw")
        if yaw is None:
            return True
        if isinstance(yaw, str):
            return yaw.strip().lower() in {
                "none",
                "null",
                "raw",
                "original",
                "original_view",
                "identity",
                "no_rotate",
                "no-rotate",
            }
        return False

    @staticmethod
    def _rotate_geometry(
        vertices, keypoints, global_rotation, yaw, align_global_rotation
    ):
        batch_size = vertices.shape[0]
        device = vertices.device
        midhip = (keypoints[:, 9] + keypoints[:, 10]) / 2.0
        centered_vertices = vertices - midhip.unsqueeze(1)

        cosine = math.cos(math.radians(yaw))
        sine = math.sin(math.radians(yaw))
        yaw_rotation = torch.tensor(
            [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
            device=device,
            dtype=torch.float32,
        ).view(1, 3, 3).expand(batch_size, -1, -1)

        if align_global_rotation:
            import roma

            corrected_rotation = global_rotation.clone()
            corrected_rotation[..., [0, 1, 2]] *= -1
            canonical_rotation = roma.euler_to_rotmat(
                "XYZ", corrected_rotation
            )
            rotation = torch.matmul(
                canonical_rotation.transpose(1, 2),
                yaw_rotation.transpose(1, 2),
            )
        else:
            rotation = yaw_rotation.transpose(1, 2)

        vertices_smpl = centered_vertices.clone()
        vertices_smpl[..., [1, 2]] *= -1
        rotated_vertices = torch.bmm(vertices_smpl, rotation)
        rotated_vertices[..., [1, 2]] *= -1
        return rotated_vertices, midhip, rotation

    def _build_canonical_geometry(self, branch_cfg, pose_output):
        if self._is_original_branch(branch_cfg):
            return None

        if branch_cfg.get("use_apose", False):
            if self._apose_cache is None:
                self._apose_cache = generate_apose(
                    self.SAM_Engine, pose_output
                )
            vertices, keypoints = self._apose_cache
            align_global_rotation = False
        else:
            vertices = pose_output["pred_vertices"]
            keypoints = pose_output["pred_keypoints_3d"]
            align_global_rotation = True

        return {
            "vertices": vertices,
            "keypoints": keypoints,
            "yaw": float(branch_cfg.get("yaw", 0.0)),
            "align_global_rotation": align_global_rotation,
        }

    def _canonicalize(
        self,
        human_features,
        source_mask,
        pose_output,
        canonical_geometry,
        source_camera,
        target_camera,
        feature_h,
        feature_w,
        image_h,
        image_w,
    ):
        if canonical_geometry is None:
            return human_features

        batch_size = human_features.shape[0]
        source_intrinsics, source_translation = source_camera
        target_intrinsics, target_translation = target_camera
        source_vertices = pose_output["pred_vertices"]

        source_index_map, _ = visible_vertex_index_map(
            source_vertices,
            source_translation,
            source_intrinsics,
            feature_h,
            feature_w,
            image_h,
            image_w,
        )
        valid_source = (source_mask.squeeze(1) > 0.5) & (
            source_index_map >= 0
        )
        flat_features = rearrange(
            human_features, "b c h w -> b (h w) c"
        )

        flat_indices = source_index_map.reshape(batch_size, -1)
        safe_indices = flat_indices.clamp(min=0)
        indexed_vertices = torch.gather(
            canonical_geometry["vertices"],
            1,
            safe_indices.unsqueeze(-1).expand(-1, -1, 3),
        )

        rotated_vertices, midhip, rotation = self._rotate_geometry(
            canonical_geometry["vertices"],
            canonical_geometry["keypoints"],
            pose_output["global_rot"],
            canonical_geometry["yaw"],
            canonical_geometry["align_global_rotation"],
        )
        _, target_depth = visible_vertex_index_map(
            rotated_vertices,
            target_translation,
            target_intrinsics,
            feature_h,
            feature_w,
            image_h,
            image_w,
        )
        valid_target = target_depth.reshape(batch_size, -1) < 1e5

        centered_sources = indexed_vertices - midhip.unsqueeze(1)
        sources_smpl = centered_sources.clone()
        sources_smpl[..., [1, 2]] *= -1
        projected_vertices = torch.bmm(sources_smpl, rotation)
        projected_vertices[..., [1, 2]] *= -1
        # Preserve float32 promotion under AMP, as in the research projection.
        projected_vertices = projected_vertices + target_translation.unsqueeze(1)

        x, y, z = projected_vertices.unbind(-1)
        z = z.clamp(min=1e-3)
        fx = target_intrinsics[:, 0, 0].unsqueeze(1)
        fy = target_intrinsics[:, 1, 1].unsqueeze(1)
        cx = target_intrinsics[:, 0, 2].unsqueeze(1)
        cy = target_intrinsics[:, 1, 2].unsqueeze(1)
        projected_x = (x / z) * fx + cx
        projected_y = (y / z) * fy + cy
        source_locations = torch.stack(
            [
                2.0 * projected_x / image_w - 1.0,
                2.0 * projected_y / image_h - 1.0,
            ],
            dim=-1,
        )

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, feature_h, device=human_features.device),
            torch.linspace(-1, 1, feature_w, device=human_features.device),
            indexing="ij",
        )
        target_locations = torch.stack([grid_x, grid_y], dim=-1)
        target_locations = target_locations.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        ).reshape(batch_size, -1, 2)

        transported = self.ot_solver(
            flat_features,
            source_locations,
            target_locations,
            source_valid_mask=valid_source.reshape(batch_size, -1),
            target_valid_mask=valid_target,
        )
        return rearrange(
            transported, "b (h w) c -> b c h w", h=feature_h
        )

    def _select_layer_groups(self, hooked_features):
        if self.hook_sample_type == "interleave":
            return [
                hooked_features[index :: self.num_FPN]
                for index in range(self.num_FPN)
            ]
        if self.hook_sample_type == "chunk":
            return [
                hooked_features[
                    index * self.layers_per_head : (index + 1)
                    * self.layers_per_head
                ]
                for index in range(self.num_FPN)
            ]
        raise ValueError(
            f"Unsupported layer_config.hook_sample_type: "
            f"{self.hook_sample_type}"
        )

    def forward(self, inputs):
        ipts, labels, _, _, seq_l = inputs
        rgb = ipts[0]
        del ipts
        if rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError("PuppetGait expects RGB inputs with shape [N, S, 3, H, W].")
        # Match the research RGB transform without changing OpenGait globally.
        crop_square = self.training or self.crop_square_rgb
        if crop_square and rgb.shape[-2] == rgb.shape[-1]:
            cutting = rgb.shape[-1] // 4
            if cutting:
                rgb = rgb[..., cutting:-cutting]

        # Keep the chunking rule identical to the submitted implementation.
        rgb_chunks = torch.chunk(
            rgb, (rgb.size(1) // self.chunk_size) + 1, dim=1
        )
        all_outputs = [[] for _ in range(self.num_branches)]
        image_h, image_w = self.image_size * 2, self.image_size
        feature_h, feature_w = image_h // 16, image_w // 16
        target_tokens = feature_h * feature_w

        for rgb_chunk in rgb_chunks:
            batch_size, sequence_length, channels, height, width = (
                rgb_chunk.shape
            )
            rgb_frames = rearrange(
                rgb_chunk, "n s c h w -> (n s) c h w"
            ).contiguous()
            frame_count = rgb_frames.shape[0]

            with torch.no_grad():
                backbone_input = F.interpolate(
                    rgb_frames,
                    size=(image_h, image_w),
                    mode="bilinear",
                    align_corners=False,
                )
                backbone_input = cast_floating_to_module_dtype(
                    backbone_input, self.Backbone
                )
                self.intermediate_features.clear()
                _ = self.Backbone(backbone_input)

                image_embeddings = self.intermediate_features[
                    len(self.hook_handles) - 1
                ]
                if image_embeddings.shape[1] > target_tokens:
                    image_embeddings = image_embeddings[:, -target_tokens:, :]
                image_embeddings = image_embeddings.transpose(1, 2).reshape(
                    frame_count, -1, feature_h, feature_w
                ).float()

                pose_output, dummy_batch = decode_body(
                    self.SAM_Engine,
                    image_embeddings,
                    image_h,
                    image_w,
                )
                self._apose_cache = None
                source_vertices = pose_output["pred_vertices"]
                source_translation = pose_output["pred_cam_t"]
                source_intrinsics = dummy_batch["cam_int"].float()

                _, source_depth = visible_vertex_index_map(
                    source_vertices,
                    source_translation,
                    source_intrinsics,
                    feature_h,
                    feature_w,
                    image_h,
                    image_w,
                )
                projection_mask = (source_depth < 1e5).float()

                hooked_features = []
                for index in range(len(self.hook_handles)):
                    feature = self.intermediate_features[index]
                    if feature.shape[1] > target_tokens:
                        feature = feature[:, -target_tokens:, :]
                    hooked_features.append(feature.float())

            reduced_features = []
            for index, layer_group in enumerate(
                self._select_layer_groups(hooked_features)
            ):
                layer_features = torch.concat(layer_group, dim=-1)
                layer_features = partial(nn.LayerNorm, eps=1e-6)(
                    self.feature_dim * len(layer_group),
                    elementwise_affine=False,
                )(layer_features)
                layer_features = rearrange(
                    layer_features,
                    "b (h w) c -> b c h w",
                    h=feature_h,
                ).contiguous()
                reduced_features.append(
                    self.HumanSpace_Conv[index](layer_features)
                )

            human_features = torch.concat(reduced_features, dim=1)
            source_mask = F.interpolate(
                projection_mask,
                size=(self.sils_size * 2, self.sils_size),
                mode="bilinear",
                align_corners=False,
            ).detach()
            human_features *= (source_mask > 0.5).to(human_features)

            target_camera = build_canonical_camera(
                frame_count,
                rgb.device,
                image_h,
                image_w,
            )
            source_camera = (source_intrinsics, source_translation)

            for branch_index, branch_cfg in enumerate(self.branch_configs):
                canonical_geometry = self._build_canonical_geometry(
                    branch_cfg, pose_output
                )
                branch_features = self._canonicalize(
                    human_features,
                    source_mask,
                    pose_output,
                    canonical_geometry,
                    source_camera,
                    target_camera,
                    self.sils_size * 2,
                    self.sils_size,
                    image_h,
                    image_w,
                )
                branch_features = rearrange(
                    branch_features,
                    "(n s) c h w -> n c s h w",
                    n=batch_size,
                    s=sequence_length,
                ).contiguous()
                if self.training:
                    branch_output = torch.utils.checkpoint.checkpoint(
                        self.Gait_Nets[branch_index].test_1,
                        branch_features,
                        use_reentrant=False,
                    )
                else:
                    branch_output = self.Gait_Nets[branch_index].test_1(
                        branch_features
                    )
                all_outputs[branch_index].append(branch_output)

        embedding_groups = [[] for _ in range(self.num_FPN)]
        logit_groups = [[] for _ in range(self.num_FPN)]
        for branch_index in range(self.num_branches):
            sequence_features = torch.cat(
                all_outputs[branch_index], dim=2
            )
            embeddings, logits = self.Gait_Nets[branch_index].test_2(
                sequence_features, seq_l
            )
            for fpn_index in range(self.num_FPN):
                embedding_groups[fpn_index].append(embeddings[fpn_index])
                logit_groups[fpn_index].append(logits[fpn_index])

        embedding_list = [
            torch.cat(group, dim=-1) for group in embedding_groups
        ]
        logit_list = [torch.cat(group, dim=-1) for group in logit_groups]
        combined_embedding = torch.cat(embedding_list, dim=-1)

        if self.training:
            training_features = {
                "triplet": {
                    "embeddings": combined_embedding,
                    "labels": labels,
                },
                "softmax": {
                    "logits": torch.cat(logit_list, dim=-1),
                    "labels": labels,
                },
            }
        else:
            training_features = {}

        return {
            "training_feat": training_features,
            "visual_summary": {},
            "inference_feat": {
                "embeddings": combined_embedding,
                **{
                    f"embeddings_{index}": embedding
                    for index, embedding in enumerate(embedding_list)
                },
            },
        }
