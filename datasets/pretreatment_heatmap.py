import os
import cv2
import yaml
import math
import torch
import random
import pickle
import argparse
import numpy as np
from glob import glob 
from tqdm import tqdm
import matplotlib.cm as cm
import torch.distributed as dist
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import KNNImputer, SimpleImputer

torch.manual_seed(347)
random.seed(347)

#########################################################################################################
# The following code is the base class code for generating heatmap.
#########################################################################################################

class GeneratePoseTarget:
    """Generate pseudo heatmaps based on joint coordinates and confidence.
    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".
    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
        with_kp (bool): Generate pseudo heatmaps for keypoints. Default: True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Default: False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Default: ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
                      (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                      (6, 12), (12, 14), (14, 16), (11, 12)),
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
        left_limb (tuple[int]): Indexes of left limbs, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left limbs of skeletons we defined for COCO-17p.
        right_limb (tuple[int]): Indexes of right limbs, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right limbs of skeletons we defined for COCO-17p.
    """

    def __init__(self,
                 sigma=0.6,
                 use_score=True,
                 with_kp=True,
                 with_limb=False,
                 skeletons=((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                            (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                            (13, 15), (6, 12), (12, 14), (14, 16), (11, 12)),
                 double=False,
                 left_kp=(1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp=(2, 4, 6, 8, 10, 12, 14, 16),
                 left_limb=(0, 2, 4, 5, 6, 10, 11, 12),
                 right_limb=(1, 3, 7, 8, 9, 13, 14, 15),
                 scaling=1.,
                 eps= 1e-3,
                 img_h=64,
                 img_w = 64):

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double
        self.eps = eps

        assert self.with_kp + self.with_limb == 1, ('One of "with_limb" and "with_kp" should be set as True.')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons
        self.left_limb = left_limb
        self.right_limb = right_limb
        self.scaling = scaling
        self.img_h = img_h
        self.img_w = img_w

    def generate_a_heatmap(self, arr, centers, max_values, point_center):
        """Generate pseudo heatmap for one keypoint in one frame.
        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: 1 * 2.
            max_values (np.ndarray): The max values of each keypoint. Shape: (1, ).
            point_center: Shape: (1, 2)
        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for center, max_value in zip(centers, max_values):
            if max_value < self.eps:
                continue

            mu_x, mu_y = center[0], center[1]

            tmp_st_x = int(mu_x - 3 * sigma)
            tmp_ed_x = int(mu_x + 3 * sigma)
            tmp_st_y = int(mu_y - 3 * sigma)
            tmp_ed_y = int(mu_y + 3 * sigma)

            st_x = max(tmp_st_x, 0) 
            ed_x = min(tmp_ed_x + 1, img_w)
            st_y = max(tmp_st_y, 0)
            ed_y = min(tmp_ed_y + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value

            arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

    def generate_a_limb_heatmap(self, arr, starts, ends, start_values, end_values, point_center):
        """Generate pseudo heatmap for one limb in one frame.
        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            starts (np.ndarray): The coordinates of one keypoint in the corresponding limbs. Shape: 1 * 2.
            ends (np.ndarray): The coordinates of the other keypoint in the corresponding limbs. Shape: 1 * 2.
            start_values (np.ndarray): The max values of one keypoint in the corresponding limbs. Shape: (1, ).
            end_values (np.ndarray): The max values of the other keypoint in the corresponding limbs. Shape: (1, ).
        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for start, end, start_value, end_value in zip(starts, ends, start_values, end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])


            
            tmp_min_x = int(min_x - 3 * sigma)
            tmp_max_x = int(max_x + 3 * sigma) 
            tmp_min_y = int(min_y - 3 * sigma)
            tmp_max_y = int(max_y + 3 * sigma)

            min_x = max(tmp_min_x, 0)
            max_x = min(tmp_max_x + 1, img_w)
            min_y = max(tmp_min_y, 0)
            max_y = min(tmp_max_y + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = ((x - start[0])**2 + (y - start[1])**2)

            # distance to end keypoints
            d2_end = ((x - end[0])**2 + (y - end[1])**2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)

            if d2_ab < 1:
                self.generate_a_heatmap(arr, start[None], start_value[None], point_center)
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
            d2_seg = a_dominate * d2_start + b_dominate * d2_end + seg_dominate * d2_line

            patch = np.exp(-d2_seg / 2. / sigma**2)
            patch = patch * value_coeff

            arr[min_y:max_y, min_x:max_x] = np.maximum(arr[min_y:max_y, min_x:max_x], patch)
    def generate_heatmap(self, arr, kps, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).
        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: V * img_h * img_w.
            kps (np.ndarray): The coordinates of keypoints in this frame. Shape: 1 * V * 2.
            max_values (np.ndarray): The confidence score of each keypoint. Shape: 1 * V.
        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        point_center = kps.mean(1)

        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                self.generate_a_heatmap(arr[i], kps[:, i], max_values[:, i], point_center)

        if self.with_limb:
            for i, limb in enumerate(self.skeletons):
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                self.generate_a_limb_heatmap(arr[i], starts, ends, start_values, end_values, point_center)

    def gen_an_aug(self, pose_data):
        """Generate pseudo heatmaps for all frames.
        Args:
            pose_data (array): [1, T, V, C]
        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = pose_data[..., :2]
        kp_shape = pose_data.shape # [1, T, V, 2]

        if pose_data.shape[-1] == 3:
            all_kpscores = pose_data[..., -1] # [1, T, V]
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        

        # scale img_h, img_w and kps
        img_h = int(self.img_h * self.scaling + 0.5)
        img_w = int(self.img_w * self.scaling + 0.5)
        all_kps[..., :2] *= self.scaling

        num_frame = kp_shape[1]
        num_c = 0
        if self.with_kp:
            num_c += all_kps.shape[2]
        if self.with_limb:
            num_c += len(self.skeletons)
        ret = np.zeros([num_frame, num_c, img_h, img_w], dtype=np.float32)

        for i in range(num_frame):
            # 1, V, C
            kps = all_kps[:, i]
            # 1, V
            kpscores = all_kpscores[:, i] if self.use_score else np.ones_like(all_kpscores[:, i])

            self.generate_heatmap(ret[i], kps, kpscores)
        return ret

    def __call__(self, pose_data):
        """
        pose_data: (T, V, C=3/2)
        1: means person number
        """
        pose_data = pose_data[None,...] # (1, T, V, C=3/2)

        heatmap = self.gen_an_aug(pose_data)
        
        if self.double:
            indices = np.arange(heatmap.shape[1], dtype=np.int64)
            left, right = (self.left_kp, self.right_kp) if self.with_kp else (self.left_limb, self.right_limb)
            for l, r in zip(left, right):  # noqa: E741
                indices[l] = r
                indices[r] = l
            heatmap_flip = heatmap[..., ::-1][:, indices]
            heatmap = np.concatenate([heatmap, heatmap_flip])
        return heatmap

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'with_kp={self.with_kp}, '
                    f'with_limb={self.with_limb}, '
                    f'skeletons={self.skeletons}, '
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp})')
        return repr_str

class HeatmapToImage:
    """
    Convert the heatmap data to image data.
    """
    def __init__(self) -> None:
        self.cmap = cm.gray
    
    def __call__(self, heatmaps):
        """
        heatmaps: (T, 17, H, W)
        return images: (T, 1, H, W)
        """
        heatmaps = [x.transpose(1, 2, 0) for x in heatmaps]
        h, w, _ = heatmaps[0].shape
        newh, neww = int(h), int(w)
        heatmaps = [np.max(x, axis=-1) for x in heatmaps]
        heatmaps = [(self.cmap(x)[..., :3] * 255).astype(np.uint8) for x in heatmaps]
        heatmaps = [cv2.resize(x, (neww, newh)) for x in heatmaps]
        return np.ascontiguousarray(np.mean(np.array(heatmaps), axis=-1, keepdims=True).transpose(0,3,1,2))

class CenterAndScaleNormalizer:

    def __init__(self, pose_format="coco", use_conf=True, heatmap_image_height=128) -> None:
        """
        Parameters:
        - pose_format (str): Specifies the format of the keypoints. 
                            This parameter determines how the keypoints are structured and indexed. 
                            The supported formats are "coco" or "openpose-x" where 'x' can be either 18 or 25, indicating the number of keypoints used by the OpenPose model. 
        - use_conf (bool): Indicates whether confidence scores.
        - heatmap_image_height (int): Sets the height (in pixels) for the heatmap images that will be normlization. 
        """
        self.pose_format = pose_format
        self.use_conf = use_conf
        self.heatmap_image_height = heatmap_image_height

    def __call__(self, data):
        """
            Implements step (a) from Figure 2 in the SkeletonGait paper.
            data: (T, V, C)
            - T: number of frames
            - V: number of joints
            - C: dimensionality, where 2 indicates joint coordinates and 1 indicates the confidence score
            return data: (T, V, C)
        """

        if self.use_conf:
            pose_seq = data[..., :-1]
            score = np.expand_dims(data[..., -1], axis=-1)
        else:
            pose_seq = data[..., :-1]
        
        # Hip as the center point
        if self.pose_format.lower() == "coco":
            hip  = (pose_seq[:, 11] + pose_seq[:, 12]) / 2. # [t, 2]
        elif self.pose_format.split('-')[0].lower() == "openpose":
            hip  = (pose_seq[:, 9] + pose_seq[:, 12]) / 2. # [t, 2]
        else:
            raise ValueError(f"Error value for pose_format: {self.pose_format} in CenterAndScale Class.")

        # Center-normalization
        pose_seq = pose_seq - hip[:, np.newaxis, :]

        # Scale-normalization
        y_max = np.max(pose_seq[:, :, 1], axis=-1) # [t]
        y_min = np.min(pose_seq[:, :, 1], axis=-1) # [t]
        pose_seq *= ((self.heatmap_image_height // 1.5) / (y_max - y_min)[:, np.newaxis, np.newaxis]) # [t, v, 2]
        
        pose_seq += self.heatmap_image_height // 2
        
        if self.use_conf:
            pose_seq = np.concatenate([pose_seq, score], axis=-1)
        return pose_seq

class PadKeypoints:
    """
    Pad the keypoints with missing values.
    """

    def __init__(self, pad_method="knn", use_conf=True) -> None:
        """
        pad_method (str): Specifies the method used to pad the missing values.
                        The supported methods are "knn" and "simple".
        use_conf (bool): Indicates whether confidence scores.
        """
        self.use_conf = use_conf
        if pad_method.lower() == "knn":
            self.imputer = KNNImputer(missing_values=0.0, n_neighbors=4, weights="distance", add_indicator=False)
        elif pad_method.lower()  == "simple":
            self.imputer = SimpleImputer(missing_values=0.0, strategy='mean',add_indicator=True)
        else:
            raise ValueError(f"Error value for padding method: {pad_method}")
    
    def __call__(self, raw_data):
        """
        raw_data: (T, V, C)
        - T: number of frames
        - V: number of joints
        - C: dimensionality, where 2 indicates joint coordinates and 1 indicates the confidence score
        return padded_data: (T, V, C)
        """
        T, V, C = raw_data.shape
        if self.use_conf:
            data = raw_data[..., :-1]
            score = np.expand_dims(raw_data[..., -1], axis=-1)
            C = C - 1
        else:
            data = raw_data[..., :-1]
        data = data.reshape((T, V*C))
        padded_data = self.imputer.fit_transform(data)
        try:
            padded_data = padded_data.reshape((T, V, C))
        except:
            padded_data = data.reshape((T, V, C))
        if self.use_conf:
            padded_data = np.concatenate([padded_data, score], axis=-1)
        return padded_data

class COCO18toCOCO17:
    """
    Transfer COCO18 format (Openpose extracted) to COCO17 format
    """

    def __init__(self, transfer_to_coco17=True):
        """
        transfer_to_coco17 (bool): Indicates whether to transfer the keypoints from COCO18 to COCO17 format.
        """
        self.map_dict = {
                0: 0,# "nose",
                1: 15,# "left_eye",
                2: 14,# "right_eye",
                3: 17,# "left_ear",
                4: 16,# "right_ear",
                5: 5,# "left_shoulder",
                6: 2,# "right_shoulder",
                7: 6,# "left_elbow",
                8: 3,# "right_elbow",
                9: 7,# "left_wrist",
                10: 4,# "right_wrist",
                11: 11,# "left_hip",
                12: 8,# "right_hip",
                13: 12,# "left_knee",
                14: 9,# "right_knee",
                15: 13,# "left_ankle",
                16: 10,# "right_ankle"
            }
        self.transfer = transfer_to_coco17
    
    def __call__(self, data):

        """
        data: (T, 18, C)
        - T: number of frames
        - 18: number of joints of COCO18 format
        - C: dimensionality, where 2 indicates joint coordinates and 1 indicates the confidence score
        return data: (T, 17, C)
        """

        if self.transfer:
            """
            input data [T, 18, C] coco18 format
            return data [T, 17, C] coco17 format
            """
            T, _, C = data.shape
            coco17_pkl_data = np.zeros((T, 17, C))
            for i in range(17):
                coco17_pkl_data[:,i,:] = data[:,self.map_dict[i],:]
            return coco17_pkl_data
        else:
            return data

class GatherTransform(object):
    """
    Gather the different transforms.
    """
    def __init__(self, base_transform, transform_bone, transform_joint):

        """
        base_transform: Some common transform, e.g., COCO18toCOCO17, PadKeypoints, CenterAndScale
        transform_bone: GeneratePoseTarget for generate bone heatmap
        transform_joint: GeneratePoseTarget for generate joint heatmap
        """
        self.base_transform = base_transform
        self.transform_bone = transform_bone
        self.transform_joint = transform_joint

    def __call__(self, pose_data):
        x = self.base_transform(pose_data)
        heatmap_bone = self.transform_bone(x) # [T, 1, H, W]
        heatmap_joint = self.transform_joint(x) # [T, 1, H, W]
        heatmap = np.concatenate([heatmap_bone, heatmap_joint], axis=1)
        return heatmap

class HeatmapAlignment():
    def __init__(self, align=True, final_img_size=64, offset=0, heatmap_image_size=128) -> None:
        self.align = align
        self.final_img_size = final_img_size
        self.offset = offset
        self.heatmap_image_size = heatmap_image_size

    def center_crop(self, heatmap):
        """
        Input: [1, heatmap_image_size, heatmap_image_size]
        Output: [1, final_img_size, final_img_size]
        """
        raw_heatmap = heatmap[0]
        if self.align: 
            y_sum = raw_heatmap.sum(axis=1)
            y_top = (y_sum != 0).argmax(axis=0)
            y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
            height = y_btm - y_top + 1
            raw_heatmap = raw_heatmap[y_top - self.offset: y_btm + 1 + self.offset, (self.heatmap_image_size // 2) - (height // 2) : (self.heatmap_image_size // 2) + (height // 2) + 1]
        raw_heatmap = cv2.resize(raw_heatmap, (self.final_img_size, self.final_img_size), interpolation=cv2.INTER_AREA)
        return raw_heatmap[np.newaxis, :, :] # [1, final_img_size, final_img_size]

    def __call__(self, heatmap_imgs):
        """
        heatmap_imgs: (T, 1, raw_size, raw_size)
        return (T, 1, final_img_size, final_img_size)
        """
        heatmap_imgs = heatmap_imgs / 255.
        heatmap_imgs = np.array([self.center_crop(heatmap_img) for heatmap_img in heatmap_imgs]) 
        return (heatmap_imgs * 255).astype('uint8')

def GenerateHeatmapTransform(
    coco18tococo17_args,
    padkeypoints_args,
    norm_args,
    heatmap_generator_args,
    align_args
):

    base_transform = T.Compose([
        COCO18toCOCO17(**coco18tococo17_args),
        PadKeypoints(**padkeypoints_args), 
        CenterAndScaleNormalizer(**norm_args), 
    ])

    heatmap_generator_args["with_limb"] = True
    heatmap_generator_args["with_kp"] = False
    transform_bone = T.Compose([
        GeneratePoseTarget(**heatmap_generator_args), 
        HeatmapToImage(), 
        HeatmapAlignment(**align_args) 
    ])

    heatmap_generator_args["with_limb"] = False
    heatmap_generator_args["with_kp"] = True
    transform_joint = T.Compose([
        GeneratePoseTarget(**heatmap_generator_args), 
        HeatmapToImage(), 
        HeatmapAlignment(**align_args) 
    ])

    transform = T.Compose([
        GatherTransform(base_transform, transform_bone, transform_joint) # [T, 2, H, W]
    ])

    return transform

#########################################################################################################
# The following code is DDP progress codes.
#########################################################################################################
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples


class TransferDataset(Dataset):
    def __init__(self, args, generate_heatemap_cfgs) -> None:
        super().__init__()
        pose_root = args.pose_data_path
        sigma = generate_heatemap_cfgs['heatmap_generator_args']['sigma']
        self.dataset_name = args.dataset_name
        assert self.dataset_name.lower() in ["sustech1k", "grew", "ccpg", "oumvlp", "ou-mvlp", "gait3d", "casiab", "casiae"], f"Invalid dataset name: {self.dataset_name}"
        self.save_root = os.path.join(args.save_root, f"{self.dataset_name}_sigma_{sigma}_{args.ext_name}")
        os.makedirs(self.save_root, exist_ok=True)

        self.heatmap_transform = GenerateHeatmapTransform(**generate_heatemap_cfgs)

        if self.dataset_name.lower() == "sustech1k":
            self.all_ps_data_paths = sorted(glob(os.path.join(pose_root, "*/*/*/03*.pkl")))
        else:
            self.all_ps_data_paths = sorted(glob(os.path.join(pose_root, "*/*/*/*.pkl")))

    def __len__(self):
        return len(self.all_ps_data_paths)
    
    def __getitem__(self, index):
        pose_path = self.all_ps_data_paths[index]
        with open(pose_path, "rb") as f:
            pose_data = pickle.load(f)
            if self.dataset_name.lower() == "grew":
                # print(pose_data.shape)
                pose_data = pose_data[:,2:].reshape(-1, 17, 3)
        
        tmp_split = pose_path.split('/')

        heatmap_img = self.heatmap_transform(pose_data) # [T, 2, H, W]
        
        save_path_pkl = os.path.join(self.save_root, 'pkl', *tmp_split[-4:-1])
        os.makedirs(save_path_pkl, exist_ok=True)

        # save some visualization
        if index < 10:
            # save images
            save_path_img = os.path.join(self.save_root, 'images', *tmp_split[-4:-1])
            os.makedirs(save_path_img, exist_ok=True)
            # save_heatemapimg_index = random.choice(list(range(heatmap_img.shape[0])))
            for save_heatemapimg_index in range(heatmap_img.shape[0]):
                cv2.imwrite(os.path.join(save_path_img, f'bone_{save_heatemapimg_index}.jpg'), heatmap_img[save_heatemapimg_index, 0])
                cv2.imwrite(os.path.join(save_path_img, f'pose_{save_heatemapimg_index}.jpg'), heatmap_img[save_heatemapimg_index, 1])

        pickle.dump(heatmap_img, open(os.path.join(save_path_pkl, tmp_split[-1]), 'wb'))
        return None

def mycollate(_):
    return None


def get_args():
    parser = argparse.ArgumentParser(description='Utility for generating heatmaps from pose data.')
    parser.add_argument('--pose_data_path', type=str, required=True, help="Path to the root directory containing pose data (.pkl files, ID-level) files.")
    parser.add_argument('--save_root', type=str, required=True, help="Root directory where generated heatmap .pkl files will be saved (ID-level).")
    parser.add_argument('--ext_name', type=str, default='', help="Extension name to be appended to the 'save_root' for identification.")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset being preprocessed.")
    parser.add_argument('--heatemap_cfg_path', type=str, default='configs/skeletongait/pretreatment_heatmap.yaml', help="Path to the heatmap generator configuration file.")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed processing, defaults to 0 for non-distributed setups.")
    opt = parser.parse_args()
    return opt

def replace_variables(data, context=None):
    if context is None:
        context = {}

    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = replace_variables(value, context)
    elif isinstance(data, list):
        data = [replace_variables(item, context) for item in data]
    elif isinstance(data, str):
        if data.startswith('${') and data.endswith('}'):
            var_path = data[2:-1].split('.')
            var_value = context
            try:
                for part in var_path:
                    var_value = var_value[part]
                return var_value
            except KeyError:
                raise ValueError(f"Variable {data} not found in context")
    return data

if __name__ == "__main__":
    dist.init_process_group("nccl", init_method='env://')
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    args = get_args()

    # Load the heatmap generator configuration
    with open(args.heatemap_cfg_path, 'r') as stream:
        generate_heatemap_cfgs = yaml.safe_load(stream)
        generate_heatemap_cfgs = replace_variables(generate_heatemap_cfgs, generate_heatemap_cfgs)
    # Create the dataset
    dataset = TransferDataset(args, generate_heatemap_cfgs)

    # Create the dataloader
    dist_sampler = SequentialDistributedSampler(dataset, batch_size=1, rank=local_rank, num_replicas=world_size)
    dataloader = DataLoader(dataset=dataset, batch_size=1, sampler=dist_sampler, num_workers=8, collate_fn=mycollate)
    for _, tmp in tqdm(enumerate(dataloader), total=len(dataloader)):
        pass

    
