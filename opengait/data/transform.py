import numpy as np
import random
import torchvision.transforms as T
import cv2
import math
from data import transform as base_transform
from utils import is_list, is_dict, get_valid_args


class NoOperation():
    def __call__(self, x):
        return x


class BaseSilTransform():
    def __init__(self, divsor=255.0, img_shape=None):
        self.divsor = divsor
        self.img_shape = img_shape

    def __call__(self, x):
        if self.img_shape is not None:
            s = x.shape[0]
            _ = [s] + [*self.img_shape]
            x = x.reshape(*_)
        return x / self.divsor


class BaseParsingCuttingTransform():
    def __init__(self, divsor=255.0, cutting=None):
        self.divsor = divsor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(x.shape[-1] // 64) * 10
        if cutting != 0: 
            x = x[..., cutting:-cutting]
        if x.max() == 255 or x.max() == 255.:
            return x / self.divsor
        else:
            return x / 1.0


class BaseSilCuttingTransform():
    def __init__(self, divsor=255.0, cutting=None):
        self.divsor = divsor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(x.shape[-1] // 64) * 10
        if cutting != 0: 
            x = x[..., cutting:-cutting]
        return x / self.divsor


class BaseRgbTransform():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485*255, 0.456*255, 0.406*255]
        if std is None:
            std = [0.229*255, 0.224*255, 0.225*255]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

    def __call__(self, x):
        return (x - self.mean) / self.std


# **************** Data Agumentation ****************


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            return seq[..., ::-1]


class RandomErasing(object):
    def __init__(self, prob=0.5, sl=0.05, sh=0.2, r1=0.3, per_frame=False):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.per_frame = per_frame

    def __call__(self, seq):
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq
            else:
                for _ in range(100):
                    seq_size = seq.shape
                    area = seq_size[1] * seq_size[2]

                    target_area = random.uniform(self.sl, self.sh) * area
                    aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))

                    if w < seq_size[2] and h < seq_size[1]:
                        x1 = random.randint(0, seq_size[1] - h)
                        y1 = random.randint(0, seq_size[2] - w)
                        seq[:, x1:x1+h, y1:y1+w] = 0.
                        return seq
            return seq
        else:
            self.per_frame = False
            frame_num = seq.shape[0]
            ret = [self.__call__(seq[k][np.newaxis, ...])
                   for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate(ret, 0)


class RandomRotate(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            dh, dw = seq.shape[-2:]
            # rotation
            degree = random.uniform(-self.degree, self.degree)
            M1 = cv2.getRotationMatrix2D((dh // 2, dw // 2), degree, 1)
            # affine
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 2, 3, 1)
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 3, 1, 2)
            return seq


class RandomPerspective(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            h, w = seq.shape[-2:]
            cutting = int(w // 44) * 10
            x_left = list(range(0, cutting))
            x_right = list(range(w - cutting, w))
            TL = (random.choice(x_left), 0)
            TR = (random.choice(x_right), 0)
            BL = (random.choice(x_left), h)
            BR = (random.choice(x_right), h)
            srcPoints = np.float32([TL, TR, BR, BL])
            canvasPoints = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            perspectiveMatrix = cv2.getPerspectiveTransform(
                np.array(srcPoints), np.array(canvasPoints))
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 2, 3, 1)
            seq = [cv2.warpPerspective(_[0, ...], perspectiveMatrix, (w, h))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 3, 1, 2)
            return seq


class RandomAffine(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            dh, dw = seq.shape[-2:]
            # rotation
            max_shift = int(dh // 64 * 10)
            shift_range = list(range(0, max_shift))
            pts1 = np.float32([[random.choice(shift_range), random.choice(shift_range)], [
                              dh-random.choice(shift_range), random.choice(shift_range)], [random.choice(shift_range), dw-random.choice(shift_range)]])
            pts2 = np.float32([[random.choice(shift_range), random.choice(shift_range)], [
                              dh-random.choice(shift_range), random.choice(shift_range)], [random.choice(shift_range), dw-random.choice(shift_range)]])
            M1 = cv2.getAffineTransform(pts1, pts2)
            # affine
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 2, 3, 1)
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 3, 1, 2)
            return seq
        
# ******************************************

def Compose(trf_cfg):
    assert is_list(trf_cfg)
    transform = T.Compose([get_transform(cfg) for cfg in trf_cfg])
    return transform


def get_transform(trf_cfg=None):
    if is_dict(trf_cfg):
        transform = getattr(base_transform, trf_cfg['type'])
        valid_trf_arg = get_valid_args(transform, trf_cfg, ['type'])
        return transform(**valid_trf_arg)
    if trf_cfg is None:
        return lambda x: x
    if is_list(trf_cfg):
        transform = [get_transform(cfg) for cfg in trf_cfg]
        return transform
    raise "Error type for -Transform-Cfg-"

# **************** For GaitSSB ****************
# Fan, et al: Learning Gait Representation from Massive Unlabelled Walking Videos: A Benchmark, T-PAMI2023

class RandomPartDilate():
    def __init__(self, prob=0.5, top_range=(12, 16), bot_range=(36, 40)):
        self.prob = prob
        self.top_range = top_range
        self.bot_range = bot_range
        self.modes_and_kernels = {
            'RECT': [[5, 3], [5, 5], [3, 5]],
            'CROSS': [[3, 3], [3, 5], [5, 3]],
            'ELLIPSE': [[3, 3], [3, 5], [5, 3]]}
        self.modes = list(self.modes_and_kernels.keys())

    def __call__(self, seq):
        '''
            Using the image dialte and affine transformation to simulate the clorhing change cases.
        Input:
            seq: a sequence of silhouette frames, [s, h, w]
        Output:
            seq: a sequence of agumented frames, [s, h, w]
        '''
        if random.uniform(0, 1) >= self.prob:
                return seq
        else:
            mode = random.choice(self.modes)
            kernel_size = random.choice(self.modes_and_kernels[mode])
            top = random.randint(self.top_range[0], self.top_range[1])
            bot = random.randint(self.bot_range[0], self.bot_range[1])

            seq = seq.transpose(1, 2, 0) # [s, h, w] -> [h, w, s]
            _seq_ = seq.copy()
            _seq_ = _seq_[top:bot, ...]
            _seq_ = self.dilate(_seq_, kernel_size=kernel_size, mode=mode)
            seq[top:bot, ...] = _seq_
            seq = seq.transpose(2, 0, 1) # [h, w, s] -> [s, h, w]
            return seq

    def dilate(self, img, kernel_size=[3, 3], mode='RECT'):
        '''
            MORPH_RECT, MORPH_CROSS, ELLIPSE
        Input:
            img: [h, w]
        Output:
            img: [h, w]
        '''
        assert mode in ['RECT', 'CROSS', 'ELLIPSE']
        kernel = cv2.getStructuringElement(getattr(cv2, 'MORPH_'+mode), kernel_size)
        dst = cv2.dilate(img, kernel)
        return dst

class RandomPartBlur():
    def __init__(self, prob=0.5, top_range=(9, 20), bot_range=(29, 40), per_frame=False):
        self.prob = prob
        self.top_range = top_range
        self.bot_range = bot_range
        self.per_frame = per_frame

    def __call__(self, seq):
        '''
        Input:
            seq: a sequence of silhouette frames, [s, h, w]
        Output:
            seq: a sequence of agumented frames, [s, h, w]
        '''
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq
            else:
                top = random.randint(self.top_range[0], self.top_range[1])
                bot = random.randint(self.bot_range[0], self.bot_range[1])

                seq = seq.transpose(1, 2, 0) # [s, h, w] -> [h, w, s]
                _seq_ = seq.copy()
                _seq_ = _seq_[top:bot, ...]
                _seq_ = cv2.GaussianBlur(_seq_, ksize=(3, 3), sigmaX=0)
                _seq_ = (_seq_ > 0.2).astype(np.float)
                seq[top:bot, ...] = _seq_
                seq = seq.transpose(2, 0, 1) # [h, w, s] -> [s, h, w]

            return seq
        else:
            self.per_frame = False
            frame_num = seq.shape[0]
            ret = [self.__call__(seq[k][np.newaxis, ...]) for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate(ret, 0)

def DA4GaitSSB(
    cutting = None,
    ra_prob = 0.2,
    rp_prob = 0.2,
    rhf_prob = 0.5,
    rpd_prob = 0.2,
    rpb_prob = 0.2,
    top_range = (9, 20),
    bot_range = (39, 50),
):
    transform = T.Compose([
            RandomAffine(prob=ra_prob),
            RandomPerspective(prob=rp_prob),
            BaseSilCuttingTransform(cutting=cutting),
            RandomHorizontalFlip(prob=rhf_prob),
            RandomPartDilate(prob=rpd_prob, top_range=top_range, bot_range=bot_range),
            RandomPartBlur(prob=rpb_prob, top_range=top_range, bot_range=bot_range),
    ])
    return transform

# **************** For pose-based methods ****************
class RandomSelectSequence(object):
    """
    Randomly select different subsequences
    """
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            start = np.random.randint(0, data.shape[0] - self.sequence_length)
        except ValueError:
            raise ValueError("The sequence length of data is too short, which does not meet the requirements.")
        end = start + self.sequence_length
        return data[start:end]


class SelectSequenceCenter(object):
    """
    Select center subsequence
    """
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            start = int((data.shape[0]/2) - (self.sequence_length / 2))
        except ValueError:
            raise ValueError("The sequence length of data is too short, which does not meet the requirements.")
        end = start + self.sequence_length
        return data[start:end]


class MirrorPoses(object):
    """
    Performing Mirror Operations
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if np.random.random() <= self.prob:
            center = np.mean(data[:, :, 0], axis=1, keepdims=True)
            data[:, :, 0] = center - data[:, :, 0] + center

        return data


class NormalizeEmpty(object):
    """
    Normliza Empty Joint
    """
    def __call__(self, data):
        frames, joints = np.where(data[:, :, 0] == 0)
        for frame, joint in zip(frames, joints):
            center_of_gravity = np.mean(data[frame], axis=0)
            data[frame, joint, 0] = center_of_gravity[0]
            data[frame, joint, 1] = center_of_gravity[1]
            data[frame, joint, 2] = 0
        return data


class RandomMove(object):
    """
    Move: add Random Movement to each joint
    """
    def __init__(self,random_r =[4,1]):
        self.random_r = random_r
    def __call__(self, data):
        noise = np.zeros(3)
        noise[0] = np.random.uniform(-self.random_r[0], self.random_r[0])
        noise[1] = np.random.uniform(-self.random_r[1], self.random_r[1])
        data += np.tile(noise,(data.shape[0], data.shape[1], 1))
        return data


class PointNoise(object):
    """
    Add Gaussian noise to pose points
    std: standard deviation
    """
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, data):
        noise = np.random.normal(0, self.std, data.shape).astype(np.float32)
        return data + noise


class FlipSequence(object):
    """
    Temporal Fliping
    """
    def __init__(self, probability=0.5):
        self.probability = probability
    def __call__(self, data):
        if np.random.random() <= self.probability:
            return np.flip(data,axis=0).copy()
        return data


class InversePosesPre(object):
    '''
    Left-right flip of skeletons
    '''
    def __init__(self, probability=0.5, joint_format='coco'):
        self.probability = probability
        if joint_format == 'coco':
            self.invers_arr = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        elif joint_format in ['alphapose', 'openpose']:
            self.invers_arr = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
        else:
            raise ValueError("Invalid joint_format.")
            

    def __call__(self, data):
        for i in range(len(data)):
            if np.random.random() <= self.probability:
                data[i]=data[i,self.invers_arr,:]
        return data


class JointNoise(object):
    """
    Add Gaussian noise to joint
    std: standard deviation
    """

    def __init__(self, std=0.25):
        self.std = std

    def __call__(self, data):
        # T, V, C
        noise = np.hstack((
            np.random.normal(0, self.std, (data.shape[1], 2)),
            np.zeros((data.shape[1], 1))
        )).astype(np.float32)

        return data + np.repeat(noise[np.newaxis, ...], data.shape[0], axis=0)


class GaitTRMultiInput(object):
    def __init__(self, joint_format='coco',):
        if joint_format == 'coco':
            self.connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
        elif joint_format in ['alphapose', 'openpose']:
            self.connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
        else:
            raise ValueError("Invalid joint_format.")

    def __call__(self, data):
        # (C, T, V) -> (I, C * 2, T, V)
        data = np.transpose(data, (2, 0, 1))

        data = data[:2, :, :]

        C, T, V = data.shape
        data_new = np.zeros((5, C, T, V))
        # Joints
        data_new[0, :C, :, :] = data
        for i in range(V):
            data_new[1, :, :, i] = data[:, :, i] - data[:, :, 0]
        # Velocity
        for i in range(T - 2):
            data_new[2, :, i, :] = data[:, i + 1, :] - data[:, i, :]
            data_new[3, :, i, :] = data[:, i + 2, :] - data[:, i, :]
        # Bones
        for i in range(len(self.connect_joint)):
            data_new[4, :, :, i] = data[:, :, i] - data[:, :, self.connect_joint[i]]
        
        I, C, T, V = data_new.shape
        data_new = data_new.reshape(I*C, T, V)
        # (C T V) -> (T V C)
        data_new = np.transpose(data_new, (1, 2, 0))

        return data_new


class GaitGraphMultiInput(object):
    def __init__(self, center=0, joint_format='coco'):
        self.center = center
        if joint_format == 'coco':
            self.connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
        elif joint_format in ['alphapose', 'openpose']:
            self.connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
        else:
            raise ValueError("Invalid joint_format.")

    def __call__(self, data):
        T, V, C = data.shape
        x_new = np.zeros((T, V, 3, C + 2))
        # Joints
        x = data
        x_new[:, :, 0, :C] = x
        for i in range(V):
            x_new[:, i, 0, C:] = x[:, i, :2] - x[:, self.center, :2]
        # Velocity
        for i in range(T - 2):
            x_new[i, :, 1, :2] = x[i + 1, :, :2] - x[i, :, :2]
            x_new[i, :, 1, 3:] = x[i + 2, :, :2] - x[i, :, :2]
        x_new[:, :, 1, 3] = x[:, :, 2]
        # Bones
        for i in range(V):
            x_new[:, i, 2, :2] = x[:, i, :2] - x[:, self.connect_joint[i], :2]
        # Angles
        bone_length = 0
        for i in range(C - 1):
            bone_length += np.power(x_new[:, :, 2, i], 2)
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C - 1):
            x_new[:, :, 2, C+i] = np.arccos(x_new[:, :, 2, i] / bone_length)
        x_new[:, :, 2, 3] = x[:, :, 2]
        return x_new

class GaitGraph1Input(object):
    '''
    Transpose the input
    '''
    def __call__(self, data):
        # (T V C) -> (C T V)
        data = np.transpose(data, (2, 0, 1))
        return data[...,np.newaxis]

class SkeletonInput(object):
    '''
    Transpose the input
    '''
    def __call__(self, data):
        # (T V C) -> (T C V)
        data = np.transpose(data, (0, 2, 1))
        return data[...,np.newaxis]

class TwoView(object):
    def __init__(self,trf_cfg):
        assert is_list(trf_cfg)
        self.transform = T.Compose([get_transform(cfg) for cfg in trf_cfg])
    def __call__(self, data):
        return np.concatenate([self.transform(data), self.transform(data)], axis=1)


class MSGGTransform():
    def __init__(self, joint_format="coco"):
        if joint_format == "coco": #17
            self.mask=[6,8,14,12,7,13,5,10,16,11,9,15]
        elif joint_format in ['alphapose', 'openpose']: #18
            self.mask=[2,3,9,8,6,12,5,4,10,11,7,13]
        else:
            raise ValueError("Invalid joint_format.")
        
    def __call__(self, x):
        result=x[...,self.mask,:].copy()
        return result
