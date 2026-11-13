"""PuppetGait's RGB video reader; all other data handling remains in OpenGait."""

import random

import torch
from torch.nn import functional as F

from data import sampler as samplers
from data.collate_fn import CollateFn
from data.dataset import DataSet
from utils import get_attr_from, get_valid_args


def resize_with_padding(image, target_h=256, target_w=128):
    image = image.unsqueeze(0)
    height, width = image.shape[-2:]
    scale = torch.min(torch.tensor([target_h / height, target_w / width]))
    resized_h, resized_w = (height * scale).long(), (width * scale).long()
    image = F.interpolate(
        image, size=(resized_h, resized_w), mode="bilinear", align_corners=False,
    )
    pad_h, pad_w = target_h - resized_h, target_w - resized_w
    return F.pad(
        image,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
    ).squeeze(0)


def remove_black_border_batch(images, threshold=10, target_h=256, target_w=128):
    images = images.float()
    mask = images.mean(dim=1) > threshold
    result = torch.zeros(images.shape[0], 3, target_h, target_w)
    for index, image in enumerate(images):
        if mask[index].any():
            rows = torch.where(mask[index].any(dim=1))[0]
            columns = torch.where(mask[index].any(dim=0))[0]
            image = image[:, rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
        result[index] = resize_with_padding(image, target_h, target_w)
    return result


class RGBDataSet(DataSet):
    def __init__(self, data_cfg, training):
        self.training = training
        super().__init__(data_cfg, training)

    def __loader__(self, paths):
        if all(path.endswith(".pkl") for path in paths):
            return super().__loader__(paths)
        if len(paths) != 1 or not paths[0].endswith(".avi"):
            raise ValueError("Select one RGB AVI or RGB PKL with data_in_use.")
        from torchvision.io import read_video

        video, _, _ = read_video(paths[0], output_format="TCHW", pts_unit="sec")
        if video.shape[0] == 0:
            raise ValueError(f"Empty RGB video: {paths[0]}")
        if self.training:
            # Preserve the research loader's sampling before TripletSampler.
            indices = sorted(random.sample(range(video.shape[0]), min(40, video.shape[0])))
            video = video[indices]
        return [remove_black_border_batch(video).numpy()]


def build_rgb_loader(cfgs, data_cfg, train):
    engine = "trainer_cfg" if train else "evaluator_cfg"
    sampler_cfg = cfgs[engine]["sampler"]
    dataset = RGBDataSet(data_cfg, train)
    sampler_type = get_attr_from([samplers], sampler_cfg["type"])
    arguments = get_valid_args(sampler_type, sampler_cfg, ["sample_type", "type"])
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler_type(dataset, **arguments),
        collate_fn=CollateFn(dataset.label_set, sampler_cfg),
        num_workers=data_cfg["num_workers"],
    )
