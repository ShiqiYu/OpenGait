import math
import random
import torch
import torch.distributed as dist
import torch.utils.data as tordata


class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            sample_indices = []
            pid_list = sync_random_sample_list(
                self.dataset.label_set, self.batch_size[0])

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                indices = sync_random_sample_list(
                    indices, k=self.batch_size[1])
                sample_indices += indices

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)


def sync_random_sample_list(obj_list, k):
    if len(obj_list) < k:
        idx = random.choices(range(len(obj_list)), k=k)
        idx = torch.tensor(idx)
    else:
        idx = torch.randperm(len(obj_list))[:k]
    if torch.cuda.is_available():
        idx = idx.cuda()
    torch.distributed.broadcast(idx, src=0)
    idx = idx.tolist()
    return [obj_list[i] for i in idx]


class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if batch_size % world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({})".format(
                world_size, batch_size))

        if batch_size != 1:
            complement_size = math.ceil(self.size / batch_size) * \
                batch_size
            indices += indices[:(complement_size - self.size)]
            self.size = complement_size

        batch_size_per_rank = int(self.batch_size / world_size)
        indx_batch_per_rank = []

        for i in range(int(self.size / batch_size_per_rank)):
            indx_batch_per_rank.append(
                indices[i*batch_size_per_rank:(i+1)*batch_size_per_rank])

        self.idx_batch_this_rank = indx_batch_per_rank[rank::world_size]

    def __iter__(self):
        yield from self.idx_batch_this_rank

    def __len__(self):
        return len(self.dataset)
