from torch.utils.data.distributed import DistributedSampler as _DistributedSampler


class DistributedSampler(_DistributedSampler):
    def __init__(self, *args, start_index=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = start_index

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(indices[self.start_index :])

    def __len__(self):
        return self.num_samples - self.start_index
