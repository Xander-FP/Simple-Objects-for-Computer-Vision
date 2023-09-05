from torch.utils.data import Sampler
# This class might not be required.
class CLSampler(Sampler):
    def __init__(self, data_source, batch_size=1, shuffle=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):

        if self.shuffle:
            raise NotImplementedError
        else:
            return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)