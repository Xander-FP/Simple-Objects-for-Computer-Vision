import numpy as np

def split_train_valid_idx(total_set, valid_size = 0.1):
        split = int(np.floor(valid_size * len(total_set)))
        train_idx, valid_idx = total_set[split:], total_set[:split]
        return train_idx, valid_idx