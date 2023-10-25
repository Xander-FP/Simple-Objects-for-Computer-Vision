# TODO: Check if the sampler data gets updated!!

from GeneralML import split_train_valid_idx
import math
import numpy as np

class Scheduler:
    def __init__(self, data_size):
        self.epoch = 0
        self.data_size = data_size
        self.converged = False

    def adjust_available_data(self, early_stopping, train_loss, valid_loss):
        self.epoch += 1
        print('General stepping - All data available')
        self.converged = True
        if self.epoch < 50:
            return False
        return early_stopping(train_loss, valid_loss)

    def get_initial_indexes(self):
        self.train_idx, self.valid_idx = split_train_valid_idx(list(range(self.data_size)))
        return self.train_idx, self.valid_idx 

class BabyStep(Scheduler):
    def __init__(self, data_size, num_buckets, max_epochs):
        super(BabyStep, self).__init__(data_size)
        self.total_buckets = num_buckets
        self.buckets_added = 1
        self.max_epochs = max_epochs

    def get_initial_indexes(self):
        size = int(np.ceil(self.data_size * (self.buckets_added/self.total_buckets)))
        self.train_idx, self.valid_idx = split_train_valid_idx(list(range(size)))
        return self.train_idx, self.valid_idx 

    def adjust_available_data(self, early_stopping, train_loss, valid_loss):
        self.epoch += 1
        converged = early_stopping(train_loss, valid_loss)
        if converged or self.epoch == self.max_epochs:
            early_stopping.reset()
            if self.buckets_added != self.total_buckets:
                self.buckets_added += 1
            else:
                self.converged = True

            prev_total = len(self.train_idx) + len(self.valid_idx)
            new_total = int(np.ceil(self.data_size * (self.buckets_added/self.total_buckets)))
            print('stepping BabyStep: ' + str(new_total))
            new_idx = list(range(prev_total, new_total))
            new_train, new_valid = split_train_valid_idx(new_idx)
            self.train_idx.extend(new_train)
            self.valid_idx.extend(new_valid)
        return converged

class RootP(Scheduler):
    # @params
    # max_epochs: The maximum number of epochs that the network is allowed to train
    #   This is used to calculate final -> Final resembles the first epoch where all the data is available
    # start: The percentage of the data that is available at the first epoch
    def __init__(self, data_size, max_epochs, start):
        super(RootP, self).__init__(data_size)
        self.final = int(max_epochs * 8/10)
        self.start = start
        self.max_epochs = max_epochs

    def get_initial_indexes(self):
        size = int(np.ceil(self.data_size * self.start))
        self.train_idx, self.valid_idx = split_train_valid_idx(list(range(size)))
        return self.train_idx, self.valid_idx 

    def adjust_available_data(self, early_stopping, train_loss, valid_loss):
        self.epoch += 1
        self.converged = True
        prev_total = len(self.train_idx) + len(self.valid_idx)
        new_total = int(np.ceil(self.data_size * min(1,math.sqrt((1-self.start**2) * self.epoch/self.final + self.start**2))))
        print('stepping RootP: ' + str(new_total))
        new_idx = list(range(prev_total, new_total))
        new_train, new_valid = split_train_valid_idx(new_idx)
        self.train_idx.extend(new_train)
        self.valid_idx.extend(new_valid)

        if self.epoch == self.final:
            return early_stopping(train_loss, valid_loss)
        return False

        
