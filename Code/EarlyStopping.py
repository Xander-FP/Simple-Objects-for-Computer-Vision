import torch
import os

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.02, name='default'):
        self.history = []
        self.prev_loss = float('inf')
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.name = name

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta and validation_loss > self.prev_loss:
            self.counter +=1
            if self.counter >= self.tolerance:  
                print('Early Stopping: CONVERGED')
                self.early_stop = True
        self.prev_loss = validation_loss
        return self.early_stop
        
    def save_checkpoint(self, model, optimizer, epoch, validation_loss):
        if not os.path.exists('checkpoints' + self.name):
            os.makedirs('checkpoints' + self.name)
        if epoch > self.tolerance:
            delete_path = 'checkpoints' + self.name + '/epoch' + str(epoch - self.tolerance - 1) + '.pt'
            if os.path.exists(delete_path):
                os.remove(delete_path)
                self.history.pop(0)
        path = 'checkpoints' + self.name + '/epoch' + str(epoch) + '.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        self.history.append({'epoch': epoch, 'validation_loss': validation_loss})

    def load_checkpoint(self, model, optimizer, epoch, restore = False):
        if restore:
            path = 'checkpointsRestore' + '/epoch' + str(epoch) + '.pt'
        else:
            path = 'checkpoints' + self.name + '/epoch' + str(epoch) + '.pt'
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def reset(self):
        self.counter = 0
        self.early_stop = False
        self.history = []