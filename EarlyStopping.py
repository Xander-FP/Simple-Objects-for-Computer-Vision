import torch
import os

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.2):
        self.history = []
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                print('Early Stopping: CONVERGED')
                self.early_stop = True
        return self.early_stop
        
    def save_checkpoint(self, model, optimizer, epoch, validation_loss):
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if epoch > self.tolerance:
            delete_path = 'checkpoints/epoch' + str(epoch - self.tolerance + 1) + '.pt'
            os.remove(delete_path)
            self.history.pop(0)
        path = 'checkpoints/epoch' + str(epoch) + '.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        self.history.append({'epoch': epoch, 'validation_loss': validation_loss})

    def load_checkpoint(self, model, optimizer, epoch):
        path = 'checkpoints/epoch' + str(epoch) + '.pt'
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def reset(self):
        self.counter = 0
        self.early_stop = False
        self.history = []