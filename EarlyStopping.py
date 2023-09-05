import torch
import os

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

    def save_checkpoint(self, model, optimizer, epoch):
        # Check if folder exists
        print('Epoch:' + str(epoch))
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
        if epoch > 4:
            delete_path = 'checkpoints/epoch' + str(epoch - 5) + '.pt'
            os.remove(delete_path)
        path = 'checkpoints/epoch' + str(epoch) + '.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, model, optimizer, epoch):
        path = 'checkpoints/epoch' + str(epoch) + '.pt'
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])