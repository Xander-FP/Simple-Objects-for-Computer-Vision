from Trainer import Trainer
from AlexNet import AlexNet
from Resnet import ResNet
import torch.nn as nn
import torch
from ray import tune
from ray import train
import wandb


# Configurations and setup
seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPORT_LOGS = False
SHOULD_TUNE = True

def experiment(conf):
    # Setup auto Hyperparameter tuning with tune
    options = {
        'architecture': 'AlexNet',
        'num_classes': 5,
        'batch_size': 64,
        'learning_rate': conf['lr'] ,
        'criterion': nn.CrossEntropyLoss(),
        'weight_decay': 0.001, 
        'momentum': 0.9,
        'curriculum': False
    }

    if REPORT_LOGS:
        wandb.init(project="cifar10",config=options)

    # This is akin to using the One-Pass scheduler
    data_dirs = [
        {'dataset': 'G:\Datasets\Cifar10\Generated', 'epochs': 10, 'classes': 5},
        {'dataset': 'G:\Datasets\Cifar10', 'epochs': 0, 'classes': 10} 
        ]

    alexnet = AlexNet(num_classes=data_dirs[0]['classes'])
    trainer = Trainer(data_dirs=data_dirs, model=alexnet, device=device, seed=seed)

    # TODO: For the local pacing function -> Look at creating samplers 
    # should_checkpoint = config.get("should_checkpoint", False)
    trainer.train(options, tune, wandb, REPORT_LOGS, SHOULD_TUNE)

# experiment(None)

if SHOULD_TUNE:
    analysis = tune.run(
        experiment, 
        config={
            'lr': tune.grid_search([0.001, 0.005, 0.01]),
            # 'weight_decay': tune.grid_search([0.001, 0.01, 0.1]),
            # 'momentum': tune.grid_search([0.001, 0.01, 0.1])
        },
        metric="_metric/validation_loss",
        mode="min",
        )
    print("Best config: ", analysis.get_best_config(metric="_metric/validation_loss", mode="min"))
else:
    experiment(
        {'lr': 0.001}
    )


# When selecting using CL -> Create a sampler with the index mapping
# Also re-read own paper on curriculum learning


# loading the data
# CNN for localisation of (object)text detection -> YOLO network
# Text detection in the wild.

# For OCR there are different final layers -> Only if we have time
# RNN based approach
# Transformer based approach