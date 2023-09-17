from Trainer import Trainer
from AlexNet import AlexNet
from Resnet import ResNet
import torch.nn as nn
import torch
from ray import tune
from ray import train
import wandb

# Configurations and setup
seed = None

if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OPTIONS = {
        'architecture': 'AlexNet',
        'epochs': 100,
        'num_classes': 5,
        'batch_size': 64,
        'learning_rate': 0.001,
        'criterion': nn.CrossEntropyLoss(),
        'weight_decay': 0.001, 
        'momentum': 0.9,
        'curriculum': True,
        'report_logs': False,
        'should_tune': False,
        'scheduler': 'N' # N for no scheduler, B for BabyStep, R for RootP
    }

def experiment(conf):
    if conf['report_logs']:
        wandb.init(project="cifar10",config=conf)

    # This is akin to using the One-Pass scheduler
    data_dirs = [
        {'path': 'G:\Datasets\Cifar10\Generated', 'classes': 5},
        # {'path': 'G:\Datasets\Cifar10', 'epochs': 20, 'classes': 10} 
        ]

    alexnet = AlexNet(num_classes=data_dirs[0]['classes'])
    trainer = Trainer(data_dirs=data_dirs, model=alexnet, device=device, seed=seed)

    # TODO: For the local pacing function -> Look at creating samplers 
    # should_checkpoint = config.get("should_checkpoint", False)
    trainer.start(conf, tune, wandb)

if OPTIONS['should_tune']:
    OPTIONS['learning_rate'] = tune.grid_search([0.001, 0.005, 0.01])
    # OPTIONS['weight_decay'] = tune.grid_search([0.001, 0.01, 0.1])
    # OPTIONS['momentum'] = tune.grid_search([0.001, 0.01, 0.1])
    analysis = tune.run(
        experiment, 
        config = OPTIONS,
        metric = "_metric/validation_loss",
        mode = "min",
        )
    print("Best config: ", analysis.get_best_config(metric="_metric/validation_loss", mode="min"))
else:
    experiment(OPTIONS)


# When selecting using CL -> Create a sampler with the index mapping
# Also re-read own paper on curriculum learning


# loading the data
# CNN for localisation of (object) text detection -> YOLO network
# Text detection in the wild.

# For OCR there are different final layers -> Only if we have time
# RNN based approach
# Transformer based approach

# For the CL approach we want a pool of easy examples
# Randomly sample from that pool
# Add to that pool when the model has converged on the easier examples

# TODO: Some way to limit the amount of data that is available to the model.