from Trainer import Trainer
from AlexNet import AlexNet
from Resnet import ResNet
import torch.nn as nn
import torch
from ray import tune
from ray import train
import wandb
import pyhopper

# To transfer files: scp Main.py xcoetzer@scp.chpc.ac.za:/mnt/lustre/users/xcoetzer/ 
# To transfer files to local: scp xcoetzer@scp.chpc.ac.za:/mnt/lustre/users/xcoetzer/ Downloads

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
        'batch_size': 64,
        'learning_rate': 0.001,
        'criterion': nn.CrossEntropyLoss(),
        'weight_decay': 0.001, 
        'momentum': 0.9,
        'opt': 'sgd',
        'curriculum': True,
        'report_logs': False,
        'should_tune': True,
        'scheduler': 'R' # N for no scheduler, B for BabyStep, R for RootP
    }

def experiment(conf):
    if conf['report_logs']:
        wandb.init(project="cifar10",config=conf)

    # This is akin to using the One-Pass scheduler
    data_dirs = [
        # {'path': 'G:\Datasets\Cifar10\Generated', 'classes': 5, 'name': 'Cifar10Generated'},
        {'path': './datasets', 'classes': 10, 'name': 'Cifar10'} 
        ]

    alexnet = AlexNet(num_classes=data_dirs[0]['classes'])
    trainer = Trainer(data_dirs=data_dirs, model=alexnet, device=device, seed=seed)

    # TODO: For the local pacing function -> Look at creating samplers 
    # should_checkpoint = config.get("should_checkpoint", False)
    best = trainer.start(conf, tune, wandb)
    return best

if __name__ == "__main__":
    if OPTIONS['should_tune']:

        # {
        #     "dropout": pyhopper.float(0, 0.5, precision=1),
        #     "weight_decay": pyhopper.choice([0, 1e-5, 1e-4, 1e-3], is_ordinal=True),
        # }

        OPTIONS['learning_rate'] = pyhopper.float(1e-5, 1e-2, log=True, precision=1)
        OPTIONS['weight_decay'] = pyhopper.float(0.001, 0.1, log=True, precision=1)
        OPTIONS['momentum'] = pyhopper.float(0.1, 0.9, precision=2, init = 0.9)
        # OPTIONS['opt'] = pyhopper.choice(["adam", "sgd"], init="adam")
        search = pyhopper.Search(
            OPTIONS
        )

        search.run(pyhopper.wrap_n_times(experiment,10), "min", "4h", n_jobs='per-gpu', checkpoint_path="new_test.ckpt")
        # search.save("run_completed.ckpt")
    else:
        experiment(OPTIONS)



# loading the data
# CNN for localisation of (object) text detection -> YOLO network
# Text detection in the wild.

# For OCR there are different final layers -> Only if we have time
# RNN based approach
# Transformer based approach