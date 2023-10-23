from Trainer import Trainer
from AlexNet import AlexNet
from Resnet import ResNet, ResidualBlock
import torch.nn as nn
import torch
from ray import tune
from ray import train
import wandb
import pyhopper
import sys

print()
# Generated simple objects:
# Set 1 contains 10000 images of each shape: ellipse, triangle, rectangle, star, hexagon
#   The images consist of only black and white pixels and are either filled or not filled.
#   The images are rotated and sized randomly such that they still fit in the picture.

# Set 2 contains images of the same shapes as set 1, but with a random color and a random background color.
#   The set also contains images from the MPEG-7 dataset, which are rotated and sized randomly.

# Configurations and setup
seed = None

if seed is not None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OPTIONS = {
        'dataset_name': sys.argv[2], # 'Cifar10' or 'Brain'
        'architecture': sys.argv[3], # 'ResNet' or 'AlexNet'
        'epochs': 70, # Cifar10: 80, Brain: 70
        'batch_size': 64,
        'learning_rate': 0.0003, # Res_Cifar: 0.006, Alex_Cifar: 0.0002, Alex_brain: 0.001, Res_brain: 0.0003
        'criterion': nn.CrossEntropyLoss(),
        'weight_decay': 0.01, # 0.002 Res_Cifar: 0.002, Alex_Cifar: 0.008, Alex_brain: 0.03, Res_brain: 0.01
        'momentum': 0.9, # Res_Cifar: 0.15, Alex_Cifar: 0.9, Alex_brain: 0.69, Res_brain: 0.9
        'opt': 'sgd',
        'curriculum': True,
        'report_logs': False,
        'should_tune': sys.argv[1] == 'True',
        'scheduler': 'R', # N for no scheduler, B for BabyStep, R for RootP
        'should_restore': False,
        'new_epoch': 0
    }

def experiment(conf):
    if conf['report_logs']:
        wandb.init(project="cifar10",config=conf)

    # This is akin to using the One-Pass scheduler
    if conf['dataset_name'] == 'Cifar10':
        data_dirs = [
            # {'path': './datasets/Generated_Set1', 'classes': 5, 'name': 'Generated1'},
            # {'path': './datasets/Generated_Set2', 'classes': 75, 'name': 'Generated2'},
            {'path': './datasets', 'classes': 10, 'name': 'CIFAR'} 
            # {'path': 'G:\Datasets\Brain_Tumor1_Generated', 'classes': 4, 'name': 'Brain1'} 
            ]
    elif conf['dataset_name'] == 'Brain':
        data_dirs = [
            {'path': './datasets/Generated_Set1', 'classes': 5, 'name': 'Generated1'},
            # {'path': './datasets/Generated_Set2', 'classes': 75, 'name': 'Generated2'},
            {'path': 'G:\Datasets\Brain_Tumor1', 'classes': 4, 'name': 'Brain'} 
            ]
    else:
        data_dirs = [
            # {'path': './datasets/Generated_Set1', 'classes': 5, 'name': 'Generated1'},
            # {'path': './datasets/Generated_Set2', 'classes': 75, 'name': 'Generated2'},
            {'path': 'G:/Datasets/farm_insects_Brain', 'classes': 15, 'name': 'CIFAR'} 
            ]

    if conf['architecture'] == 'ResNet':
        model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=data_dirs[0]['classes'])
    else:
        model = AlexNet(num_classes=data_dirs[0]['classes'])
    trainer = Trainer(data_dirs=data_dirs, model=model, device=device, dataset_name = conf['dataset_name'])

    # should_checkpoint = config.get("should_checkpoint", False)
    loss, model, res_file = trainer.start(conf, tune, wandb)

    if not conf['should_tune']:
        loss, acc = trainer.test(model, conf['batch_size'], conf['criterion'])
        res_file.write("\nUSING TEST SET: ")
        res_file.write("{loss: " + str(loss) + ", acc: " + str(acc) + "}\n")

    return loss

if __name__ == "__main__":
    if OPTIONS['should_tune']:
        OPTIONS['learning_rate'] = pyhopper.float(1e-5, 1e-2, log=True, precision=1)
        OPTIONS['weight_decay'] = pyhopper.float(0.001, 0.1, log=True, precision=1)
        OPTIONS['momentum'] = pyhopper.float(0.1, 0.9, precision=2, init = 0.9)

        search = pyhopper.Search(
            OPTIONS
        )

        search.run(pyhopper.wrap_n_times(experiment,3), "min", "9h", n_jobs='per-gpu', checkpoint_path="r_b.ckpt")
    else:
        for i in range(3):
            experiment(OPTIONS)



# loading the data
# CNN for localisation of (object) text detection -> YOLO network
# Text detection in the wild.

# For OCR there are different final layers -> Only if we have time
# RNN based approach
# Transformer based approach