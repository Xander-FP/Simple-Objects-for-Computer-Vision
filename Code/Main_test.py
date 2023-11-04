from Trainer import Trainer
from AlexNet import AlexNet
from Resnet import ResNet, ResidualBlock
import torch.nn as nn
import torch
import torchvision
from ray import tune
from ray import train
import wandb
import pyhopper
import sys
from ViT import ViT
import time

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
        'dataset_name': 'Crop',
        'regression': True,
        'architecture': 'ViT', # 'ViT' or 'AlexNet'
        'epochs': 40, # Cifar10: 80, Brain: 70
        'batch_size': 512,
        'learning_rate': 0.0001, # Res_Cifar: 0.006, Alex_Cifar: 0.0002, Alex_brain: 0.001, Res_brain: 0.0003
        'weight_decay': 0.002, # 0.002 Res_Cifar: 0.002, Alex_Cifar: 0.008, Alex_brain: 0.03, Res_brain: 0.01
        'momentum': 0.15, # Res_Cifar: 0.15, Alex_Cifar: 0.9, Alex_brain: 0.69, Res_brain: 0.9
        'opt': 'sgd',
        'curriculum': False,
        'report_logs': False,
        'should_tune': True,
        'scheduler': 'N', # N for no scheduler, B for BabyStep, R for RootP
        'should_restore': False,
        'new_epoch': 0,
        'image_shape': (3, 280, 280), # channels, height, width
        'patch_num': 7,
        'block_num': 2,
        'hidden_layers_transformer': 12,
        'head_num': 2
    }

def experiment(conf):
    if conf['report_logs']:
        wandb.init(project="cifar10",config=conf)

    # This is akin to using the One-Pass scheduler
    if conf['regression']:
        data_dirs = [
            # {'path': './datasets/Generated_Set1', 'classes': 5, 'name': 'Generated1'},
            # {'path': './datasets/Generated_Set2', 'classes': 75, 'name': 'Generated2'},
            {'path': './datasets', 'classes': 1, 'name': 'Brain'} 
        ]
        conf['criterion'] = nn.MSELoss()
    else:
        data_dirs = [
            # {'path': './datasets/Generated_Set1', 'classes': 5, 'name': 'Generated1'},
            # {'path': './datasets/Generated_Set2', 'classes': 75, 'name': 'Generated2'},
            {'path': './datasets', 'classes': 101, 'name': 'Crop'} 
        ]
        conf['criterion'] = nn.CrossEntropyLoss()

    if conf['architecture'] == 'ViT':
        # model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=data_dirs[0]['classes'])
        # model.load_state_dict(torch.load('./resnet50.pth'), strict=False)
        model = ViT(OPTIONS['image_shape'], OPTIONS['patch_num'], OPTIONS['block_num'], OPTIONS['hidden_layers_transformer'], OPTIONS['head_num'], out_d=data_dirs[0]['classes']).to(device)
    else:
        # model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        # torch.save(model.state_dict(), 'alexnet.pth')
        model = AlexNet(num_classes=data_dirs[0]['classes'])

    trainer = Trainer(data_dirs=data_dirs, model=model, device=device, dataset_name = conf['dataset_name'])

    # should_checkpoint = config.get("should_checkpoint", False)
    loss, model, res_file = trainer.start(conf, tune, wandb)

    if not conf['should_tune']:
        trainer.test(model, conf['batch_size'], conf['criterion'], conf['regression'])

    return loss

if __name__ == "__main__":
    if OPTIONS['should_tune']:
        OPTIONS['learning_rate'] = pyhopper.float(1e-5, 1e-2, log=True, precision=1)
        OPTIONS['weight_decay'] = pyhopper.float(0.001, 0.1, log=True, precision=1)
        OPTIONS['momentum'] = pyhopper.float(0.1, 0.9, precision=2, init = 0.9)

        search = pyhopper.Search(
            OPTIONS
        )

        search.run(pyhopper.wrap_n_times(experiment,1), "min", "9h", n_jobs='per-gpu', checkpoint_path="alex_reg2.ckpt")
    else:
        for i in range(1):
            experiment(OPTIONS)



# loading the data
# CNN for localisation of (object) text detection -> YOLO network
# Text detection in the wild.

# For OCR there are different final layers -> Only if we have time
# RNN based approach
# Transformer based approach