from json_reader import JSONFileReader
from graphing import Graphing
import os
import matplotlib.pyplot as plt

RESULTS_PATH = './Results/Output'
ALEX_BRAIN_BASE_1 = 'AlexNet/Brain_Tumor/Base1'
ALEX_BRAIN_BASE_2 = 'AlexNet/Brain_Tumor/Base2'
ALEX_BRAIN_EXPERIMENT_1 = 'AlexNet/Brain_Tumor/Experiment1'
ALEX_BRAIN_EXPERIMENT_2 = 'AlexNet/Brain_Tumor/Experiment2'
ALEX_BRAIN_PRETRAINED = 'AlexNet/Brain_Tumor/Pretrained'
ALEX_CIFAR_BASE_1 = 'AlexNet/Cifar/Base1'
ALEX_CIFAR_BASE_2 = 'AlexNet/Cifar/Base2'
ALEX_CIFAR_EXPERIMENT_1 = 'AlexNet/Cifar/Experiment1'
ALEX_CIFAR_EXPERIMENT_2 = 'AlexNet/Cifar/Experiment2'
ALEX_CIFAR_PRETRAINED = 'AlexNet/Cifar/Pretrained'

RESNET_BRAIN_BASE_1 = 'ResNet/Brain_Tumor/Base1'
RESNET_BRAIN_BASE_2 = 'ResNet/Brain_Tumor/Base2'
RESNET_BRAIN_EXPERIMENT_1 = 'ResNet/Brain_Tumor/Experiment1'
RESNET_BRAIN_EXPERIMENT_2 = 'ResNet/Brain_Tumor/Experiment2'
RESNET_BRAIN_PRETRAINED = 'ResNet/Brain_Tumor/Pretrained'
RESNET_CIFAR_BASE_1 = 'ResNet/Cifar/Base1'
RESNET_CIFAR_BASE_2 = 'ResNet/Cifar/Base2'
RESNET_CIFAR_EXPERIMENT_1 = 'ResNet/Cifar/Experiment1'
RESNET_CIFAR_EXPERIMENT_2 = 'ResNet/Cifar/Experiment2'
RESNET_CIFAR_PRETRAINED = 'ResNet/Cifar/Pretrained'

reader = JSONFileReader(os.path.join(RESULTS_PATH,ALEX_BRAIN_EXPERIMENT_1))

reader.fix_all_files()

# base1 = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_BRAIN_BASE_1))
# base2 = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_BRAIN_BASE_2))
# experiment1 = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_BRAIN_EXPERIMENT_1))
# experiment2 = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_BRAIN_EXPERIMENT_2))
# pretrained = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_BRAIN_PRETRAINED))

# fig, ax = plt.subplots()

# Add a legend
# ax.legend()

# Customize the plot (add labels, title, etc.)
# ax.set_xlabel('X-axis Label')
# ax.set_ylabel('Y-axis Label')
# ax.set_title('Combined Graphs with Legend')
# Graphing.plot_accuracy(data)
# Graphing.plot_loss(data)

# df = Graphing.json_to_df(data)