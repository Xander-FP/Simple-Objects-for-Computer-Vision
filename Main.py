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

def plot_train_final(base1, base2, experiment1, experiment2, pretrained):
    fig, ax = plt.subplots()
    # Customize the plot (add labels, title, etc.)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Validation Loss')
    # ax.set_title('Resnet50 on Cifar10 Dataset')
    Graphing.plot_loss(base1, ax, 'Base1')
    Graphing.plot_loss(base2, ax, 'Base2')
    Graphing.plot_loss(experiment1, ax, 'Experiment1')
    Graphing.plot_loss(experiment2, ax, 'Experiment2')
    Graphing.plot_loss(pretrained, ax, 'Pretrained')

    ax.legend()
    plt.show()

def plot_train_generated(experiment1, experiment2):
    _, ax1 = plt.subplots()
    _, ax2 = plt.subplots()
    ax1.set_xlabel('Epochs')
    ax2.set_xlabel('Epochs')
    ax1.set_ylabel('Validation Loss')
    ax2.set_ylabel('Validation Loss')
    ax1.set_title('Generated set 1')
    ax2.set_title('Generated set 2')
    Graphing.plot_cl(experiment1, ax1, ax2, 'Experiment1', 'Valid_Loss')
    Graphing.plot_cl(experiment2, ax1, ax2, 'Experiment2', 'Valid_Loss')
    ax1.legend()
    ax2.legend()
    plt.show()

reader = JSONFileReader(os.path.join(RESULTS_PATH,RESNET_CIFAR_PRETRAINED))

# base1 = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_BRAIN_BASE_1))
# base2 = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_BRAIN_BASE_2))
# experiment1 = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_BRAIN_EXPERIMENT_1))
# experiment2 = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_BRAIN_EXPERIMENT_2))
# pretrained = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_BRAIN_PRETRAINED))

# base1 = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_BRAIN_BASE_1))
# base2 = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_BRAIN_BASE_2))
# experiment1 = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_BRAIN_EXPERIMENT_1))
# experiment2 = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_BRAIN_EXPERIMENT_2))
# pretrained = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_BRAIN_PRETRAINED))

# base1 = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_CIFAR_BASE_1))
# base2 = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_CIFAR_BASE_2))
# experiment1 = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_CIFAR_EXPERIMENT_1))
# experiment2 = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_CIFAR_EXPERIMENT_2))
# pretrained = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_CIFAR_PRETRAINED))

base1 = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_CIFAR_BASE_1))
base2 = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_CIFAR_BASE_2))
experiment1 = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_CIFAR_EXPERIMENT_1))
experiment2 = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_CIFAR_EXPERIMENT_2))
pretrained = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_CIFAR_PRETRAINED))

# plot_train_final(base1, base2, experiment1, experiment2, pretrained)
# plot_train_generated(experiment1, experiment2)

Graphing.show_test_data(base1, 'Base1')
Graphing.show_test_data(base2, 'Base2')
Graphing.show_test_data(experiment1, 'Experiment1')
Graphing.show_test_data(experiment2, 'Experiment2')
Graphing.show_test_data(pretrained, 'Pretrained')


# Graphing.plot_loss(data)

# df = Graphing.json_to_df(data)