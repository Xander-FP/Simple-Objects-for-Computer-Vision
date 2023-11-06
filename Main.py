from json_reader import JSONFileReader
from graphing import Graphing
import os
import matplotlib.pyplot as plt

RESULTS_PATH = './Results/Output'
ALEX_CLASS = 'AlexNet/Classification'
ALEX_REG = 'AlexNet/Regression'
VIT_CLASS = 'ViT/Classification'
VIT_REG = 'ViT/Regression'

def plot_train_final(training):
    fig, ax = plt.subplots()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Validation Loss')
    ax.set_title('AlexNet Regression')
    Graphing.plot_loss(training, ax, 'Training')

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

reader = JSONFileReader(os.path.join(RESULTS_PATH,VIT_CLASS))

# reader.fix_all_files()

training = reader.read_all_json_files(os.path.join(RESULTS_PATH,ALEX_REG))

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

# base1 = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_CIFAR_BASE_1))
# base2 = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_CIFAR_BASE_2))
# experiment1 = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_CIFAR_EXPERIMENT_1))
# experiment2 = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_CIFAR_EXPERIMENT_2))
# pretrained = reader.read_all_json_files(os.path.join(RESULTS_PATH,RESNET_CIFAR_PRETRAINED))

plot_train_final(training)
# plot_train_generated(experiment1, experiment2)

# Graphing.show_test_data(base1, 'Base1')
# Graphing.show_test_data(base2, 'Base2')
# Graphing.show_test_data(experiment1, 'Experiment1')
# Graphing.show_test_data(experiment2, 'Experiment2')
# Graphing.show_test_data(pretrained, 'Pretrained')


# Graphing.plot_loss(data)

# df = Graphing.json_to_df(data)