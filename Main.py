from json_reader import JSONFileReader
from graphing import Graphing
import os

RESULTS_PATH = './Results/Output'
ALEX_BRAIN_BASE_1 = 'AlexNet/Brain_Tumor/Base1'

reader = JSONFileReader(os.path.join(RESULTS_PATH,ALEX_BRAIN_BASE_1))

data = reader.read_all_json_files()

Graphing.plot_accuracy(data)

df = Graphing.json_to_df(data)