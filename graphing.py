import pandas as pd
import matplotlib.pyplot as plt

class Graphing:

    def json_to_df(json_data):
        df = pd.DataFrame(json_data)
        df = df.transpose()
        df = df.reset_index()
        df = df.rename(columns={'index': 'file_name'})
        return df
    
    def plot_accuracy(json_data, axis, label):
        all_training_dfs = Graphing._extract_training_as_df(json_data)
        mean_df = all_training_dfs.groupby('Epoch').mean()
        std_df = all_training_dfs.groupby('Epoch').std()

        axis.plot(mean_df.index, mean_df['Accuracy'], label=label)
        axis.fill_between(mean_df.index, mean_df['Accuracy'] - std_df['Accuracy'], mean_df['Accuracy'] + std_df['Accuracy'], alpha=0.2)

    def plot_loss(json_data, axis, label):
        all_training_dfs = Graphing._extract_training_as_df(json_data)
        mean_df = all_training_dfs.groupby('Epoch').mean()
        std_df = all_training_dfs.groupby('Epoch').std()

        axis.plot(mean_df.index, mean_df['Valid_Loss'], label=label)
        axis.fill_between(mean_df.index, mean_df['Valid_Loss'] - std_df['Valid_Loss'], mean_df['Valid_Loss'] + std_df['Valid_Loss'], alpha=0.2)

    def plot_cl(json_data, axis1, axis2, label, attribute):
        generated1_df, generated2_df = Graphing._extract_cl_as_df(json_data)
        mean_df1 = generated1_df.groupby('Epoch').mean()
        std_df1 = generated1_df.groupby('Epoch').std()
        mean_df2 = generated2_df.groupby('Epoch').mean()
        std_df2 = generated2_df.groupby('Epoch').std()

        axis1.plot(mean_df1.index, mean_df1[attribute], label=label)
        axis2.plot(mean_df2.index, mean_df2[attribute], label=label)

        axis1.fill_between(mean_df1.index, mean_df1[attribute] - std_df1[attribute], mean_df1[attribute] + std_df1[attribute], alpha=0.2)
        axis2.fill_between(mean_df2.index, mean_df2[attribute] - std_df2[attribute], mean_df2[attribute] + std_df2[attribute], alpha=0.2)

    def show_test_data(json_data, label):
        test_dfs = Graphing._extract_test_as_df(json_data)
        print('*******************' + label + '*******************')
        print(test_dfs['acc'].to_string(index=False))
        mean = test_dfs.mean()
        std = test_dfs.std()
        max = test_dfs.max()
        median = test_dfs.median()

        with open('test_data.txt', 'a') as file:
            file.write('*******************' + label + '*******************\n')
            file.write('Mean:\n')
            file.write(str(mean))
            file.write('\nMedian:\n')
            file.write(str(median))
            file.write('\nStd:\n')
            file.write(str(std))
            file.write('\nMax:\n')
            file.write(str(max))
            file.write('\n\n')

    def _extract_test_as_df(json_data):
        test_dfs = pd.DataFrame()
        for file in json_data:
            data = json_data[file]
            filtered_data = [obj['test'] for obj in data if "test" in obj]

            df = pd.DataFrame(filtered_data)
            test_dfs = pd.concat([test_dfs, df])
        return test_dfs

    def _extract_cl_as_df(json_data):
        generated_dfs1 = pd.DataFrame()
        generated_dfs2 = pd.DataFrame()
        for file in json_data:
            data1, data2 = Graphing._get_initial_train(json_data[file])
            filtered_data1 = [obj for obj in data1 if "Epoch" in obj and isinstance(obj["Epoch"], int)]
            filtered_data2 = [obj for obj in data2 if "Epoch" in obj and isinstance(obj["Epoch"], int)]

            df1 = pd.DataFrame(filtered_data1)
            df2 = pd.DataFrame(filtered_data2)
            generated_dfs1 = pd.concat([generated_dfs1, df1])
            generated_dfs2 = pd.concat([generated_dfs2, df2])
        return generated_dfs1, generated_dfs2

    def _extract_training_as_df(json_data):
        all_training_dfs = pd.DataFrame()
        for file in json_data:
            if len(json_data[file]) > 150:
                data = Graphing._get_final_train(json_data[file])
            else:
                data = json_data[file]
            filtered_data = [obj for obj in data if "Epoch" in obj and isinstance(obj["Epoch"], int)]

            df = pd.DataFrame(filtered_data)
            all_training_dfs = pd.concat([all_training_dfs, df])
        return all_training_dfs
    
    def _get_initial_train(json_data):
        data1 = []
        data2 = []
        sets = 0
        for obj in json_data:
            if sets == 0 or "test" in obj:
                data1.append(obj)
            if sets == 1 or "test" in obj:
                data2.append(obj)
            if "Epoch" in obj and obj["Epoch"] == "None":
                sets += 1
        return data1, data2
    
    def _get_final_train(json_data):
        data = []
        sets = 0
        for obj in json_data:
            if sets > 1:
                data.append(obj)
            if "Epoch" in obj and obj["Epoch"] == "None":
                sets += 1
        return data

    def get_test_loss(json_data):
        all_test_dfs = Graphing._extract_test_as_df(json_data)
        mean_df = all_test_dfs.groupby('Epoch').mean()
        std_df = all_test_dfs.groupby('Epoch').std()

        print(all_test_dfs)
