import pandas as pd
import matplotlib.pyplot as plt

class Graphing:

    def json_to_df(json_data):
        df = pd.DataFrame(json_data)
        df = df.transpose()
        df = df.reset_index()
        df = df.rename(columns={'index': 'file_name'})
        return df
    
    def plot_accuracy(json_data, axis):
        all_training_dfs = Graphing._extract_training_as_df(json_data)
        mean_df = all_training_dfs.groupby('Epoch').mean()
        std_df = all_training_dfs.groupby('Epoch').std()

        axis.plot(mean_df.index, mean_df['Accuracy'])
        axis.fill_between(mean_df.index, mean_df['Accuracy'] - std_df['Accuracy'], mean_df['Accuracy'] + std_df['Accuracy'], color='blue', alpha=0.2)

    def plot_loss(json_data):
        all_training_dfs = Graphing._extract_training_as_df(json_data)
        mean_df = all_training_dfs.groupby('Epoch').mean()
        std_df = all_training_dfs.groupby('Epoch').std()

        plt.plot(mean_df.index, mean_df['Valid_Loss'])
        plt.fill_between(mean_df.index, mean_df['Valid_Loss'] - std_df['Valid_Loss'], mean_df['Valid_Loss'] + std_df['Valid_Loss'], color='blue', alpha=0.2)
        plt.show()


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
