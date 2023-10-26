import pandas as pd
import matplotlib.pyplot as plt

class Graphing:

    def json_to_df(json_data):
        df = pd.DataFrame(json_data)
        df = df.transpose()
        df = df.reset_index()
        df = df.rename(columns={'index': 'file_name'})
        return df
    
    def plot_accuracy(json_data):
        all_training_dfs = Graphing._extract_training_as_df(json_data)
        mean_df = all_training_dfs.groupby('Epoch').mean()
        std_df = all_training_dfs.groupby('Epoch').std()

        plt.plot(mean_df.index, mean_df['Accuracy'])
        # Plot the average line and shaded region
        # plt.axhline(mean_df.index, mean_df['Valid_Loss'], color='blue', linestyle='--', label='Average')
        plt.fill_between(mean_df.index, mean_df['Accuracy'] - std_df['Accuracy'], mean_df['Accuracy'] + std_df['Accuracy'], color='blue', alpha=0.2)
        plt.show()


    def _extract_training_as_df(json_data):
        all_training_dfs = pd.DataFrame()
        for file in json_data:
            filtered_data = [obj for obj in json_data[file] if "Epoch" in obj and isinstance(obj["Epoch"], int)]
            df = pd.DataFrame(filtered_data)
            print(all_training_dfs)
            all_training_dfs = pd.concat([all_training_dfs, df])
        return all_training_dfs
