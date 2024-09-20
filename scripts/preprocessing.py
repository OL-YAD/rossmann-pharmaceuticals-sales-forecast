import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 



# calculate percentage of missing values
def calculate_missing_percentage(dataframe):
    # Determine the total number of elements in the DataFrame
    total_elements = np.prod(dataframe.shape)

    # Count the number of missing values in each column
    missing_values = dataframe.isna().sum()

    # Sum the total number of missing values
    total_missing = missing_values.sum()

    # Compute the percentage of missing values
    percentage_missing = (total_missing / total_elements) * 100

    # Print the result, rounded to two decimal places
    print(f"The dataset has {round(percentage_missing, 2)}% missing values.")


def check_missing_values(df):
    """Check for missing values in the dataset."""
    missing_values = df.isnull().sum()
    missing_percentages = 100 * df.isnull().sum() / len(df)
    column_data_types = df.dtypes
    missing_table = pd.concat([missing_values, missing_percentages, column_data_types], axis=1, keys=['Missing Values', '% of Total Values','Data type'])
    return missing_table.sort_values('% of Total Values', ascending=False).round(2)

def outlier_box_plots(df):
    for column in df:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[column])
        plt.title(f'Box plot of {column}')
        plt.show()
