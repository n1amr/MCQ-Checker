import os

import pandas as pd


def load_csv(path, continue_=True):
    dataframe = pd.DataFrame(columns=['FileName', 'Mark'])
    if os.path.exists(path) and continue_:
        dataframe = pd.read_csv(path)
    return dataframe


def save_csv(path, dataframe):
    dataframe['Mark'] = dataframe['Mark'].astype('int8')
    dataframe.to_csv(path, index=False)
