import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold

# Pandas setting to display more dataset rows and columns
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Prepare data in folds
if __name__ == '__main__':
    df = pd.read_csv("../../data/tabular/train.csv")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold , (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        print((len(train_idx), len(val_idx)))
        df.loc[val_idx,'kfold'] = fold

    df.to_csv('../../data/tabular/train_fold.csv',index=False)