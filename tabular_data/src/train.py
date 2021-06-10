import os
import pandas as pd
from sklearn import metrics

from .dispatcher import MODELS
from .utils import *
from sklearn.metrics import accuracy_score, log_loss,classification_report
from sklearn.base import BaseEstimator, TransformerMixin


import pprint
pp = pprint.PrettyPrinter(indent=4)

import sys
sys.path.insert(0, '/home/puneet/Projects/kaggle/tabular_data')

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get('TEST_DATA')
FOLD = os.environ.get('FOLD')
MODEL = os.environ.get('MODEL')

def get_fold_mapping(kfold:int=5)->None:
    results = {}
    for k in range(kfold):
        results[k]= [i for i in range(kfold) if i != k]
    return results

def train(df: pd.DataFrame, clf, kfold:int = 0)-> None:

    fold_mapping = get_fold_mapping()
    print(fold_mapping, type(list(fold_mapping.keys())[0]))

    train_df = df[df.kfold.isin(fold_mapping[kfold])].reset_index(drop=True)
    valid_df = df[df.kfold == kfold].reset_index(drop=True)

    yTrain = train_df.target.values
    yValid = valid_df.target.values

    train_df = train_df.drop(['id','target','kfold'], axis=1)
    valid_df = valid_df.drop(['id','target','kfold'],axis=1)

    valid_df = valid_df[train_df.columns]
    
    # fit the model. model will take care of pipeline
    clf.fit(train_df, yTrain,full=True)

    #preds = final_pipeline.predict_proba(valid_df)
    y_pred = clf.predict(valid_df)
    y_proba = clf.predict_proba(valid_df)

    # dump the whole pipeline

    print(classification_report(yValid, y_pred))
    print("Log loss ", log_loss(yValid, y_proba))
    save_to_file(clf, f'./models/{MODEL}_{FOLD}.pkl')

if __name__ == '__main__':
    df_train = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)

    # drop the first column
    clf = MODELS[MODEL]

    train(df_train, clf, int(FOLD))