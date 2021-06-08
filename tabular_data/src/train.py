import os
import pandas as pd
from sklearn import metrics
from sklearn.pipeline import make_pipeline

from .dispatcher import MODELS
from .features import preprocessor
from .utils import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import pprint
pp = pprint.PrettyPrinter(indent=4)


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

    train_df = train_df.drop(['Unnamed: 0','target','kfold'], axis=1)
    valid_df = valid_df.drop(['Unnamed: 0','target','kfold'],axis=1)

    valid_df = valid_df[train_df.columns]
    
    # feature preproessing
    final_pipeline =  make_pipeline(preprocessor, clf)
    final_pipeline.fit(train_df, yTrain)
    

    #preds = final_pipeline.predict_proba(valid_df)
    y_pred = final_pipeline.predict(valid_df)

    # dump the whole pipeline
    print(classification_report(yValid, y_pred))
    save_to_file(final_pipeline, f'./models/{MODEL}_{FOLD}.pkl')

if __name__ == '__main__':
    df_train = pd.read_csv(TRAINING_DATA)
    df_test = pd.read_csv(TEST_DATA)

    # drop the first column
    clf = MODELS[MODEL]

    train(df_train, clf, int(FOLD))