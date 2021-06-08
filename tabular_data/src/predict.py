import os
import pandas as pd
import numpy as np

from .dispatcher import MODELS
from .utils import *

TEST_DATA = os.environ.get('TEST_DATA')
MODEL = os.environ.get("MODEL")

def predict(test_df, model):
    test_idx = test_df['id'].values
    predictions = []
    estimators = []

    for FOLD in range(5):
        print(f"Predicting for {FOLD}")
        # go through pipeline obj
        clf = load_file(f"./models/{MODEL}_{FOLD}.pkl")
        estimators.append(clf)
    
    # Voting classifier
    eclf = VotingClassifier(estimators=estimators)

    # prediction
    pred = eclf.predict(test_df)
    
    sub = pd.DataFrame(np.column_stack((test_idx, pred)), columns=['id', 'target'])
    return sub

if __name__ == '__main__':
    test_df = pd.read_csv(TEST_DATA)
    clf = MODELS[MODEL]
    submission = predict(test_df, clf)
    submission.loc[:,'id'] = submission.loc[:,'id'].astype(int)
    submission.to_csv(f"./models/{MODEL}.csv",index=False)