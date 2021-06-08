import dill as pickle   
from sklearn.metrics import roc_auc_score
from statistics import mode

def save_to_file(obj, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def load_file(file_path):
  with open(file_path, 'rb') as in_strm:
    return pickle.load(in_strm)

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc

  return roc_auc_dict


import numpy as np 

class VotingClassifier(object):
    """ Implements a voting classifier for pre-trained classifiers"""

    def __init__(self, estimators):
        self.estimators = estimators

    def predict(self, X):
        # get values
        Y = np.zeros([X.shape[0], len(self.estimators)], dtype=float)
        for i, clf in enumerate(self.estimators):
            Y[:, i] = clf.predict_proba(X)
        # apply voting 
        y = np.empty(shape=(X.shape[0]), dtype=object)
        
        for i in range(X.shape[0]):
            y[i] = mode(Y[i,:])
        return y