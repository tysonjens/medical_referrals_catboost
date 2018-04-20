
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from regression_tools.dftransformers import (
    ColumnSelector, Identity, FeatureUnion, MapFeature, Intercept)
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_score
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from paramsearch import paramsearch
from itertools import product,chain
import warnings
import pickle,os
warnings.filterwarnings('ignore')

params = {'depth':[3,5,8,12],
          'iterations':[100,250,1000],
          'learning_rate':[0.05,0.1,0.5],
          'l2_leaf_reg':[10,20],
          'thread_count':4,
          'one_hot_max_size':100
         }

categorical_features_indices = np.where(x_train_train.dtypes != np.float)[0]

bestparams = catboost_param_tune(params,x_train_train,y_train_train, categorical_features_indices)
