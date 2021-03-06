


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functions as func
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
from functions import clean_data_train_test_split
warnings.filterwarnings('ignore')


font_label={'size': 15}
font_title={'weight': 'bold', 'size': 15}

x_train_train, y_train_train, x_train_val, y_train_val, x_test, y_test = clean_data_train_test_split()

thresholds = np.linspace(0,1,101)

class_weight = [3, .2]

categorical_features_indices = np.where(x_train_train.dtypes != np.float)[0]

modcb=CatBoostClassifier(depth=8, iterations=200, learning_rate=0.05, l2_leaf_reg=30, class_weights=class_weight,
                         use_best_model=True, one_hot_max_size=100, rsm=.5)

modcb.fit(x_train_train, y_train_train,cat_features=categorical_features_indices,eval_set=(x_train_val, y_train_val),plot=True)

y_test_proba = modcb.predict_proba(x_test)[:,1]
FPR, TPR, shresholds = roc_curve(y_test, y_test_proba)

func.plotroc(FPR, TPR)

precisions, aarates = func.get_prec_aa_prof(thresholds, y_test, y_test_proba)
func.plot_prec_aa(precisions, aarates)


profs, prof_thresh = func.profit_curve(np.array([[6,-150],[0,0]]), y_test_proba[:10000], np.array(y_test)[:10000])

func.plot_profit_curve(prof_thresh, profs)

roc_auc = auc(FPR, TPR)


featimps = np.hstack([np.array(x_train_train.columns).reshape(-1,1), np.array(modcb.feature_importances_).reshape(-1,1)])


featsort = featimps[:,1].argsort()[::-1]

featimpplt = plt.figure(figsize=(12,4))
ax3 = featimpplt.add_subplot(111)
ax3.bar(x = featimps[:,0][featsort],height=featimps[:,1][featsort])
ax3.set_xticklabels( featimps[:,0][featsort], rotation=45 )
ax3.set_title('Feature Importances', fontdict=font_title);
