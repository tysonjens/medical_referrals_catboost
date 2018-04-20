

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


thresholds = np.linspace(0,1,51)

categorical_features_indices = np.where(x_train_train.dtypes != np.float)[0]

class_weight = [3, .2]


modcb=CatBoostClassifier(depth=6, iterations=1000, learning_rate=0.01, l2_leaf_reg=20, class_weights=class_weight,
                         use_best_model=True, one_hot_max_size=100, rsm=.5)
modcb.fit(x_train_train, y_train_train,cat_features=categorical_features_indices,eval_set=(x_train_val, y_train_val),plot=True)

y_test_proba = modcb.predict_proba(x_test)[:,1]
FPR, TPR, shresholds = roc_curve(y_test, y_test_proba)

plotroc(FPR, TPR)

precisions, aarates = get_prec_aa_prof(thresholds, y_test, y_test_proba)
plot_prec_aa(precisions, aarates)


profs, prof_thresh = profit_curve(np.array([[6,-150],[-1,3]]), y_test_proba[:10000], np.array(y_test)[:10000])


font = {'family': 'sanserif',
        'color':  'black',
        'weight': 'normal',
        'size': 20,
        }

roc_auc = auc(FPR, TPR)


featimps = np.hstack([np.array(x_train_train.columns).reshape(-1,1), np.array(modcb.feature_importances_).reshape(-1,1)])


featsort = featimps[:,1].argsort()[::-1]

featimpplt = plt.figure(figsize=(12,4))
ax3 = featimpplt.add_subplot(111)
ax3.bar(x = featimps[:,0][featsort],height=featimps[:,1][featsort])
ax3.set_xticklabels( featimps[:,0][featsort], rotation=45 )
ax3.set_title('Feature Importances', fontdict=font);
