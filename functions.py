
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





def clean_data_train_test_split():

    ## read referrals data, clean and drop some columns
    referrals = pd.read_csv('../data/2017_refs.csv', sep='|')
    referrals['refstat'].replace(['APPROVED', 'REJECTED', 'OTHER'], [1,0,0], inplace=True)
    referrals['refstat'].fillna(1., inplace=True)
    referrals.rename(index=str, columns={"sex": "is_male", "refstat": "is_approve"}, inplace=True)
    referrals.fillna(-999, inplace=True)
    referrals['dater'] =  pd.to_datetime(referrals['dater'], infer_datetime_format=True)
    referrals.drop(['REFERRAL_KEY','plantype', 'patient', 'regdate', 'created_by', 'site_name'], axis = 1, inplace=True)

    ## Train test split chronologically
    test = referrals[referrals['dater']>'2017-08-31']
    train = referrals[referrals['dater']<='2017-08-31']

    train_val = train[train['dater']>'2017-07-31']
    train_train = train[train['dater']<='2017-07-31']

    y_train_train = train_train['is_approve']
    x_train_train = train_train.drop(['is_approve','dater'], axis=1)

    y_train_val = train_val['is_approve']
    x_train_val = train_val.drop(['is_approve','dater'], axis=1)

    y_test = test['is_approve']
    x_test = test.drop(['is_approve','dater'], axis=1)

    return x_train_train, y_train_train, x_train_val, y_train_val, x_test, y_test

def plotroc(FPR, TPR, savestring=None):
    roc_auc = auc(FPR, TPR)
    plt.figure(figsize=(8,6))
    lw = 2
    plt.plot(FPR, TPR, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if savestring==None:
        pass
    else:
        plt.savefig(savestring)


def plot_prec_aa(precs, aarates, savestring=None):
    plt.figure(figsize=(8,6))
    lw = 2
    plt.plot(aarates, precs, color='red',
             lw=lw)
    plt.plot([0, 1], [0.98, 0.98], color='black', lw=lw, linestyle='--')
    plt.plot([.4, .4], [.9, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.xlabel('Auto Approval Rate')
    plt.ylabel('Precision')
    plt.text(0.6, .99, 'Goal', color='green')
    plt.title('Auto-Approval Rate & Precision')
    if savestring==None:
        pass
    else:
        plt.savefig(savestring)


def standard_confusion_matrix(y_true, y_pred):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])


def get_prec_aa_prof(thresholds, y_true, y_proba):
    precs = []
    aarate = []
    n = len(y_true)
    for thresh in thresholds:
        tp, fp, fn, tn = standard_confusion_matrix(y_true, (y_proba > thresh).astype(int)).ravel()
        precs.append(tp/(tp + fp))
        aarate.append((tp+fp)/n)
    return precs, aarate


def profit_curve(cost_benefit_mat, y_pred_proba, y_true):
    """Function to calculate list of profits based on supplied cost-benefit
    matrix and prediced probabilities of data points and thier true labels.

    Parameters
    ----------
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
    labels          : ndarray - 1D, true label of datapoints, 0 or 1

    Returns
    -------
    profits    : ndarray - 1D
    thresholds : ndarray - 1D
    """
    n_obs = float(len(y_true))
    # Make sure that 1 is going to be one of our thresholds

    thresholds = np.linspace(0,1,101)
    profits = []
    for threshold in thresholds:
        y_predict = y_pred_proba >= threshold
        confusion_matrix = standard_confusion_matrix(y_true, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit_mat) * 20 / 1000000
        profits.append(threshold_profit)
    return np.array(profits), np.array(thresholds)

def plot_profit_curve(profthresh, profs):
    proffig = plt.figure(figsize=(12,8))
    ax1 = proffig.add_subplot(111)
    ax1.plot([1,0],[0,0], c='black')
    ax1.set_title("Profit Curve", fontdict=font)
    ax1.set_ylabel('Profit ($1MMs)', fontdict=font)
    ax1.set_xlabel('Thresholds', fontdict=font)
    ax1.plot(profthresh, profs);

def crossvaltest(params,train_set,train_label,cat_dims,n_splits=2):
    kf = KFold(n_splits=n_splits,shuffle=True)
    res = []
    for train_index, test_index in kf.split(train_set):
        train = train_set.iloc[train_index,:]
        test = train_set.iloc[test_index,:]

        labels = train_label.ix[train_index]
        test_labels = train_label.ix[test_index]

        clf = CatBoostClassifier(**params)
        clf.fit(train, np.ravel(labels), cat_features=cat_dims)

        res.append(np.mean(clf.predict(test)==np.ravel(test_labels)))
    return np.mean(res)

def plotfeatureimps(x_train, model):
    featimps = np.hstack([np.array(x_train_train.columns).reshape(-1,1), np.array(model.feature_importances_).reshape(-1,1)])
    featsort = featimps[:,1].argsort()[::-1]
    featimpplt = plt.figure(figsize=(12,4))
    ax3 = featimpplt.add_subplot(111)
    ax3.bar(x = featimps[:,0][featsort],height=featimps[:,1][featsort])
    ax3.set_xticklabels( featimps[:,0][featsort], rotation=45 )
    ax3.set_title('Feature Importances', fontdict=font);

# this function runs grid search on several parameters
def catboost_param_tune(params,train_set,train_label,cat_dims=None,n_splits=2):
    ps = paramsearch(params)
    # search 'border_count', 'l2_leaf_reg' etc. individually
    #   but 'iterations','learning_rate' together
    for prms in chain(
                      ps.grid_search(['l2_leaf_reg']),
                      ps.grid_search(['iterations','learning_rate']),
                      ps.grid_search(['depth'])
                     ):
        res = crossvaltest(prms,train_set,train_label,cat_dims,n_splits)
        # save the crossvalidation result so that future iterations can reuse the best parameters
        ps.register_result(res,prms)
        print('Best score: {}. Best params: {}'.format(ps.bestscore(),ps.bestparam()))
    return ps.bestparam()


if __name__ == '__main__':

    # Get data and run model

    x_train_train, y_train_train, x_train_val, y_train_val, x_test, y_test = clean_data_train_test_split()

    categorical_features_indices = np.where(x_train_train.dtypes != np.float)[0]

    class_weight = [3, .2]

    modcb=CatBoostClassifier(depth=5, iterations=150, learning_rate=0.3, l2_leaf_reg=20, class_weights=class_weight,
                             use_best_model=True, one_hot_max_size=100, rsm=.5)

    modcb.fit(x_train_train, y_train_train,cat_features=categorical_features_indices,eval_set=(x_train_val, y_train_val),plot=True)

    # Generate Predictions

    y_test_proba = modcb.predict_proba(x_test)[:,1]

    ## Visualizations

    font = {'family': 'sanserif',
            'color':  'black',
            'weight': 'normal',
            'size': 20,
            }
    thresholds = np.linspace(0,1,51)

    ## Plot Roc Curve

    FPR, TPR, shresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(FPR, TPR)
    plotroc(FPR, TPR)

    ## Plot Precision at 40%

    precisions, aarates = get_prec_aa_prof(thresholds, y_test, y_test_proba)
    plot_prec_aa(precisions, aarates)

    ## Plot Profit Curve

    profs, prof_thresh = profit_curve(np.array([[6,-150],[0,0]]), y_test_proba[:10000], np.array(y_test)[:10000])
    plot_profit_curve(prof_thresh, profs)

    ## Plot feature importances

    plotfeatureimps(x_train_train, modcb)

    ## show plots
    plt.show()
