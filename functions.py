

def clean_data_train_test_split():

    ## read referrals data, clean and drop some columns
    referrals = pd.read_csv('data/2017_refs.csv', sep='|')
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
    plt.title('Auto-Approval rate & Precision')
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


if __name__ = '__main__':
