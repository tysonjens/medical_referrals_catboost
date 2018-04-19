## CatBoost for Utilization Management
*Applying Catboost to classify medical referrals as "approved" or "denied".*

by: Tyson Ward


#### Executive Summary

The CatBoost algorithm was able

#### Background

Most large healthcare provider groups are shifting from fee-for-service (FFS) models, where the group makes money each time it performs a service, to value-based models (VBM), where the group takes financial responsibility for patient from insurance companies in return for a monthly premium (PMPM). In VBM groups are more aligned with patients' long term health because they reduce costs by proactively managing patients' health today in the hopes of preventing them from becoming sick in the future.

For medical provider groups with mature VBM, there is a need to monitor what physicians and specialists are requesting for patients to ensure care provided is medically necessary and covered by the plans the group has contracted with. In addition, they need to ensure the patient is being referred to a specialist who is effective and is known to work with other "in-network" facilities and specialists.

This capability is called *utilization management* because the provider group is trying to manage how its patients utilize healthcare resources. The request from a physician to "utilize" resources is documented in a referral.

___

#### Table of Contents
1. [Measures of Success](#measures-of-success)
2. [Data](#data)
      * Feature Engineering
      * Training, Validation and Test
Sets
3. [CatBoost](#catboost)
4. [Results](#results)
5. [Next Steps](#next-steps)

## Measures of Success


1. Precision at 40% auto approval rate
2. Area under the receiver operation characteristc curve
3. Incremental profit

At present the auto-approval rate for referrals is 30% - a new model would need to approve at least 40% of referrals to be worth the implementation effort. The model also needs to be precise - to have very few false positives.  This is because approving referrals that would normally be denied could be costly - worse, it could be for a treatment that isn't medically necessary for the patient.

<img alt="Precision vs. Auto Approval Rate" src="imgs/AA_prec_goal.png" width=''>

<sub><b>Figure: </b> Success is precision > 98% while auto-approvals are greater than 40%. </sub>

## Data

The data includes 2,000,000 referrals to 40 or more specialties during 2017. Each referral is either approved (93%) or denied (7%), but denial rates vary across specialty.

<img alt="Referral Volume by Specialty" src="imgs/vol_by_specs.png" width='600'>

<sub><b>Figure: </b> Success is precision > 98% while auto-approvals are greater than 40%. </sub>

#### Feature Engineering

CatBoost allows for categorical features to be left "as is" - it isn't necessary to one-hot-encode them. But it's important to know what's going on under the hood. The parameter `one_hot_max_size` (it accepts values 1 to 255) serves as the cut-off point for how CatBoost will treat each categorical variable. Features where the number of levels is *less than* the cutoff will be one-hot-encoded. If the number of levels is *greater than* the cutoff, CatBoost transforms them into numerical features per the following:

<img alt="Categorical to Numerical Transformation" src="imgs/cat_to_num.png" width='300'>



## CatBoost

CatBoost was released by Yandex in 2017. While it is open source, documentation is focused on making it easy to use, and less about how it works. Key features/facts include:

* Quite good "out of the box"
* Early Stopping
* Easily handles categorical variables - hence "Cat" - Boost
* Quality output during training
* Won some Kaggle competitions
* Slower to train relative to XGBoost, and LightBoost

## Models

#### Logistic Regression Results

*In a prior project, Logistic Regression Models were tested. Categorical variables (with several thousand levels) were encoded into numerical features based on each level's average of the response variable. Note that this is similar to CatBoost's treatment of categorical variables with many levels.*

<img alt="Logistic Prec AA" src="imgs/AA_prec_test_few2.png" width='400'>

___
<img alt="Logistic Prec AA" src="imgs/ROC_test_few2.png" width='400'>

#### CatBoost - Out of the Box

```python
model_oob = CatBoostClassifier()
```
#### Grid Search

The tutorial from the blog *Effective ML* was used to conduct coarse and refined searches over the following parameters:

* `l2_leaf_reg` - Used for leaf value calculation.
* `iterations` - Number of trees
* `learning_rate`
* `depth` - how deep is each tree

#### CatBoost - Model 1

```python
model1 = CatBoostClassifier(iterations=100, l2_leaf_reg=10,
learning_rate=.5, depth=3, class_weights=class_weight,
use_best_model=True, eval_metric='Accuracy')
```

#### CatBoost - Model 3

```python
model3 = CatBoostClassifier(depth=8, iterations=200, learning_rate=0.1,
l2_leaf_reg=30, class_weights=class_weight, use_best_model=True,
eval_metric='Accuracy')
```


#### CatBoost - Model 1 - production

*If the model is put into production, referrals won't be manually labeled anymore (& assumed to be accurate). Below is outline for a process to tune the model monthly using a small percentage of referrals (random selection) that are held out & manually labeled in order to continually tune the model.*

<img alt="Categorical to Numerical Transformation" src="imgs/toward_imp.png" width='600'>

#### Receiver Operating Characteristic

<img alt="Categorical to Numerical Transformation" src="imgs/roc_auc_compare.png" width='600'>

#### Precision at 40% Auto Approval Rate

<img alt="Categorical to Numerical Transformation" src="imgs/prec_aa_compare.png" width='600'>

#### Profit Curve

A profit curve can help us choose which threshold to set to obtain the largest amount of incremental value from our algorithm. In this case we're looking for a model that can *precisely* identify a large amount of referrals - that is, we want true positives but very, very few false positives. Below is the justification for values of the 4 possible outcomes:


|    Outcome    | Reason |
|--------|-------|
| **True Positive (TP)** | (+$6) Currently the provider spends money to identify approvals - if the algorithm can do it effectively the company can save this budget. |
| **False Positive (FP)** | (-$200) if the algorithm predicts "approve" when the referral should be denied, the company will spend money on the referral that it otherwise wouldn't. The cost will vary greatly, but we assign a somewhat liberal value of -$200 to account for this risk. |
| **False Negative (FN)** | (-$1) Similar to TP, if the algorithm predicts that a referral will be denied, the company will spend money to determine whether is indeed a denial. |
| **True Negative (TN)** | (+$3) When the algorithm effectively identifies a denial, the company can triage the referral more quickly and more effectively. This adds value. |


**Summary - Values for Outcomes**

|        | Act + | Act -   |
|--------|-------|---------|
| Pred + | TP +6 | FP -200 |
| Pred - | FN -1 |  TN +3  |

<img alt="Profit Curve Comparison" src="imgs/prof_fig_compare.png" width='600'>

## Next Steps

#### Acknowledgements

* [Yandex - CatBoost](https://tech.yandex.com/catboost/)
* [Effective ML - blog for gridsearch](https://effectiveml.com/using-grid-search-to-optimise-catboost-parameters.html)
* [Press Release: Yandex launches CatBoost](https://techcrunch.com/2017/07/18/yandex-open-sources-catboost-a-gradient-boosting-machine-learning-librar/)
