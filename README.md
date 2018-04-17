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


1. Area under the receiver operation characteristc curve (AUC-ROC)
2. Precision at 40% auto approval rate
3. Incremental profit

At present the auto-approval rate for referrals is 30% - a new model would need to approve at least 40% of referrals to be worth the implementation effort. The model also needs to be precise - to have very few false positives.  This is because approving referrals that would normally be denied could be costly - worse, it could be for a treatment that isn't medically necessary for the patient.

<img alt="Prescision vs. Auto Approval Rate" src="imgs/AA_prec_goal.png" width='400'>

<sub><b>Figure: </b> Success is precision > 98% while auto-approvals are greater than 40%. </sub>

## Data

#### Feature Engineering

CatBoost allows for categorical features to be left "as is" - it isn't necessary to one-hot-encode them. But it's important to know what's going on under the hood. The parameter `one_hot_max_size` (it accepts values 1 to 255) serves as the cut-off point for how CatBoost will treat each categorical variable. Features where the number of levels is *less than* the cutoff will be one-hot-encoded. If the number of levels is *greater than* the cutoff, CatBoost transforms them into numerical features per the following:

<img alt="Categorical to Numerical Transformation" src="imgs/cat_to_num.png" width='400'>



## CatBoost

CatBoost was released by Yandex in 2017

#### Grid Search

The tutorial from the blog *Effective ML* was used to conduct coarse and refined searches over the following parameters:

* `l2_leaf_reg`
* `iterations`
* `learning_rate`
* `depth`

## Results

#### Logistic Regression Results

#### CatBoost - Out of the Box

#### CatBoost - Parameters Tuned


#### Receiver Operating Characteristic (ROC)

#### Precision at 40% Auto-approval rate

#### Profit Curve

A profit curve can help us choose which threshold to set to obtain the largest amount of incremental value from our algorithm. In this case we're looking for a model that can *precisely* identify a large amount of referrals - that is, we want true positives but very, very few false positives. Below is the justification for values of the 4 possible outcomes:

* **True Positive (TP)** - (+$6) Currently the provider spends money to identify approvals - if the algorithm can do it effectively the company can save this budget.
* **False Positive (FP)** - (-$200) if the algorithm predicts "approve" when the referral should be denied, the company will spend money on the referral that it otherwise wouldn't. The cost will vary greatly, but we assign a somewhat liberal value of -$200 to account for this risk.
* **False Negative (FN)** - (-$1) Similar to TP, if the algorithm predicts that a referral will be denied, the company will spend money to determine whether is indeed a denial.
* **True Negative (TN)** - (+$3) When the algorithm effectively identifies a denial, the company can triage the referral more quickly and more effectively. This adds value.

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

## Next Steps

#### Acknowledgements

* [Yandex - CatBoost](https://tech.yandex.com/catboost/)
* [Effective ML - blog for gridsearch](https://effectiveml.com/using-grid-search-to-optimise-catboost-parameters.html)
* [Press Release: Yandex launches CatBoost](https://techcrunch.com/2017/07/18/yandex-open-sources-catboost-a-gradient-boosting-machine-learning-librar/)
