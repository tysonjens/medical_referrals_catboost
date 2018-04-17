## CatBoost for Utilization Management
*Applying Catboost to classify medical referrals as "approved" or "denied".*

by: Tyson Ward


#### Executive Summary

#### Background

Most large healthcare provider groups are shifting from a fee-for-service (FFS) model, where the group makes money each time it performs a service, to value-based model (VBM), where the group takes financial responsibility for patient from insurance companies in return for a monthly fee. In VBM groups are more aligned with patients' long term health because they reduce costs by proactively managing patients' health today in the hopes of preventing them from becoming sick in the future.

For medical provider groups with mature VBM, there is a need to monitor what physicians and specialists are requesting for patients to ensure care provided is medically necessary and covered by the plans the group has contracted with.  In addition, they need to ensure the patient is being referred to a specialist who is effective and is known to work with other "in-network" facilities and specialists.

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

## CatBoost

#### Grid Search

## Results

#### Receiver Operating Characteristic (ROC)

#### Precision at 40% Auto-approval rate

#### Profit Curve



## Next Steps
