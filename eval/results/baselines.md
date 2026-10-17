# Baseline results

Fit: time steps 1-29. Validation: 30-34 (early stopping + F1-optimal threshold). Test: 35-49, evaluated once at the validation threshold. Positive class = illicit.

| Model | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test precision | Test recall | Test F1 | Val threshold | Test prevalence |
|---|---|---|---|---|---|---|---|---|
| LogReg (local) | 0.5634 | 0.1833 | 0.8339 | 0.2592 | 0.7498 | 0.3852 | 0.8774 | 0.0650 |
| LightGBM (local) | 0.9814 | 0.7657 | 0.9039 | 0.7822 | 0.7064 | 0.7424 | 0.3167 | 0.0650 |
| LightGBM (all) | 0.9922 | 0.7917 | 0.9275 | 0.9154 | 0.7193 | 0.8056 | 0.7880 | 0.0650 |

Test prevalence is the PR-AUC of a random classifier.
