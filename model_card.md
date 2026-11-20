# HiveSpace Reservation No-Show Predictor — Model Card

## 1. Model Overview

| Field | Details |
|---|---|
| Model Name | HiveSpace Reservation No-Show Predictor |
| Model Type | Logistic Regression |
| Problem Type | Binary Classification |
| Intended User | HiveSpace reservation and operations staff |
| Development Dataset | 1,000 synthetic reservation records |
| Current Status | Experimental — not production-ready |

## 2. Purpose

The purpose of this model is to estimate the probability that a member will not attend a scheduled coworking space reservation.

HiveSpace staff could use these predictions to identify reservations that may need confirmation and help manage table availability.

The model is intended to support staff decisions, not automatically cancel reservations, penalize members, or deny access.

## 3. Model Details

The selected model is Logistic Regression.

It uses the following input features:

- **Member Tenure:** The number of days a member has been associated with the service.
- **Time Slot:** The scheduled reservation period (Morning, Afternoon, or Evening).

The target variable is `no_show`:

- `1` indicates that the member did not attend.
- `0` indicates that the member attended.

The `member_id` and `reservation_date` fields are excluded from model training.

## 4. Training Process

The model was developed using a synthetic dataset containing 1,000 reservation records.

The training process included:

1. Splitting the dataset into 80% training data (800 records) and 20% testing data (200 records).
2. Using stratified splitting to preserve the target-class distribution.
3. Applying `StandardScaler` to the numerical feature `member_tenure_days`.
4. Applying `OneHotEncoder` to the categorical feature `time_slot`.
5. Combining preprocessing and Logistic Regression using a Scikit-learn Pipeline.
6. Evaluating models using stratified 5-fold cross-validation on the training data.
7. Selecting Logistic Regression as the best-performing actual machine-learning model.
8. Evaluating the selected pipeline on the separate test dataset.

The preprocessing steps are fitted within each cross-validation training fold to prevent data leakage.

## 5. Performance

The model was evaluated using probability-based RMSE (the square root of the Brier score). Lower values indicate better probability predictions.

### Cross-Validation Results

| Model | Average CV RMSE |
|---|---:|
| Prior Baseline | 0.445249 |
| Logistic Regression | 0.446960 |
| Random Forest | 0.539218 |
| Decision Tree | 0.631357 |

### Selected Model Results

| Metric | Result |
|---|---:|
| Selected Model | Logistic Regression |
| Average CV RMSE | 0.446960 |
| Test RMSE | 0.4461 |

Although Logistic Regression performed better than the other actual machine-learning models tested, the Prior Baseline achieved a slightly lower CV RMSE.

Therefore, the current model has not demonstrated better predictive performance than the simple baseline.

## 6. Limitations

The current model has several limitations:

- **Synthetic Data:** Training data is artificially generated and may not represent real HiveSpace reservation behavior.
- **Limited Features:** Only member tenure and reservation time slot are used. Important factors such as booking lead time, booking channel, and previous attendance history are not included.
- **Limited Predictive Improvement:** Logistic Regression does not outperform the Prior Baseline, so its practical predictive value remains unproven.
- **Limited Dataset Size:** The model was evaluated using only 1,000 synthetic records, which is insufficient to establish reliable real-world performance.
- **Changing Behavior:** Member attendance patterns may change over time, potentially reducing model reliability.

## 7. Ethical Considerations

### Potential Bias Against New Members

Member tenure is one of the model's input features. New members have shorter tenure, and the model may assign them different no-show probabilities based on this characteristic.

This could unfairly disadvantage new members if staff use predictions to restrict reservations or apply penalties.

### Responsible Use

- Predictions should not be used as the sole reason to cancel or reject a reservation.
- All members should receive fair treatment regardless of membership duration.
- Staff should review predictions before making operational decisions.
- The model should be tested for potential bias across different member groups before real-world deployment.
- Real reservation data should be monitored to identify inaccurate predictions or unfair outcomes.

## 8. Intended Use and Future Improvements

This model is currently intended for experimentation and evaluation, not direct production deployment.

Before practical use, future development should focus on:

- Training and evaluating the model using real, representative reservation data.
- Adding booking lead time, booking channel, and previous attendance history.
- Improving predictive performance compared with the Prior Baseline.
- Evaluating fairness across member groups.
- Monitoring model performance as reservation patterns change.

## 9. Conclusion

The HiveSpace No-Show Predictor demonstrates a basic machine-learning approach to estimating reservation no-show probabilities.

The current Logistic Regression model provides a foundation for further experimentation, but its performance does not yet justify production use. Real-world validation, additional features, and fairness evaluation are necessary before using predictions to support operational decisions.
