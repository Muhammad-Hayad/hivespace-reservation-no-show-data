# HiveSpace Reservation No-Show Predictor

## Project Overview

This project generates a synthetic dataset of 1,000 coworking space reservations for HiveSpace and uses it to explore machine learning approaches for predicting reservation no-shows.

The project currently includes two parts:

1. **Dataset Generation:** Creates realistic synthetic reservation records.
2. **Model Comparison:** Compares machine learning models using stratified 5-fold cross-validation and evaluates their predictive performance against a simple baseline.

## Task 1: Dataset Generation

### Dataset Information

The generated dataset contains 1,000 records with the following columns:

| Column | Description |
|---|---|
| `member_id` | Unique-style identifier assigned to each member |
| `reservation_date` | Date of the reservation |
| `time_slot` | Reservation time category (Morning, Afternoon, Evening) |
| `member_tenure_days` | Number of days the member has been associated with HiveSpace |
| `no_show` | Reservation outcome: 0 = attended, 1 = no-show |

### Data Generation Logic

- Generates 1,000 synthetic reservations.
- Assigns member IDs and reservation dates.
- Uses three reservation time slots.
- Generates member tenure between 1 and 1,200 days.
- Assigns a higher base no-show probability to newer members.
- Base no-show probability decreases gradually from approximately 35% for brand-new members to 25% after 365 days (one year).
- For members with more than 365 days of tenure, the base no-show probability remains at 25% rather than decreasing further.
- Includes time-slot adjustments and random variation to avoid overly strong correlations.
- Generates binary no-show outcomes using probability-based random sampling.

### No-Show Distribution

The generated dataset contains:

- Attended reservations: 728 (72.8%)
- No-show reservations: 272 (27.2%)

The distribution is not balanced artificially. No-show outcomes are generated through probability-based random sampling.

## Task 2: Model Comparison and Evaluation

### Objective

Compare different classification models using cross-validation and determine whether the available reservation features provide useful predictive information.

### Features and Target

The following features are used:

- `member_tenure_days`
- `time_slot`

The target variable is:

- `no_show`

The `member_id` column is excluded because it is an identifier rather than a meaningful predictive feature. `reservation_date` is also excluded from the current model.

### Data Preparation

- The dataset is split into 80% training data and 20% testing data.
- Stratified splitting preserves the target-class distribution.
- Numerical features are processed using `StandardScaler`.
- Categorical features are processed using `OneHotEncoder`.
- Preprocessing and model training are combined in a Scikit-learn Pipeline.

Preprocessing is performed inside the pipeline during cross-validation. This ensures that scaling and encoding are fitted only on each training fold, helping prevent data leakage.

### Models Compared

Four approaches are evaluated:

1. **Prior Baseline:** `DummyClassifier(strategy="prior")`, used as a simple reference that predicts probabilities based on the class distribution.
2. **Logistic Regression:** A linear classification model.
3. **Decision Tree:** A tree-based classification model.
4. **Random Forest:** An ensemble of decision trees.

### Evaluation Method

- Stratified 5-fold cross-validation is used on the training dataset.
- Models are compared using probability-based RMSE.
- RMSE is calculated from predicted probabilities (`predict_proba`) and actual binary outcomes (0 or 1).

The metric is the square root of the Brier score. Lower values indicate better probability predictions.

### Cross-Validation Results

| Model | Average CV RMSE |
|---|---:|
| Prior Baseline | **0.445249** |
| Logistic Regression | 0.446960 |
| Random Forest | 0.539218 |
| Decision Tree | 0.631357 |

### Results and Interpretation

The Prior Baseline achieved the lowest cross-validation RMSE of **0.445249**.

None of the three machine learning models outperformed the baseline. Logistic Regression was the best-performing real machine learning model, with a CV RMSE of 0.446960.

The difference between the baseline and Logistic Regression is small (0.001711), but the baseline still performs slightly better.

This indicates that the current features (`member_tenure_days` and `time_slot`) do not provide enough predictive signal for the tested models to improve on a simple class-prior prediction.

### Selected Model

Logistic Regression is selected as the best real machine learning model and saved as:

`best_model.pkl`

Its evaluation results are:

- Cross-validation RMSE: 0.446960
- Holdout test RMSE: 0.4461

The Prior Baseline is retained for comparison and is not saved as the selected model.

**Important:** Although Logistic Regression is the best among the three real machine learning models tested, it does not outperform the Prior Baseline. Therefore, it should not be considered a reliable production-ready predictor at this stage.

### Next Steps

Further improvements should focus on feature engineering rather than simply tuning the current models.

Potential future features include:

- Booking lead time
- Day of the week
- Booking channel
- Previous cancellation history
- Member attendance and no-show history
- Reservation time patterns

These features are potential future additions and are not currently included in the synthetic dataset.

## Requirements

- Python 3.11+
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Installation

Clone the repository:

```bash
git clone https://github.com/Muhammad-Hayad/hivespace-reservation-no-show-data.git
cd hivespace-reservation-no-show-data
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

### Generate the Dataset

```bash
python generate_data.py
```

The script generates `reservations.csv` in the project directory and displays a dataset preview, no-show distribution, and no-show rates by tenure group.

### Compare Models

```bash
python compare_models.py
```

The script performs the train/test split, runs stratified 5-fold cross-validation, compares all four approaches, evaluates the selected real model on the holdout test set, and saves the Logistic Regression pipeline as `best_model.pkl`.

## Project Structure

```text
hivespace-reservation-no-show-data/
│
├── generate_data.py
├── compare_models.py
├── reservations.csv
├── best_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Git and GitHub

## Purpose

This project provides a starting point for exploring reservation no-show patterns, generating synthetic operational data, comparing machine learning models, and evaluating predictive performance honestly.

The current evaluation highlights the importance of comparing machine learning models against simple baselines before considering them useful for real-world prediction.
