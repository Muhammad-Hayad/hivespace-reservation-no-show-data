
import pandas as pd
import numpy as np
import joblib

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error, make_scorer


# Load dataset
df = pd.read_csv("reservations.csv")

# Separate features and target
X = df[["member_tenure_days", "time_slot"]]
y = df["no_show"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            ["member_tenure_days"]
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["time_slot"]
        )
    ]
)

# Define models
models = {
    "Prior Baseline": DummyClassifier(strategy="prior"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# Calculate RMSE using predicted probabilities
def probability_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


rmse_scorer = make_scorer(
    probability_rmse,
    response_method="predict_proba"
)

# Define 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Compare models using training data only
results = []

best_rmse = float("inf")
best_model_name = None
best_pipeline = None

print("\n========== MODEL COMPARISON ==========\n")

print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=rmse_scorer,
        n_jobs=-1
    )

    average_rmse = scores.mean()

    results.append({
        "Model": name,
        "Average CV RMSE": average_rmse
    })

    print(f"{name}: CV RMSE = {average_rmse:.4f}")

    # Select the best real machine learning model
    # Keep Prior Baseline only for comparison
    if name != "Prior Baseline" and average_rmse < best_rmse:
        best_rmse = average_rmse
        best_model_name = name
        best_pipeline = pipeline


# Display comparison table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Average CV RMSE")

print("\n========== CROSS-VALIDATION RESULTS ==========\n")
print(results_df.to_string(index=False))

# Train the best model on training data
best_pipeline.fit(X_train, y_train)

# Evaluate on unseen testing data
test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]

test_rmse = np.sqrt(
    mean_squared_error(y_test, test_probabilities)
)

# Save the best trained pipeline
joblib.dump(best_pipeline, "best_model.pkl")

print("\n========== FINAL MODEL ==========\n")
print(f"Best Model: {best_model_name}")
print(f"Best CV RMSE: {best_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

print("\nBest model saved as best_model.pkl")
