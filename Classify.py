import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score
import os

# Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# ------------------------------
# Feature Engineering on Training Data
# ------------------------------

# Map target variable to numeric values
label_map = {'A': 0, 'B': 1}
train_df['target'] = train_df['target'].map(label_map)
train_df = train_df.dropna(subset=['target'])

# Create new features
# Score difference between Team A and B
train_df['score_diff'] = train_df['A_score'] - train_df['B_score']

# Health difference between teams
train_df['health_diff'] = train_df['A_health'] - train_df['B_health']

# Armor difference between teams
train_df['armor_diff'] = train_df['A_armor'] - train_df['B_armor']

# You can add more engineered features based on domain knowledge

# ------------------------------
# Preprocessing
# ------------------------------

# Categorical columns to encode
categorical_cols = ['location', 'device_active']
# Make sure these are strings
train_df[categorical_cols] = train_df[categorical_cols].astype(str)

# Label encoding for categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    label_encoders[col] = le

# Prepare feature matrix and target vector
# Drop id and target columns from features
X = train_df.drop(['id', 'target'], axis=1)
y = train_df['target']

# Impute missing values (using mean strategy)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# ------------------------------
# Hyperparameter Tuning with GridSearchCV
# ------------------------------

# Define parameter grid for XGBoost
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# Use F1-score as our evaluation metric
f1_scorer = make_scorer(f1_score)

# Initialize XGBClassifier
xgb_clf = XGBClassifier(eval_metric='logloss', random_state=42)

# Grid Search
grid_search = GridSearchCV(estimator=xgb_clf,
                           param_grid=param_grid,
                           scoring=f1_scorer,
                           cv=3,
                           verbose=1,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

# Best model from grid search
best_model = grid_search.best_estimator_

# Evaluate on validation set
val_preds = best_model.predict(X_val)
val_f1 = f1_score(y_val, val_preds)
print("Validation F1-Score:", val_f1)

# ------------------------------
# Process Test Data with Same Transformations
# ------------------------------

# Apply same feature engineering to test data
test_df['score_diff'] = test_df['A_score'] - test_df['B_score']
test_df['health_diff'] = test_df['A_health'] - test_df['B_health']
test_df['armor_diff'] = test_df['A_armor'] - test_df['B_armor']

# Encode categorical columns in test data
test_df[categorical_cols] = test_df[categorical_cols].astype(str)
for col in categorical_cols:
    test_df[col] = label_encoders[col].transform(test_df[col])

# Prepare test features and impute missing values
X_test = test_df.drop(['id'], axis=1)
X_test_imputed = imputer.transform(X_test)

# ------------------------------
# Final Predictions and Submission File
# ------------------------------

# Predict using the best model from grid search
test_preds_numeric = best_model.predict(X_test_imputed)

# Convert predictions back to labels ('A' or 'B')
inverse_label_map = {0: 'A', 1: 'B'}
test_preds = [inverse_label_map[pred] for pred in test_preds_numeric]

# Create submission DataFrame
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'target': test_preds
})

submission_path = "/mnt/data/mlX2_submission_improved.csv"
submission_df.to_csv(submission_path, index=False)

# Verify that file is created
if os.path.exists(submission_path):
    print("Submission file created:", submission_path)
    print(submission_df.head())
else:
    print("Error: Submission file not found.")
