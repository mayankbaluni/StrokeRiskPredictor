import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load the datasets
train_set = pd.read_csv('/path/to/stroke_train_set.csv')
test_set = pd.read_csv('/path/to/stroke_test_set_nogt.csv')

# Data Preprocessing
# Separating the target variable and features in the training set
X = train_set.drop('stroke', axis=1)
y = train_set['stroke']

# Identifying numerical and categorical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Creating a preprocessor with transformations for different column types
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Creating a pipeline with preprocessing and a classifier
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(random_state=42))])

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Predict on validation set and evaluate
y_pred = model.predict(X_val)
f1 = f1_score(y_val, y_pred)
print(f"F1 Score: {f1}")
print(classification_report(y_val, y_pred))

# Hyperparameter Tuning and Alternative Model
model_gb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3, 5]
}

grid_search = GridSearchCV(model_gb, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and model performance
best_params = grid_search.best_params_
y_pred_gb = grid_search.predict(X_val)
f1_gb = f1_score(y_val, y_pred_gb)
print(f"Best Parameters: {best_params}")
print(f"F1 Score (Gradient Boosting): {f1_gb}")
print(classification_report(y_val, y_pred_gb))

# Preparing final predictions on the test set
final_predictions = model.predict(test_set)  # Using the best model from above
submission_df = pd.DataFrame({'ID': test_set.index, 'stroke': final_predictions})

# Saving the submission file
submission_df.to_csv('/path/to/final_submission.csv', index=False)
