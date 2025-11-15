# Train final model with decision tree

import os
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import joblib



# Load configuration
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = Path(cfg["data_dir"]).expanduser()
FILENAME = cfg["filename"]
RANDOM_SEED = cfg["random_seed"]

best_max_depth = 8
best_min_samples_leaf = 5


# define functions to build preprocessor and model pipeline
def build_preprocessor(categorical_cols, numerical_cols, use_log1p=False, use_scaler=False, handle_unknown='ignore'):
    transformers = [
        ('cat', OneHotEncoder(handle_unknown=handle_unknown), categorical_cols)
    ]

    # numeric
    if use_log1p or use_scaler:
        steps = []
        if use_log1p:
            steps.append(('log1p', FunctionTransformer(np.log1p)))
        if use_scaler:
            steps.append(('scaler', StandardScaler()))
        numeric_pipeline = Pipeline(steps)
        transformers.append(('num', numeric_pipeline, numerical_cols))
    else:
        transformers.append(('num', 'passthrough', numerical_cols))

    return ColumnTransformer(transformers)


def build_dt_pipeline(preprocessor, class_weight=None, random_state=RANDOM_SEED, 
                      max_depth=None, min_samples_leaf=1):
    # Construct a DecisionTreeClassifier pipeline given a preprocessor
    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', DecisionTreeClassifier(class_weight=class_weight, max_depth=max_depth,
                                       min_samples_leaf=min_samples_leaf,
                                       random_state=random_state))
    ])






# Load data
df = pd.read_csv(os.path.join(DATA_DIR, FILENAME))

# split columns into numerical and categorical
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
# remove target column from numerical columns
numerical_cols.remove('loan_status')
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# convert all categorical values to lowercase
for col in categorical_cols:
    df[col] = df[col].str.lower()

# fill missing values with median for numerical columns (robust to outliers)
df['person_emp_length'] = df['person_emp_length'].fillna((df['person_emp_length'].median()))
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())


# Split
X_train_full, X_test = train_test_split(df, test_size=0.2, shuffle=True, 
                                        stratify=df['loan_status'], random_state=RANDOM_SEED)

# reset index
X_train_full = X_train_full.reset_index(drop=True)

# separate target
y_train_full = X_train_full['loan_status'].values
y_test = X_test['loan_status'].values

X_train_full = X_train_full.drop(columns=['loan_status'])
X_test = X_test.drop(columns=['loan_status'])


preprocessor = build_preprocessor(
    categorical_cols=categorical_cols,
    numerical_cols=numerical_cols,
    use_log1p=False,
    use_scaler=False
)

model = build_dt_pipeline(
    preprocessor=preprocessor,
    class_weight=None,
    max_depth=best_max_depth,
    min_samples_leaf=best_min_samples_leaf
)

final_model = model.fit(X_train_full, y_train_full)

# save the final model
model_path = "final_model.pkl"
joblib.dump(final_model, model_path)






