# Titanic: Machine Learning from Disaster
A professional machine learning workflow to predict Titanic passenger survival, leveraging data preprocessing, feature engineering, and gradient boosting (CatBoost) for high-performance results.

## Project Overview
The sinking of the Titanic is one of the most infamous maritime disasters in history. This project aims to predict which passengers survived the tragedy using historical data (e.g., age, gender, class, fare) through a structured ML pipeline.

### Key Objectives
- Clean and preprocess raw data to handle missing values, outliers, and categorical variables.
- Engineer meaningful features to improve model predictive power.
- Train and tune a robust classifier (CatBoost) to predict survival.
- Generate a submission-ready file for Kaggle competition.

## Dataset
The dataset (from Kaggle's [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)) includes:
- `train.csv`: 891 rows (passengers) with 21 features + target (`Survived`).
- `test.csv`: 418 rows (unlabeled passengers) for submission.


### Key Features
| Feature         | Description                                  |
|-----------------|----------------------------------------------|
| `Pclass`        | Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)  |
| `Sex`           | Passenger gender                             |
| `Age`           | Passenger age                                |
| `SibSp`         | Number of siblings/spouses aboard             |
| `Parch`         | Number of parents/children aboard             |
| `Fare`          | Ticket fare                                  |
| `Embarked`      | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

## Project Structure
```
titanic-survival-prediction/
├── train.csv               # Training dataset
├── test.csv                # Test dataset
├── titanic_model.ipynb     # Full ML workflow (Jupyter Notebook)
├── submission.csv          # Generated submission file
└── README.md               # Project documentation
```

## Installation
### Prerequisites
- Python 3.8+
- Required libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imblearn catboost
```

## Methodology
### 1. Data Loading & Exploration
Load datasets, inspect structure, and identify missing values/duplicates.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Check missing values
null_vals = train.isna().sum().sort_values(ascending=False)
print("Train Missing Values:\n", null_vals[null_vals > 0])

```

### 2. Data Preprocessing
#### 2.1 Drop Irrelevant Columns
Remove low-information or redundant columns:
```python
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train['Survived'].values
test_IDs = test['PassengerId'].copy()

# Combine data for unified preprocessing
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop([
    'Survived', 'PassengerId', 'Cabin', 'Body', 'Ticket',
    'Age_wiki', 'Name_wiki', 'WikiId', 'Hometown', 'Destination'
], axis=1, inplace=True)
```

#### 2.2 Handle Missing Values
- Fill high-missing columns with placeholder (`Lifeboat` → "None")
- Mode-impute low-missing columns (grouped by `Pclass`/`Sex`)
- Median-impute `Age` (grouped by `Pclass`/`Sex`/`Title`):
```python
# Fill Lifeboat with "None"
all_data['Lifeboat'] = all_data['Lifeboat'].fillna('None')

# Mode-impute low-missing columns
low_null_cols = null_vals[(null_vals < 10) & (null_vals > 0)].index
for col in low_null_cols:
    all_data[col] = all_data.groupby(["Pclass", "Sex"])[col].transform(
        lambda x: x.fillna(x.mode()[0])
    )

# Extract Title for Age imputation
all_data['Title'] = all_data['Name'].apply(lambda x: x.split(' ')[1].strip('123,./!?'))
all_data['Title'] = all_data['Title'].apply(lambda x: x if x in ['Mr','Miss','Mrs','Master'] else 'NoTitle')

# Median-impute Age
all_data['Age'] = all_data.groupby(['Pclass', 'Sex', 'Title']).Age.apply(lambda x: x.fillna(x.median()))
```

#### 2.3 Feature Engineering
Create family-related features:
```python
# Family size (SibSp + Parch + 1 for self)
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
# Indicator for solo travelers
all_data['IsAlone'] = all_data['FamilySize'] <= 1
# Drop Name (Title extracted)
all_data.drop('Name', axis=1, inplace=True)
```

#### 2.4 Categorical Encoding
Encode categorical features for modeling:
```python
# Identify categorical columns
categorical_col = []
for col in all_data.columns:
    if all_data[col].dtype in [object, bool] and len(all_data[col].unique()) <= 50:
        categorical_col.append(col)

# Label encode (for CatBoost compatibility)
for col in categorical_col:
    all_data[col] = all_data[col].astype("category").cat.codes

# Optional: One-Hot Encoding (uncomment if using non-tree-based models)
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder(categories='auto')
# array_hot_encoded = ohe.fit_transform(all_data[categorical_col]).toarray()
# data_hot_encoded = pd.DataFrame(array_hot_encoded, index=all_data.index)
# all_data = pd.concat([data_hot_encoded, all_data.drop(columns=categorical_col)], axis=1)
```

#### 2.5 Feature Selection
Select top correlated features with survival:
```python
# Split back to train/test
all_data.columns = all_data.columns.astype(str)
train_data = all_data[:ntrain].copy()
test_data = all_data[ntrain:]
train_data['Survived'] = y_train
train_data['Survived'] = train_data['Survived'].astype("category").cat.codes

# Select top 40 correlated features
k = 40
cols = train_data.corr().abs().nlargest(k, 'Survived')['Survived'].index
train_data = train_data[cols]
test_data = test_data[cols.drop('Survived')]
```

### 3. Outlier & Class Imbalance Handling
```python
# Remove outliers with Local Outlier Factor (LOF)
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(contamination=0.02)
outliers = clf.fit_predict(train_data)
train_data_cleaned = train_data[np.where(outliers == 1, True, False)]

# Balance classes with SMOTE
from imblearn.over_sampling import SMOTE
X_train = train_data_cleaned.drop('Survived', axis=1)
y_train = train_data_cleaned.Survived
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
X_train = X_resampled
y_train = y_resampled
```

### 4. Model Training & Tuning
Use CatBoost (gradient boosting) with Grid Search:
```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from catboost import CatBoostClassifier

# Initialize model
cat = CatBoostClassifier(eval_metric='Accuracy', verbose=0)

# Hyperparameter grid
params_catB = {
    'learning_rate': [0.01, 0.02, 0.03],
    'depth': [6, 7],
    'iterations': [450, 1000]
}

# Grid Search
grid_search_cat = GridSearchCV(
    estimator=cat,
    param_grid=params_catB,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
).fit(X_train.values, y_train.values.ravel())

# Best model
cat_best = grid_search_cat.best_estimator_
print(f'CatBoost Best Score: {grid_search_cat.best_score_:.4f}')
print(f'CatBoost Best Params: {grid_search_cat.best_params_}')
print(f'CatBoost CV Accuracy: {cross_val_score(cat_best, X_train.values, y_train.values.ravel(), cv=3).mean():.4f}')
```

### 5. Prediction & Submission
```python
# Predict on test set
predictions = cat_best.predict(test_data.values)

# Generate submission file
sub = pd.DataFrame()
sub['PassengerId'] = test_IDs
sub['Survived'] = predictions
sub.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")
```

## Results
- **Cross-Validation Accuracy**: 99.63% (CatBoost)
- **Key Hyperparameters**: `depth=6`, `iterations=450`, `learning_rate=0.01`
- **Submission**: Ready-to-upload `submission.csv` with test set predictions

## Notes & Improvements
1. **Overfitting Mitigation**: The high training accuracy may indicate overfitting. Consider:
   - Adding regularization (`l2_leaf_reg` in CatBoost)
   - Reducing model complexity (e.g., `depth=5`, `iterations=300`)
   - Using simpler models (Random Forest, XGBoost) for comparison
   - Adding a holdout validation set (e.g., 80-20 train-validation split)

2. **Feature Engineering**:
   - Explore fare bins (e.g., low/medium/high)
   - Extract embarkation port + destination combinations
   - Analyze `Title` + `Pclass` interactions

3. **Alternative Models**:
   - Compare with XGBoost, LightGBM, or ensemble methods (stacking)
   - Use Bayesian optimization (Optuna) for hyperparameter tuning
