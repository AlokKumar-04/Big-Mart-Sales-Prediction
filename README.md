
# Big Mart Sales Prediction

This project aims to predict the sales of products in various Big Mart outlets using machine learning techniques. The dataset includes information about the products and the stores, and the task is to predict the sales for the test dataset.

## Project Structure

1. **Data Loading and Exploration**
   - Loading the training and test datasets.
   - Initial exploration to understand the data distribution and identify missing values.

2. **Data Preprocessing**
   - Handling missing values and inconsistencies in the data.
   - Feature engineering, including encoding categorical variables and scaling numerical features.

3. **Exploratory Data Analysis (EDA)**
   - Visualizing the relationships between different features and the target variable (`Item_Outlet_Sales`).
   - Identifying significant features and correlations.

4. **Model Building and Evaluation**
   - Training a Random Forest model and other regression models.
   - Evaluating model performance using metrics such as RMSE and R-squared.

5. **Prediction**
   - Generating predictions for the test dataset using the trained model.
   - Exporting the predictions for submission.

## Code Overview

```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Loading the datasets
df_train = pd.read_csv('T_train.csv')
df_test = pd.read_csv('T_test.csv')

# Data exploration and visualization
df_train.describe()
df_train.info()

# Data preprocessing
df_train['Outlet_Size'].fillna(df_train['Outlet_Size'].mode()[0], inplace=True)
# ... (additional preprocessing steps)

# Feature engineering
le = LabelEncoder()
df_train['Outlet_Type'] = le.fit_transform(df_train['Outlet_Type'])
# ... (additional feature engineering steps)

# Model training
X = df_train.drop(columns=['Item_Outlet_Sales'])
y = df_train['Item_Outlet_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Prediction on test data
test_pred_rf = model.predict(df_test)
```

## Requirements

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

Run the Jupyter notebook or the Python scripts provided to reproduce the analysis and predictions.

## Contributors

- [Your Name](https://github.com/yourusername)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
