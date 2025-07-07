````markdown
# Enhanced Machine Learning Model Evaluation for Raft Foundation Analysis

---

## Project Description

This project provides a comprehensive framework for evaluating various machine learning (ML) regression models to predict key parameters for raft foundations, specifically **settlement**, **punching shear**, and **bearing pressure**. It incorporates robust data preprocessing, feature engineering, and a systematic comparison of multiple ML algorithms under different data scaling techniques. The goal is to identify the most accurate and reliable ML model for analyzing raft foundation behavior, offering insights into critical design parameters.

---

## Features

* **Multi-Model Evaluation:** Systematically evaluates 13 different regression models, including Linear Models, Tree-based models (Random Forest, XGBoost, LightGBM, CatBoost), Support Vector Regressor, K-Nearest Neighbors, and Neural Networks.
* **Multi-Output Regression:** Designed to handle multiple target variables simultaneously, which is crucial for raft foundation analysis where several inter-related outputs are desired.
* **Multiple Scaling Methods:** Compares the performance of models using three different data scaling techniques: StandardScaler, MinMaxScaler, and RobustScaler, to determine the optimal preprocessing approach.
* **Comprehensive Metrics:** Reports a wide range of evaluation metrics for each target variable, including R-squared ($R^2$), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE), along with cross-validation $R^2$ and training $R^2$ to assess generalization and overfitting.
* **Feature Engineering & Selection:** Includes steps for creating new features and selecting the most relevant ones using `SelectKBest` to improve model performance and interpretability.
* **Best Model Identification:** Automatically identifies and highlights the best-performing model and scaler combination based on average $R^2$ score across all target variables.
* **Feature Importance Analysis:** Provides insights into the most influential input features for tree-based models and through permutation importance for all models.
* **Hyperparameter Tuning:** Implements `GridSearchCV` for hyperparameter optimization of the best-performing model to further enhance its predictive accuracy.
* **Visualizations:** Generates various plots for data exploration, correlation analysis, actual vs. predicted values, residual plots, and model accuracy comparisons.
* **Model Persistence:** Allows for saving the best-performing model and scaler for future use in prediction.
* **Custom Prediction Function:** Includes a user-friendly function to make predictions on new, user-provided input data.

---

## Installation Instructions

### Prerequisites

* Python 3.x
* Jupyter Notebook or Google Colab (recommended for ease of use)

### Environment Setup

It is recommended to create a virtual environment to manage dependencies:

```bash
python -m venv raft_env
source raft_env/bin/activate  # On Windows, use `raft_env\Scripts\activate`
````

### Install Dependencies

All necessary Python libraries can be installed using `pip`:

```bash
pip install openpyxl scikit-learn xgboost lightgbm catboost matplotlib seaborn pandas numpy
```

-----

## Usage

### Step 0: Upload the Excel file

The code is designed to run in a Google Colab environment. First, you'll need to upload your dataset, `Expanded Analysis of Rafts (1).xlsx`, to the Colab environment.

```python
from google.colab import files
uploaded = files.upload()
```

### Running the Analysis

Once the dependencies are installed and the Excel file is uploaded, you can run the entire script. The script will perform the following actions:

1.  **Load Data:** Reads the `Expanded Analysis of Rafts (1).xlsx` file.
2.  **Data Exploration and Preprocessing:** Prints data overview, missing values, data types, statistical summary, and outlier detection. Visualizes feature correlation.
3.  **Feature Engineering and Selection:** Creates new features and selects the top 10 features.
4.  **Model Training and Evaluation:** Iterates through different scalers and ML models, performing cross-validation and evaluating performance on test data for each target.
5.  **Identify Best Model:** Determines the best scaler and model combination based on average $R^2$.
6.  **Detailed Analysis:** Prints detailed metrics for the best model.
7.  **Feature Importance:** Displays feature importance (if applicable) and permutation importance plots.
8.  **Visualizations:** Generates scatter plots of actual vs. predicted values for the best model.
9.  **Model Comparison Summary:** Presents a table summarizing the performance of top models across different targets.
10. **Hyperparameter Tuning:** Tunes the hyperparameters of the best model (if defined).
11. **Save Model:** Saves the best model and scaler as `.pkl` files.
12. **Custom Prediction (Optional):** You can uncomment the `make_prediction` function call at the end of the script to interactively input values and get predictions.
13. **Comprehensive Visualizations:** Provides additional plots for model accuracies and detailed prediction analysis (actual vs. predicted, residuals, and residual distribution) specifically for models evaluated with `StandardScaler` and achieving an average R-squared of at least 0.5.

Simply run all cells in your Jupyter Notebook or Google Colab environment.

```python
# All the provided code from the prompt should be placed here in a single block
# For example:
import pandas as pd
# ... (rest of the code)
```

### Making Custom Predictions

After the script has run and saved the `best_model.pkl` and `best_scaler.pkl` files, you can load them and use the `make_prediction` function:

```python
import pickle
import pandas as pd

# Load the saved model and scaler
with open('best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('best_scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# Define the feature names as used during training
# Ensure this list matches the selected features from Step 4
feature_names_for_prediction = ['Number of Columns', 'Area Of Raft (m^2)', 'Column Area (m^2)',
                               "Compressive strength of Concrete Fc' (Mpa)", 'Concrete Unit Weight (kN/m^3)',
                               'Subgrade Modulus kN/m/m^2', 'Maximum Axial Load on Column in kN',
                               'Total Axial load on Column (kN)', 'Thickness of Raft (mm)',
                               'Load_per_Column', 'Raft_Load_Ratio', 'Column_Density', 'Strength_to_Load_Ratio']

# Remove features that were not selected by SelectKBest in Step 4.
# For example, if X.columns after selection were:
# ['Number of Columns', 'Area Of Raft (m^2)', 'Compressive strength of Concrete Fc\' (Mpa)',
#  'Subgrade Modulus kN/m/m^2', 'Maximum Axial Load on Column in kN',
#  'Total Axial load on Column (kN)', 'Thickness of Raft (mm)',
#  'Load_per_Column', 'Raft_Load_Ratio', 'Column_Density']
# Then feature_names_for_prediction should only include these 10.
# The `X.columns[selector.get_support()]` line in the original code determines this.
# You need to manually update `feature_names_for_prediction` after running the initial script to get the exact list.
# For example, if the output was `Selected features: ['F1', 'F2', 'F3', ...]`, use that list here.
# For this specific run, the selected features are obtained from the `print(f"Selected features: {list(selected_features)}")` output.

# To be precise, you should dynamically get the selected features from the script's output
# or re-run the feature selection part to get `selected_features`.
# Assuming 'selected_features' from the script is:
# ['Number of Columns', 'Area Of Raft (m^2)', 'Column Area (m^2)', "Compressive strength of Concrete Fc' (Mpa)",
#  'Concrete Unit Weight (kN/m^3)', 'Subgrade Modulus kN/m/m^2', 'Maximum Axial Load on Column in kN',
#  'Total Axial load on Column (kN)', 'Thickness of Raft (mm)', 'Column_Density']
# You would use that list.

# For the given code, the X used for training is X = X[selected_features]
# So, the feature names for prediction should be `selected_features` from the script's output.
# Let's assume based on the `selector = SelectKBest(score_func=f_regression, k=10)`
# and the initial 9 features plus 4 engineered features, that the k=10 selected features
# are a subset. You need to obtain the actual `selected_features` list from the script's output.
# For the purpose of this example, let's use a placeholder if the exact list isn't known without running.
# The code itself prints `selected_features = X.columns[selector.get_support()]`
# So, you would take that printed list.

# Example placeholder for feature_names_for_prediction (replace with actual selected features)
# Based on the code, `X` is updated to contain only `selected_features`.
# So, when `make_prediction` is called, `X.columns` will already contain the selected features.
# Therefore, you just need to pass `X.columns` from the final state of X in the script.

# Uncomment the line below in the actual script to enable interactive prediction:
# make_prediction(best_model, best_scaler, X.columns)
```

-----

## Code Structure

The project is structured within a single Python script (`.py` file or Jupyter Notebook cells) for sequential execution.

  * **Step 0: Upload Excel file and install dependencies:** Handles initial setup for Google Colab and installs required libraries.
  * **Step 1: Import Libraries:** Imports all necessary modules from `pandas`, `numpy`, `sklearn`, `xgboost`, `lightgbm`, `catboost`, `matplotlib`, and `seaborn`.
  * **Step 2: Load Dataset:** Reads the Excel data into a pandas DataFrame and cleans column names.
  * **Step 3: Data Exploration and Preprocessing:** Performs initial data checks, statistical summaries, outlier detection, and correlation analysis.
  * **Step 4: Feature Engineering and Selection:** Creates new informative features and applies `SelectKBest` for dimensionality reduction.
  * **Step 5: Train-Test Split with different scaling methods:** Divides the dataset into training and testing sets and defines various data scalers.
  * **Step 6: Comprehensive Model Definition:** Initializes a dictionary of diverse regression models.
  * **Step 7: Model Evaluation with Cross-Validation:** Iterates through each scaler and model, trains them using `MultiOutputRegressor`, evaluates performance with cross-validation and test metrics, and stores results.
  * **Step 8: Find Best Model and Scaler Combination:** Identifies the top-performing model and scaler based on average $R^2$.
  * **Step 9: Detailed Analysis of Best Model:** Prints comprehensive metrics for the optimal model.
  * **Step 10: Feature Importance Analysis:** Visualizes and lists feature importances.
  * **Step 11: Visualization of Best Model Performance:** Plots actual vs. predicted values for the best model.
  * **Step 12: Model Comparison Summary:** Presents a tabular summary of all evaluated models, filtered for acceptable $R^2$ scores.
  * **Step 13: Hyperparameter Tuning for Best Model:** Performs `GridSearchCV` for the best model (if defined in `param_grid`).
  * **Step 14: Custom Prediction Function:** A utility function for making predictions with user input.
  * **Step 15: Save Best Model (optional):** Serializes and saves the best model and scaler using `pickle`.
  * **Step 16: Visualize Model Accuracies and Predictions:** Generates detailed plots for model accuracy and prediction analysis, focusing on models evaluated with `StandardScaler` that meet performance criteria.

-----

## Contributing Guidelines

Contributions are welcome\! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

-----

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

```
MIT License

Copyright (c) 2024 Muhammad Asad Sardar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
