# Feature Engineering and Linear Regression Classification

## Overview
This Python script demonstrates feature engineering techniques and applies linear regression for binary classification on a salary dataset. The script performs comprehensive data analysis, creates multiple engineered features, trains a model, and evaluates its performance with detailed metrics.

## Features

### 1. Data Processing
- Loads salary dataset (YearsExperience vs Salary)
- Handles missing values and duplicates
- Performs exploratory data analysis

### 2. Feature Engineering
The script creates 7 engineered features from the original YearsExperience feature:
- **Polynomial Features**: Squared and cubic terms
- **Logarithmic Transformation**: Natural log transformation
- **Exponential Feature**: Exponential scaling
- **Interaction Features**: Feature interactions
- **Statistical Binning**: Experience categories (Junior/Mid/Senior)

### 3. Classification Task
- Converts continuous salary values to binary classification (High/Low salary)
- Uses median salary as threshold
- Ensures balanced class distribution

### 4. Model Training
- Applies StandardScaler for feature normalization
- Trains Linear Regression model
- Uses 80-20 train-test split with stratification

### 5. Comprehensive Evaluation
- **Regression Metrics**: MSE, RMSE, R² Score
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Detailed breakdown with TP, TN, FP, FN
- **Visualizations**: Heatmaps, bar charts, pie charts

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage

### Running the Script
```bash
python feature_engineering_linear_regression.py
```

### Dataset Path
The script looks for the dataset in two locations (in order):
1. `/kaggle/input/salary-dataset-simple-linear-regression/Salary_dataset.csv` (Kaggle environment)
2. `Salary_dataset.csv` (Current directory)

### Expected Output
The script will:
1. Print detailed step-by-step analysis to console
2. Generate `feature_engineering_results.png` with visualizations
3. Display accuracy, confusion matrix, and classification report

## Output Description

### Console Output
- Dataset statistics and information
- Feature engineering transformations
- Model training details with coefficients
- Comprehensive evaluation metrics
- Summary with key insights

### Visualization File
The generated `feature_engineering_results.png` contains four subplots:
1. **Confusion Matrix Heatmap**: Visual representation of classification results
2. **Accuracy Comparison**: Training vs Testing accuracy
3. **Class Distribution**: Pie chart showing test set balance
4. **Performance Metrics**: Bar chart with Precision, Recall, F1-Score, Accuracy

## Sample Results

```
Testing Accuracy: 100.00%
Precision: 100.00%
Recall: 100.00%
F1-Score: 100.00%

Confusion Matrix:
[[3 0]
 [0 3]]
```

## Code Structure

The script follows a clear 15-step workflow:
1. Import libraries
2. Load dataset
3. Data preprocessing
4. Feature engineering
5. Create classification target
6. Feature scaling
7. Train-test split
8. Model training
9. Make predictions
10. Regression evaluation
11. Classification evaluation
12. Confusion matrix analysis
13. Classification report
14. Visualization
15. Summary and insights

## Key Concepts Demonstrated

### Feature Engineering Techniques
- **Polynomial Features**: Capture non-linear relationships
- **Logarithmic Transformation**: Handle skewed distributions
- **Feature Scaling**: Standardization for better model performance
- **Binning**: Convert continuous to categorical features

### Regression to Classification
- Uses linear regression with binary threshold
- Threshold = 0.5 (based on 0/1 target encoding)
- Demonstrates flexibility of regression for classification tasks

### Model Evaluation
- Multiple metric perspectives (regression + classification)
- Train-test comparison to detect overfitting
- Confusion matrix decomposition
- Visual performance analysis

## Comments and Documentation
The entire script is extensively commented with:
- Section headers for each major step
- Inline explanations of key operations
- Purpose and interpretation of each metric
- Formula references where applicable

## Customization

### Modifying the Threshold
To change the classification threshold from 0.5:
```python
y_train_pred = (y_train_pred_continuous >= YOUR_THRESHOLD).astype(int)
y_test_pred = (y_test_pred_continuous >= YOUR_THRESHOLD).astype(int)
```

### Adding More Features
Add custom feature engineering in Step 4:
```python
custom_feature = X ** 4  # Example: quartic term
X_engineered = np.column_stack([X_engineered, custom_feature])
```

### Changing Train-Test Split
Modify the test_size parameter in Step 7:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_classification, 
    test_size=0.3,  # 30% test, 70% train
    random_state=42
)
```

## Dataset Format

The expected CSV format:
```
YearsExperience,Salary
1.1,39343
1.3,46205
1.5,37731
...
```

- **Column 1**: YearsExperience (float) - Years of professional experience
- **Column 2**: Salary (int) - Annual salary in currency units

## Troubleshooting

### Module Not Found Error
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### File Not Found Error
Ensure `Salary_dataset.csv` is in the same directory as the script, or update the path in the script.

### Visualization Not Displayed
If running in a headless environment, the visualization is saved as PNG but may not display. Check for `feature_engineering_results.png` in the working directory.

## License
This script is created for educational purposes as part of the PEECT413 Machine Learning course.

## Author
PEECT413 Machine Learning Course

## Version
1.0 - Initial Release
