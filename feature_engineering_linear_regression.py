"""
Feature Engineering and Linear Regression for Classification
==============================================================

This script demonstrates:
1. Reading and preprocessing a salary dataset
2. Feature engineering techniques
3. Training a linear regression model
4. Converting regression to classification
5. Model evaluation with accuracy and confusion matrix

Author: ML Course PEECT413
Dataset: Salary Dataset (Years of Experience vs Salary)
"""

# ============================================================================
# STEP 1: Import Required Libraries
# ============================================================================

# Data manipulation and numerical operations
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LinearRegression      # Linear regression model
from sklearn.preprocessing import StandardScaler       # For feature scaling
from sklearn.metrics import (
    confusion_matrix,       # For confusion matrix
    accuracy_score,         # For calculating accuracy
    classification_report,  # For detailed classification metrics
    mean_squared_error,     # For regression error metrics
    r2_score               # For R² score
)

# For handling warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("Feature Engineering and Linear Regression Classification")
print("="*70)
print()


# ============================================================================
# STEP 2: Load the Dataset
# ============================================================================

print("STEP 2: Loading Dataset")
print("-" * 70)

# Define the dataset path
# The script will first try the Kaggle path, then fall back to local path
dataset_paths = [
    '/kaggle/input/salary-dataset-simple-linear-regression/Salary_dataset.csv',
    'Salary_dataset.csv'
]

# Try to load the dataset from available paths
df = None
for path in dataset_paths:
    try:
        df = pd.read_csv(path)
        print(f"✓ Dataset loaded successfully from: {path}")
        break
    except FileNotFoundError:
        continue

# If dataset not found in any path, exit
if df is None:
    raise FileNotFoundError(
        "Dataset not found. Please ensure Salary_dataset.csv is available."
    )

# Display basic information about the dataset
print(f"\nDataset Shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print()


# ============================================================================
# STEP 3: Data Preprocessing
# ============================================================================

print("STEP 3: Data Preprocessing")
print("-" * 70)

# Check for missing values
print("Checking for missing values:")
print(df.isnull().sum())

# Handle missing values if any (forward fill method)
if df.isnull().sum().sum() > 0:
    print("\n⚠ Missing values detected. Filling with forward fill method...")
    df = df.ffill()
    print("✓ Missing values handled")
else:
    print("✓ No missing values found")

# Check for duplicates
duplicate_count = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicate_count}")
if duplicate_count > 0:
    print("⚠ Removing duplicate rows...")
    df = df.drop_duplicates()
    print("✓ Duplicates removed")

print(f"\nFinal dataset shape: {df.shape}")
print()


# ============================================================================
# STEP 4: Feature Engineering
# ============================================================================

print("STEP 4: Feature Engineering")
print("-" * 70)

# Extract features (X) and target (y)
# YearsExperience is our independent variable
# Salary is our dependent variable
X = df[['YearsExperience']].values
y = df['Salary'].values

print(f"Original features shape: {X.shape}")
print(f"Original target shape: {y.shape}")

# Feature Engineering Technique 1: Polynomial Features
# Creating squared and cubic features from YearsExperience
print("\n1. Creating Polynomial Features:")
print("   - YearsExperience² (squared term)")
print("   - YearsExperience³ (cubic term)")

# Create new features
years_squared = (X ** 2).flatten()
years_cubed = (X ** 3).flatten()

# Feature Engineering Technique 2: Logarithmic Transformation
# Useful for capturing non-linear relationships
print("\n2. Creating Logarithmic Feature:")
print("   - log(YearsExperience + 1)")

# Add 1 to avoid log(0) which is undefined
years_log = np.log(X + 1).flatten()

# Feature Engineering Technique 3: Exponential Feature
print("\n3. Creating Exponential Feature:")
print("   - exp(YearsExperience / 10)")

# Divide by 10 to keep values reasonable
years_exp = np.exp(X / 10).flatten()

# Feature Engineering Technique 4: Interaction Features
# For demonstration, we'll create interaction with constant
print("\n4. Creating Interaction Features:")
print("   - YearsExperience × scaled_constant")

# Create a scaled constant feature
scaled_constant = np.full(X.shape[0], 2.0)
interaction = (X.flatten() * scaled_constant)

# Feature Engineering Technique 5: Statistical Features
# Create rolling mean-like feature (using position-based averaging)
print("\n5. Creating Statistical Features:")
print("   - Experience category (binned)")

# Bin the experience into categories
experience_bins = pd.cut(X.flatten(), bins=3, labels=['Junior', 'Mid', 'Senior'])
# Convert categories to numerical values
experience_numeric = experience_bins.codes

# Combine all engineered features
print("\n6. Combining All Features:")
X_engineered = np.column_stack([
    X.flatten(),           # Original feature
    years_squared,         # Squared term
    years_cubed,          # Cubic term
    years_log,            # Logarithmic term
    years_exp,            # Exponential term
    interaction,          # Interaction term
    experience_numeric    # Categorical feature
])

print(f"   Original features: 1")
print(f"   Engineered features: 7")
print(f"   Engineered feature matrix shape: {X_engineered.shape}")

# Create a DataFrame for better visualization
feature_names = [
    'YearsExperience',
    'YearsExp_Squared',
    'YearsExp_Cubed',
    'YearsExp_Log',
    'YearsExp_Exp',
    'Interaction',
    'Experience_Category'
]

X_engineered_df = pd.DataFrame(X_engineered, columns=feature_names)
print("\nFirst 5 rows of engineered features:")
print(X_engineered_df.head())
print()


# ============================================================================
# STEP 5: Creating Classification Target
# ============================================================================

print("STEP 5: Creating Classification Target from Salary")
print("-" * 70)

# For classification, we need to convert continuous salary to binary classes
# We'll use the median salary as threshold
median_salary = np.median(y)
print(f"Median Salary: ${median_salary:,.2f}")

# Create binary classification target
# 1 (High Salary) if salary >= median, 0 (Low Salary) otherwise
y_classification = (y >= median_salary).astype(int)

print(f"\nClass Distribution:")
unique, counts = np.unique(y_classification, return_counts=True)
for class_label, count in zip(unique, counts):
    class_name = "High Salary" if class_label == 1 else "Low Salary"
    percentage = (count / len(y_classification)) * 100
    print(f"   Class {class_label} ({class_name}): {count} samples ({percentage:.1f}%)")
print()


# ============================================================================
# STEP 6: Feature Scaling
# ============================================================================

print("STEP 6: Feature Scaling")
print("-" * 70)

# Initialize the StandardScaler
# Standardization: (X - mean) / std_deviation
# This ensures all features have mean=0 and std=1
scaler = StandardScaler()

# Fit the scaler on the engineered features and transform
X_scaled = scaler.fit_transform(X_engineered)

print("✓ Features standardized using StandardScaler")
print(f"   Mean of scaled features: {X_scaled.mean(axis=0).round(4)}")
print(f"   Std of scaled features: {X_scaled.std(axis=0).round(4)}")
print()


# ============================================================================
# STEP 7: Train-Test Split
# ============================================================================

print("STEP 7: Splitting Data into Training and Testing Sets")
print("-" * 70)

# Split the data into training (80%) and testing (20%) sets
# stratify ensures both sets have similar class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_classification,
    test_size=0.2,
    random_state=42,
    stratify=y_classification
)

print(f"Training set size: {X_train.shape[0]} samples ({(X_train.shape[0]/len(X_scaled))*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} samples ({(X_test.shape[0]/len(X_scaled))*100:.1f}%)")

print(f"\nTraining set class distribution:")
unique_train, counts_train = np.unique(y_train, return_counts=True)
for class_label, count in zip(unique_train, counts_train):
    class_name = "High Salary" if class_label == 1 else "Low Salary"
    print(f"   Class {class_label} ({class_name}): {count} samples")

print(f"\nTesting set class distribution:")
unique_test, counts_test = np.unique(y_test, return_counts=True)
for class_label, count in zip(unique_test, counts_test):
    class_name = "High Salary" if class_label == 1 else "Low Salary"
    print(f"   Class {class_label} ({class_name}): {count} samples")
print()


# ============================================================================
# STEP 8: Model Training - Linear Regression
# ============================================================================

print("STEP 8: Training Linear Regression Model")
print("-" * 70)

# Initialize the Linear Regression model
# Linear Regression equation: y = β0 + β1*x1 + β2*x2 + ... + βn*xn
model = LinearRegression()

# Train the model on training data
print("Training the model...")
model.fit(X_train, y_train)
print("✓ Model trained successfully")

# Display model parameters
print(f"\nModel Parameters:")
print(f"   Intercept (β0): {model.intercept_:.4f}")
print(f"   Number of coefficients: {len(model.coef_)}")
print(f"\nFeature Coefficients:")
for feature_name, coef in zip(feature_names, model.coef_):
    print(f"   {feature_name}: {coef:.4f}")
print()


# ============================================================================
# STEP 9: Making Predictions
# ============================================================================

print("STEP 9: Making Predictions")
print("-" * 70)

# Predict on training data (continuous values)
y_train_pred_continuous = model.predict(X_train)
print(f"Training predictions (continuous) - Shape: {y_train_pred_continuous.shape}")
print(f"   Min: {y_train_pred_continuous.min():.4f}, Max: {y_train_pred_continuous.max():.4f}")

# Predict on testing data (continuous values)
y_test_pred_continuous = model.predict(X_test)
print(f"Testing predictions (continuous) - Shape: {y_test_pred_continuous.shape}")
print(f"   Min: {y_test_pred_continuous.min():.4f}, Max: {y_test_pred_continuous.max():.4f}")

# Convert continuous predictions to binary classification
# Using 0.5 as threshold (since our target is 0 or 1)
print(f"\nConverting continuous predictions to binary classification:")
print(f"   Threshold: 0.5 (values >= 0.5 → Class 1, values < 0.5 → Class 0)")

y_train_pred = (y_train_pred_continuous >= 0.5).astype(int)
y_test_pred = (y_test_pred_continuous >= 0.5).astype(int)

print(f"\n✓ Predictions converted to binary classes")
print(f"   Training predictions: {np.unique(y_train_pred, return_counts=True)}")
print(f"   Testing predictions: {np.unique(y_test_pred, return_counts=True)}")
print()


# ============================================================================
# STEP 10: Model Evaluation - Regression Metrics
# ============================================================================

print("STEP 10: Regression Metrics Evaluation")
print("-" * 70)

# Calculate regression metrics
train_mse = mean_squared_error(y_train, y_train_pred_continuous)
test_mse = mean_squared_error(y_test, y_test_pred_continuous)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(y_train, y_train_pred_continuous)
test_r2 = r2_score(y_test, y_test_pred_continuous)

print("Training Set Metrics:")
print(f"   Mean Squared Error (MSE): {train_mse:.4f}")
print(f"   Root Mean Squared Error (RMSE): {train_rmse:.4f}")
print(f"   R² Score: {train_r2:.4f}")

print("\nTesting Set Metrics:")
print(f"   Mean Squared Error (MSE): {test_mse:.4f}")
print(f"   Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"   R² Score: {test_r2:.4f}")
print()


# ============================================================================
# STEP 11: Model Evaluation - Classification Metrics
# ============================================================================

print("STEP 11: Classification Metrics Evaluation")
print("-" * 70)

# Calculate accuracy for training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

# Calculate accuracy for testing set
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Check for overfitting/underfitting
accuracy_diff = train_accuracy - test_accuracy
print(f"\nAccuracy Difference (Train - Test): {accuracy_diff:.4f}")
if abs(accuracy_diff) < 0.05:
    print("✓ Model shows good generalization (low overfitting)")
elif accuracy_diff > 0.05:
    print("⚠ Model might be overfitting (train accuracy >> test accuracy)")
else:
    print("⚠ Model might be underfitting (test accuracy >> train accuracy)")
print()


# ============================================================================
# STEP 12: Confusion Matrix
# ============================================================================

print("STEP 12: Confusion Matrix")
print("-" * 70)

# Compute confusion matrix for testing set
cm = confusion_matrix(y_test, y_test_pred)

print("Confusion Matrix (Testing Set):")
print(cm)
print()

# Extract confusion matrix components
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix Components:")
print(f"   True Negatives (TN): {tn}")
print(f"   False Positives (FP): {fp}")
print(f"   False Negatives (FN): {fn}")
print(f"   True Positives (TP): {tp}")
print()

# Calculate additional metrics from confusion matrix
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Derived Metrics from Confusion Matrix:")
print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall (Sensitivity): {recall:.4f} ({recall*100:.2f}%)")
print(f"   Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
print(f"   F1-Score: {f1_score:.4f}")
print()


# ============================================================================
# STEP 13: Classification Report
# ============================================================================

print("STEP 13: Detailed Classification Report")
print("-" * 70)

# Generate detailed classification report
print("\nClassification Report (Testing Set):")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=['Low Salary (0)', 'High Salary (1)'],
    digits=4
))


# ============================================================================
# STEP 14: Visualization
# ============================================================================

print("STEP 14: Creating Visualizations")
print("-" * 70)

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feature Engineering & Linear Regression Classification Analysis', 
             fontsize=16, fontweight='bold')

# Subplot 1: Confusion Matrix Heatmap
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax1,
            xticklabels=['Low Salary', 'High Salary'],
            yticklabels=['Low Salary', 'High Salary'])
ax1.set_xlabel('Predicted Label', fontweight='bold')
ax1.set_ylabel('True Label', fontweight='bold')
ax1.set_title(f'Confusion Matrix (Test Accuracy: {test_accuracy*100:.2f}%)', 
              fontweight='bold')

# Subplot 2: Accuracy Comparison
ax2 = axes[0, 1]
accuracies = [train_accuracy * 100, test_accuracy * 100]
labels = ['Training', 'Testing']
colors = ['#4CAF50', '#2196F3']
bars = ax2.bar(labels, accuracies, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Accuracy (%)', fontweight='bold')
ax2.set_title('Model Accuracy Comparison', fontweight='bold')
ax2.set_ylim([0, 105])
ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

# Subplot 3: Class Distribution
ax3 = axes[1, 0]
class_counts = [counts_test[0], counts_test[1]]
class_labels = ['Low Salary\n(Class 0)', 'High Salary\n(Class 1)']
wedges, texts, autotexts = ax3.pie(class_counts, labels=class_labels, autopct='%1.1f%%',
                                     colors=['#FF6B6B', '#4ECDC4'], startangle=90,
                                     textprops={'fontweight': 'bold'})
ax3.set_title('Test Set Class Distribution', fontweight='bold')

# Subplot 4: Model Performance Metrics
ax4 = axes[1, 1]
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
values = [precision * 100, recall * 100, f1_score * 100, test_accuracy * 100]
colors_metrics = ['#FF9800', '#9C27B0', '#00BCD4', '#4CAF50']
bars = ax4.barh(metrics, values, color=colors_metrics, alpha=0.7, edgecolor='black')
ax4.set_xlabel('Score (%)', fontweight='bold')
ax4.set_title('Model Performance Metrics (Testing Set)', fontweight='bold')
ax4.set_xlim([0, 105])
# Add value labels on bars
for bar, val in zip(bars, values):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
            f'{val:.2f}%', ha='left', va='center', fontweight='bold', 
            fontsize=10, color='black')

plt.tight_layout()

# Save the figure
output_filename = 'feature_engineering_results.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved as '{output_filename}'")

# Display the plot
plt.show()
print()


# ============================================================================
# STEP 15: Summary and Insights
# ============================================================================

print("=" * 70)
print("SUMMARY AND INSIGHTS")
print("=" * 70)

print("\n📊 Dataset Information:")
print(f"   • Total samples: {len(df)}")
print(f"   • Features: {df.shape[1]} original → 7 engineered features")
print(f"   • Training samples: {len(X_train)}")
print(f"   • Testing samples: {len(X_test)}")

print("\n🔧 Feature Engineering Applied:")
print(f"   • Polynomial features (squared, cubic)")
print(f"   • Logarithmic transformation")
print(f"   • Exponential transformation")
print(f"   • Interaction features")
print(f"   • Statistical binning (experience categories)")

print("\n📈 Model Performance:")
print(f"   • Training Accuracy: {train_accuracy*100:.2f}%")
print(f"   • Testing Accuracy: {test_accuracy*100:.2f}%")
print(f"   • Precision: {precision*100:.2f}%")
print(f"   • Recall: {recall*100:.2f}%")
print(f"   • F1-Score: {f1_score*100:.2f}%")

print("\n🎯 Confusion Matrix Summary:")
print(f"   • Correctly classified: {tp + tn} out of {len(y_test)} samples")
print(f"   • Misclassified: {fp + fn} out of {len(y_test)} samples")
print(f"   • True Positives: {tp}")
print(f"   • True Negatives: {tn}")
print(f"   • False Positives: {fp}")
print(f"   • False Negatives: {fn}")

print("\n💡 Key Insights:")
if test_accuracy >= 0.8:
    print(f"   • The model shows excellent classification performance (>80% accuracy)")
elif test_accuracy >= 0.7:
    print(f"   • The model shows good classification performance (70-80% accuracy)")
else:
    print(f"   • The model shows moderate performance. Consider more feature engineering")

if abs(accuracy_diff) < 0.05:
    print(f"   • The model generalizes well (low overfitting)")
elif accuracy_diff > 0.05:
    print(f"   • Consider regularization to reduce overfitting")

if precision > 0.8 and recall > 0.8:
    print(f"   • Balanced performance in identifying both classes")
elif precision > recall:
    print(f"   • Model is more conservative (fewer false positives)")
else:
    print(f"   • Model is more aggressive (fewer false negatives)")

print("\n" + "=" * 70)
print("✓ Feature Engineering and Classification Analysis Complete!")
print("=" * 70)
