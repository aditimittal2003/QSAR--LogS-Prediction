# 1. Define the objective - Predicting LogS values

# The objective is to build a regression model to predict the log solubility of molecules.
# The target variable is 'LogS' (log solubility in mols per litre).
# This is a regression task as we are predicting a continuous numerical value.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
import os
import requests
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# --- Setup ---
os.makedirs('saved_model', exist_ok=True)
sns.set_style("whitegrid")

# --- Function to Download Data ---
def download_data(url, filename):
    """Downloads the dataset if it doesn't already exist."""
    if not os.path.exists(filename):
        print(f"Dataset not found. Downloading from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Dataset saved successfully as '{filename}'.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the dataset: {e}")
            exit()
    else:
        print(f"Dataset '{filename}' already exists.")

# --- Function to calculate Applicability Domain (Range-Based Method) ---
def calculate_ad(X_train, X_data):
    """Calculates a simple binary Applicability Domain (AD) based on descriptor range."""
    ad_results = pd.DataFrame(index=X_data.index)
    ad_cols = []

    # Calculate min/max range for each feature in the training set
    for col in X_train.columns:
        min_val = X_train[col].min()
        max_val = X_train[col].max()
        # Check if data point is outside the training range for this feature
        ad_results[col] = (X_data[col] < min_val) | (X_data[col] > max_val)
        ad_cols.append(col)

    # A molecule is considered OUTSIDE the AD if it is outside the range for ANY descriptor
    ad_results['Outside_AD'] = ad_results[ad_cols].any(axis=1)

    return ad_results['Outside_AD']



# 2. Collect and prepare data
# Define data source and call the download function
DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
DATA_FILE = "delaney-processed.csv"
download_data(DATA_URL, DATA_FILE)

# Load the data from the local file
df_final = pd.read_csv(DATA_FILE)

# Rename columns for clarity
df_final = df_final.rename(columns={'measured log solubility in mols per litre': 'LogS', 'smiles': 'smiles'})

# Data Cleaning (Handling invalid SMILES - molecules that cannot be parsed by RDKit)
mols = [Chem.MolFromSmiles(smi) for smi in df_final['smiles']]
valid_mols_idx = [i for i, m in enumerate(mols) if m is not None]
mols = [mols[i] for i in valid_mols_idx]
df_final = df_final.iloc[valid_mols_idx].reset_index(drop=True) # Reset index after dropping rows

# 3. Represent Molecules
print("Calculating RDKit features...")
desc_names = [desc[0] for desc in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
descriptors = [calculator.CalcDescriptors(mol) for mol in mols]
desc_df = pd.DataFrame(descriptors, columns=desc_names, index=df_final.index)

# Data Cleaning (Remove inf/NaN in descriptors)
desc_df.replace([np.inf, -np.inf], np.nan, inplace=True)
desc_df.dropna(axis=1, how='any', inplace=True)

X = desc_df
y = df_final['LogS']

# Feature Selection (Variance and Correlation)
print("Selecting features based on variance and correlation...")
vt = VarianceThreshold(threshold=0.05)
X_var = vt.fit_transform(X)
X_final_features = pd.DataFrame(X_var, columns=X.columns[vt.get_support()], index=X.index) # Keep index

corr_matrix = X_final_features.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_final_features = X_final_features.drop(to_drop, axis=1)

print(f"Total descriptors kept after selection: {X_final_features.shape[1]}")


# 4. Split the Data
# split the data section:
from sklearn.model_selection import KFold

print("Defining the 5-Fold cross-validation strategy...")
# Instead of a single split, we define the KFold object that will create 5 different splits.
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)


# 5. Build the model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10),
    "LightGBM": lgb.LGBMRegressor(random_state=42, verbose=-1, n_estimators=100, learning_rate=0.05),
}

# Store the detailed results for each model
cv_results_list = []
print("\nExecuting 5-Fold Cross-Validation for each model...")

for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    
    # Use cross_validate to get R², RMSE, and the train scores
    results = cross_validate(
        pipeline, 
        X_final_features, 
        y, 
        cv=cv_strategy, 
        scoring=('r2', 'neg_root_mean_squared_error'), # Specify multiple metrics
        return_train_score=True, #  Tell it to calculate train scores
        n_jobs=-1
    )
    
    # Calculate the mean of the metrics from the 5 folds
    mean_test_r2 = np.mean(results['test_r2'])
    mean_train_r2 = np.mean(results['train_r2'])
    # RMSE scores are returned negative, so we flip the sign
    mean_test_rmse = -np.mean(results['test_neg_root_mean_squared_error'])
    overfitting_delta = mean_train_r2 - mean_test_r2
    
    # Store the results
    cv_results_list.append({
        "Model": name,
        "Mean R² (Test)": mean_test_r2,
        "Mean R² (Train)": mean_train_r2,
        "Mean RMSE (Test)": mean_test_rmse,
        "Mean Overfitting Δ": overfitting_delta
    })
    
    print(f"- {name}: Mean R² = {mean_test_r2:.4f} | Mean RMSE = {mean_test_rmse:.4f} | Mean Overfitting Δ = {overfitting_delta:.4f}")

cv_results_df = pd.DataFrame(cv_results_list)


# 6. Validate the model
# The validation metrics (R², RMSE, MAE, Overfitting Δ) were calculated in the previous step
# during the model training loop for comparison.
# The best model is selected based on the R² score on the test set.

print("\n Cross-Validation Metrics Summary: ")

# Sort the results by the mean test R² score
cv_results_df = cv_results_df.sort_values(by="Mean R² (Test)", ascending=False)

# Find the best model
best_model_name = cv_results_df.iloc[0]['Model']

print(f"\n🏆 Best Performing Model: {best_model_name}")
print("\nDisplaying the mean scores from 5-fold cross-validation:")

# Retrain on Full Dataset and Save Artifacts
print(f"\n--- Retraining Best Model ({best_model_name}) on Full Dataset ---")

# 1. Get the best model class from the dictionary
best_model_class = models[best_model_name]

# 2. Create the final scaler and scale the full dataset
final_scaler = StandardScaler()
X_full_scaled = pd.DataFrame(final_scaler.fit_transform(X_final_features), columns=X_final_features.columns)

# 3. Create a new instance of the best model and train it on all the data
final_model = best_model_class
final_model.fit(X_full_scaled, y)
print("Final model has been trained.")

# 4. Save the final model and the scaler
print("\n--- Saving Final Model and Scaler ---")
joblib.dump(final_model, 'saved_model/best_model.joblib')
joblib.dump(final_scaler, 'saved_model/scaler.joblib')
print("Final model and scaler have been saved.")

# Save Final Features List
print("\n--- Saving Final Features List ---")
features = list(X_final_features.columns)
with open('saved_model/features.json', 'w') as f:
    json.dump(features, f)
print("Final features list has been saved.")


# 7. Check the applicability domain
print("Checking the Applicability Domain (AD)...")

# The AD is now based on the min/max of the entire dataset, because this is what
# the final model was trained on.
ad_params = X_final_features.agg(['min', 'max']).to_dict()
with open('saved_model/ad_params.json', 'w') as f:
    json.dump(ad_params, f)

print("- Saved AD parameters to 'saved_model/ad_params.json'")


# 8. Visualise and Interpret
print("\n--- Interpreting the Final Trained Model ---")

# The primary performance visualization is the boxplot of CV scores shown earlier.
# Here, we will visualize the feature importances of the *final model*
# that was trained on the entire dataset.

# Check for feature_importances_ (for tree-based models like RandomForest, LightGBM)
if hasattr(final_model, 'feature_importances_'):
    print(f"\nFeature Importance for the final {best_model_name} model:")
    
    # Use the final_model and the columns from the full dataset
    importances = final_model.feature_importances_
    feature_names = X_full_scaled.columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    top_n = 10
    print(feature_importance_df.head(top_n))

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n), palette="viridis")
    plt.title(f'Top {top_n} Feature Importance - Final {best_model_name} Model')
    plt.savefig('feature_importance_plot.png', bbox_inches='tight')
    print(f"\n- Saved Top {top_n} Feature Importance plot to 'feature_importance_plot.png'")

# Check for coef_ (for linear models like Linear Regression, Ridge)
elif hasattr(final_model, 'coef_'):
     print(f"\nFeature Coefficients for the final {best_model_name} model:")
     
     # Use the final_model and the columns from the full dataset
     coefficients = final_model.coef_
     feature_names = X_full_scaled.columns

     feature_importance_df = pd.DataFrame({
         'Feature': feature_names,
         'Coefficient': coefficients
     }).sort_values(by='Coefficient', key=abs, ascending=False) # Sort by absolute value

     top_n = 10
     print(feature_importance_df.head(top_n))

     # Visualization
     plt.figure(figsize=(10, 8))
     sns.barplot(x='Coefficient', y='Feature', data=feature_importance_df.head(top_n), palette='coolwarm')
     plt.title(f'Top {top_n} Feature Coefficients (by absolute value) - Final {best_model_name} Model')
     plt.savefig('feature_coefficients_plot.png', bbox_inches='tight')
     print(f"\n- Saved Top {top_n} Feature Coefficients plot to 'feature_coefficients_plot.png'")

else:
    print(f"\nFeature importance or coefficients are not available for the {best_model_name} model.")
