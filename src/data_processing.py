import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def download_and_load_data():
    """Downloads the Kaggle loan-default dataset (via a reliable raw GitHub mirror) and loads it as a pandas DataFrame."""
    url = "https://raw.githubusercontent.com/Pankaj-Str/Complete-Python-Mastery/refs/heads/main/53%20DataSet/Loan_default.csv"
    print(f"Loading data from URL mirror: {url}")
    df = pd.read_csv(url)
    return df

class LoanDataPreprocessor:
    """Handles feature engineering, scaling, and encoding for both training and inference."""
    def __init__(self):
        self.numeric_cols = [
            'Age', 'Income', 'LoanAmount', 'CreditScore', 
            'MonthsEmployed', 'NumCreditLines', 'InterestRate', 
            'LoanTerm', 'DTIRatio'
        ]
        self.binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']
        self.categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']
        
        self.scaler = StandardScaler()
        self.feature_columns_ = None  # Saves the exact order and list of features after preprocessing
        self.categorical_dummies_ = {} # Remembers the categories for dummy variables
        
    def _map_binary_cols(self, df):
        """Maps Yes/No binary columns to 1/0."""
        df_mapped = df.copy()
        for col in self.binary_cols:
            if col in df_mapped.columns:
                # Handle cases where user might pass boolean or numeric values
                df_mapped[col] = df_mapped[col].apply(lambda x: 1 if str(x).lower().strip() in ['yes', 'true', '1'] else 0)
        return df_mapped

    def fit(self, X):
        """Fits scalers and column indices on training features."""
        # Work on a copy to prevent modifying original df
        X_fit = X.copy()
        
        # 1. Map binary columns
        X_fit = self._map_binary_cols(X_fit)
        
        # 2. Fit Numeric Scaler
        self.scaler.fit(X_fit[self.numeric_cols])
        
        # 3. Handle Categorical Columns - Save unique categories for one-hot encoding consistency
        for col in self.categorical_cols:
            self.categorical_dummies_[col] = list(X_fit[col].unique())
            
        # 4. Determine final feature list structure
        # To do this, we transform a sample row to see what columns are produced
        dummy_df = pd.get_dummies(X_fit[self.categorical_cols + self.binary_cols], columns=self.categorical_cols)
        processed_sample = pd.concat([X_fit[self.numeric_cols], dummy_df], axis=1)
        self.feature_columns_ = list(processed_sample.columns)
        
        return self

    def transform(self, X):
        """Transforms features (binary mapping, one-hot encoding, scaling, column alignment)."""
        X_trans = X.copy()
        
        # 1. Map binary columns
        X_trans = self._map_binary_cols(X_trans)
        
        # 2. Scale numeric features
        scaled_numeric = pd.DataFrame(
            self.scaler.transform(X_trans[self.numeric_cols]),
            columns=self.numeric_cols,
            index=X_trans.index
        )
        
        # 3. One-hot encode categoricals
        encoded_cats = pd.get_dummies(X_trans[self.categorical_cols], columns=self.categorical_cols)
        
        # Combine all features
        combined = pd.concat([scaled_numeric, X_trans[self.binary_cols], encoded_cats], axis=1)
        
        # Align features to ensure we have exactly the same columns as when we fit
        # If columns are missing (e.g. at inference time), fill them with 0.
        # If columns are extra, drop them.
        for col in self.feature_columns_:
            if col not in combined.columns:
                combined[col] = 0
                
        # Reorder and filter columns to match training features exactly
        final_df = combined[self.feature_columns_]
        return final_df

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def transform_single_dict(self, data_dict):
        """Transforms a single dictionary of inputs (e.g. from Streamlit) into a 2D numpy array/DataFrame."""
        # Convert dictionary to DataFrame
        single_row_df = pd.DataFrame([data_dict])
        return self.transform(single_row_df)

def get_train_test_data(df, test_size=0.2, random_state=42):
    """Splits target and features, and creates train-test splits."""
    # Target
    y = df['Default']
    
    # Drop identifying column and target
    X = df.drop(columns=['LoanID', 'Default'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
