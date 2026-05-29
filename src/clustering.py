import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def train_kmeans(X_train_processed, n_clusters=4, random_state=42):
    """Trains K-Means model on the processed (scaled/encoded) features."""
    print(f"Training K-Means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans.fit(X_train_processed)
    return kmeans

def train_pca(X_train_processed, n_components=3, random_state=42):
    """Fits PCA model on the processed features."""
    print(f"Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_train_processed)
    return pca

def get_cluster_profiles(df_original, cluster_labels):
    """Computes averages of numerical columns for each cluster to build profiles.
    
    df_original should be the un-scaled dataframe with original values, but 
    containing the numerical and binary columns of interest.
    """
    df_copy = df_original.copy()
    # Ensure columns are numeric
    numeric_cols = [
        'Age', 'Income', 'LoanAmount', 'CreditScore', 
        'MonthsEmployed', 'NumCreditLines', 'InterestRate', 
        'LoanTerm', 'DTIRatio', 'Default'
    ]
    # Filter to numeric columns that are present in the df
    cols_to_profile = [col for col in numeric_cols if col in df_copy.columns]
    
    df_copy['Cluster'] = cluster_labels
    profiles = df_copy.groupby('Cluster')[cols_to_profile].mean().reset_index()
    
    # Let's count how many samples are in each cluster
    counts = df_copy.groupby('Cluster').size().reset_index(name='Size')
    profiles = pd.merge(profiles, counts, on='Cluster')
    
    return profiles

def get_pca_explained_variance(pca):
    """Returns the explained variance and cumulative explained variance ratios."""
    individual_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(individual_var)
    return individual_var, cumulative_var

def get_pca_loadings(pca, feature_names):
    """Extracts the weights (loadings) of each feature for the principal components."""
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
        index=feature_names
    )
    return loadings

def interpret_cluster(cluster_id, profile_row):
    """Returns a rich business label and details for a customer segment based on its profile."""
    # Custom business rules based on the profiles
    income = profile_row.get('Income', 50000)
    credit_score = profile_row.get('CreditScore', 600)
    interest_rate = profile_row.get('InterestRate', 12)
    default_rate = profile_row.get('Default', 0.1) * 100
    
    if credit_score > 680 and income > 80000:
        label = "Premium Low-Risk Borrowers"
        desc = "High-income individuals with excellent credit history. They secure lower interest rates and have extremely low default rates."
        color = "green"
    elif credit_score < 580 and default_rate > 18:
        label = "Subprime High-Risk Borrowers"
        desc = "Borrowers with lower credit scores and higher default probability. They face high interest rates and represent the highest default risk segment."
        color = "red"
    elif interest_rate > 15:
        label = "High-Interest / Moderate Risk"
        desc = "Borrowers paying high interest rates but maintaining average income and credit. Moderate default risk."
        color = "orange"
    elif income < 40000:
        label = "Entry-Level / Low-Income Borrowers"
        desc = "Younger or lower-income borrowers with small-to-moderate loan requirements. Moderate default risk due to limited financial buffer."
        color = "blue"
    else:
        label = "Standard Balanced Borrowers"
        desc = "Average income, stable credit score, and moderate interest rates. Represents the bank's core steady customer base."
        color = "gray"
        
    return {
        'label': label,
        'description': desc,
        'color': color,
        'avg_income': float(income),
        'avg_credit': float(credit_score),
        'avg_interest': float(interest_rate),
        'default_rate': float(default_rate)
    }
