import os
import time
import pandas as pd
import numpy as np

# Import custom src packages
from src.data_processing import download_and_load_data, LoanDataPreprocessor, get_train_test_data
from src.models import train_logistic_regression, train_random_forest, evaluate_model, get_feature_importance
from src.clustering import train_kmeans, train_pca, get_cluster_profiles, get_pca_explained_variance, get_pca_loadings
from src.utils import save_artifact, save_metrics, ensure_directory_exists

def main():
    start_time = time.time()
    print("==================================================")
    print("🚀 STARTING LOAN APPROVAL SYSTEM ML PIPELINE")
    print("==================================================")
    
    # 1. Download and load data
    df = download_and_load_data()
    print(f"Data shape: {df.shape}")
    
    # 2. Split train/test data
    X_train, X_test, y_train, y_test = get_train_test_data(df, test_size=0.2, random_state=42)
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # 3. Fit preprocessor
    print("\n--- Step 1: Preprocessing & Scaling Features ---")
    preprocessor = LoanDataPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"Processed feature matrix columns count: {X_train_processed.shape[1]}")
    
    # Save preprocessor immediately
    save_artifact(preprocessor, "preprocessor.pkl")
    
    # 4. Train Classification Models
    print("\n--- Step 2: Training Logistic Regression Model ---")
    lr_model = train_logistic_regression(X_train_processed, y_train)
    lr_metrics = evaluate_model(lr_model, X_test_processed, y_test)
    print(f"Logistic Regression F1 Score: {lr_metrics['f1_score']:.4f}")
    save_artifact(lr_model, "logistic_regression.pkl")
    
    print("\n--- Step 3: Training Random Forest Model ---")
    rf_model = train_random_forest(X_train_processed, y_train)
    rf_metrics = evaluate_model(rf_model, X_test_processed, y_test)
    print(f"Random Forest F1 Score: {rf_metrics['f1_score']:.4f}")
    save_artifact(rf_model, "random_forest.pkl")
    
    # 5. Extract Feature Importances
    feature_names = preprocessor.feature_columns_
    lr_importance = get_feature_importance(lr_model, feature_names)
    rf_importance = get_feature_importance(rf_model, feature_names)
    
    # 6. Train Clustering Model (K-Means)
    print("\n--- Step 4: Training Customer Segmentation (K-Means) ---")
    # Using K=4 for segmentation
    kmeans_model = train_kmeans(X_train_processed, n_clusters=4)
    train_clusters = kmeans_model.predict(X_train_processed)
    
    # Save K-Means model
    save_artifact(kmeans_model, "kmeans.pkl")
    
    # Build cluster profiles based on original training data (including Default label for segment default rates)
    X_train_with_target = X_train.copy()
    X_train_with_target['Default'] = y_train
    cluster_profiles = get_cluster_profiles(X_train_with_target, train_clusters)
    print("Cluster Profiles Generated.")
    print(cluster_profiles)
    
    # 7. Train Dimensionality Reduction (PCA)
    print("\n--- Step 5: Training Dimensionality Reduction (PCA) ---")
    pca_model = train_pca(X_train_processed, n_components=3)
    save_artifact(pca_model, "pca.pkl")
    
    ind_var, cum_var = get_pca_explained_variance(pca_model)
    pca_loadings = get_pca_loadings(pca_model, feature_names)
    print(f"PCA Cumulative Explained Variance (3 PCs): {cum_var[-1]:.4f}")
    
    # 8. Save Metrics & Outputs for Streamlit Dashboard
    print("\n--- Step 6: Compiling and Saving Metrics for Dashboard ---")
    metrics_summary = {
        'logistic_regression_metrics': {
            'accuracy': lr_metrics['accuracy'],
            'precision': lr_metrics['precision'],
            'recall': lr_metrics['recall'],
            'f1_score': lr_metrics['f1_score'],
            'roc_auc': lr_metrics['roc_auc'],
            'confusion_matrix': lr_metrics['confusion_matrix']
        },
        'random_forest_metrics': {
            'accuracy': rf_metrics['accuracy'],
            'precision': rf_metrics['precision'],
            'recall': rf_metrics['recall'],
            'f1_score': rf_metrics['f1_score'],
            'roc_auc': rf_metrics['roc_auc'],
            'confusion_matrix': rf_metrics['confusion_matrix']
        },
        'lr_importance': lr_importance.to_dict(orient='records'),
        'rf_importance': rf_importance.to_dict(orient='records'),
        'cluster_profiles': cluster_profiles.to_dict(orient='records'),
        'pca_explained_variance': {
            'individual': ind_var.tolist(),
            'cumulative': cum_var.tolist()
        },
        'pca_loadings': pca_loadings.to_dict(orient='index')
    }
    
    save_metrics(metrics_summary, "metrics.json")
    
    # Add a sample subset of PCA projections for fast background visualizer
    print("\nSaving a small subset of training PCA projection coordinates for visualization...")
    X_train_sample = X_train_processed.sample(n=1000, random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]
    
    pca_coords = pca_model.transform(X_train_sample)
    pca_vis_df = pd.DataFrame(
        pca_coords,
        columns=['PC1', 'PC2', 'PC3'],
        index=X_train_sample.index
    )
    # Join with original variables for tooltips
    original_sample = X_train.loc[X_train_sample.index]
    pca_vis_df['Cluster'] = kmeans_model.predict(X_train_sample)
    pca_vis_df['Default'] = y_train_sample
    pca_vis_df['Age'] = original_sample['Age']
    pca_vis_df['Income'] = original_sample['Income']
    pca_vis_df['LoanAmount'] = original_sample['LoanAmount']
    pca_vis_df['CreditScore'] = original_sample['CreditScore']
    
    pca_vis_df.to_csv("models/pca_vis_subset.csv", index=False)
    print("PCA visualization sample dataset saved.")
    
    elapsed_time = time.time() - start_time
    print("==================================================")
    print(f"🎉 PIPELINE COMPLETED IN {elapsed_time/60:.2f} MINUTES")
    print("==================================================")

if __name__ == "__main__":
    main()
