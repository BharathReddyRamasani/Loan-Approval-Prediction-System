import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_logistic_regression(X_train, y_train, random_state=42):
    """Trains a Logistic Regression model with class balancing."""
    print("Training Logistic Regression model...")
    # class_weight='balanced' is crucial here due to the ~88% vs ~12% class imbalance
    lr_model = LogisticRegression(
        max_iter=1000, 
        class_weight='balanced', 
        random_state=random_state,
        n_jobs=-1
    )
    lr_model.fit(X_train, y_train)
    return lr_model

def train_random_forest(X_train, y_train, random_state=42):
    """Trains a Random Forest classifier with limited depth for speed and generalization."""
    print("Training Random Forest model (this may take a minute due to dataset size)...")
    # Setting max_depth=12, min_samples_leaf=5, and n_estimators=100 to balance speed, model size, and accuracy.
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and returns key metrics and the confusion matrix."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics

def get_feature_importance(model, feature_names):
    """Extracts feature importance or coefficients depending on the model type."""
    if isinstance(model, RandomForestClassifier):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feature_importance_list = []
        for i in indices:
            feature_importance_list.append({
                'feature': feature_names[i],
                'importance': float(importances[i])
            })
        return pd.DataFrame(feature_importance_list)
        
    elif isinstance(model, LogisticRegression):
        # For Logistic Regression, we look at the absolute value of coefficients
        coefs = model.coef_[0]
        indices = np.argsort(np.abs(coefs))[::-1]
        
        feature_coef_list = []
        for i in indices:
            feature_coef_list.append({
                'feature': feature_names[i],
                'importance': float(np.abs(coefs[i])),
                'coefficient': float(coefs[i])
            })
        return pd.DataFrame(feature_coef_list)
        
    else:
        raise ValueError("Unsupported model type for feature importance extraction.")

def predict_risk_category(default_prob):
    """Maps default probability to risk levels."""
    risk_score = default_prob * 100
    if risk_score < 15:
        return 'Low Risk', 'green'
    elif risk_score < 35:
        return 'Medium Risk', 'orange'
    else:
        return 'High Risk', 'red'
