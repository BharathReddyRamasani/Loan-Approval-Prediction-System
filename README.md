# 🏦 FinGuard | Loan Approval & Credit Risk Intelligence System

FinGuard is a professional, production-grade end-to-end Machine Learning platform and interactive dashboard designed to assess, score, and segment loan applicants. Built using Kaggle's public **Loan Default Dataset** (255,347 entries), it demonstrates the combined power of classification, risk modeling, unsupervised clustering, and dimensionality reduction.

---

## 📈 Algorithms & Features

This system integrates four distinct ML workflows to provide a complete credit evaluation ecosystem:

1. **Logistic Regression (Decision Classifier)**  
   * **Purpose**: Generates the binary **Loan Approved / Rejected** decision.
   * **Implementation**: Uses inverse-class frequency balancing (`class_weight='balanced'`) to handle data imbalance and establish stable decision thresholds.

2. **Random Forest Classifier (Risk Scorer)**  
   * **Purpose**: Computes continuous **Default Probability** (0% - 100%) and maps it to credit risk tiers.
   * **Implementation**: Uses an ensemble of 100 decision trees to generate calibrated probabilities, categorizing applicants as **Low Risk** (<15%), **Medium Risk** (15%-35%), or **High Risk** (>35%).

3. **K-Means Clustering (Customer Segmentation)**  
   * **Purpose**: Clusters applicants into **4 distinct credit profiles** to help the bank craft tailored marketing and financial strategies.
   * **Archetypes**:
     * 🟢 *Premium Low-Risk Borrowers*: High income, high FICO scores, low interest rates.
     * 🔴 *Subprime High-Risk Borrowers*: Low FICO, high default rates.
     * 🟡 *High-Interest / Moderate Risk*: Moderate profiles with higher interest rates.
     * 🔵 *Entry-Level / Low-Income Borrowers*: Young or lower-income with small-to-moderate loans.

4. **Principal Component Analysis (PCA) (Feature Reduction)**  
   * **Purpose**: Projects the 20+ scaled and encoded categorical columns into orthogonal principal components for 2D/3D visualizations.
   * **Implementation**: Provides loading coefficient analysis to understand what features explain the variance, and displays the applicant's relative position in the borrower space.

---

## 📁 Repository Structure

```text
Loan-Approval-Prediction-System/
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # Dataset downloader, scaler, and one-hot encoder
│   ├── models.py            # Logistic Regression and Random Forest wrappers
│   ├── clustering.py        # K-Means and PCA pipelines + interpretations
│   └── utils.py             # Model serialization & metric saving helper functions
├── models/                  # [GENERATED] Saved PKL models, scalers, and JSON metrics
├── app.py                   # Custom Streamlit UI dashboard
├── train_pipeline.py        # Pipeline execution script to train all models
├── requirements.txt         # Project package dependencies
├── .gitignore               # Standard Python gitignore rules
└── README.md                # Project documentation and deployment guide
```

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python 3.9+ installed. Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Machine Learning Pipeline
To download the dataset, pre-process the attributes, train the classifiers, run K-Means and PCA, and output all serialization assets:

```bash
python train_pipeline.py
```
This script will output evaluation metrics to the console and save the trained models inside the `models/` directory.

### 3. Launch the Dashboard
To start the interactive Streamlit user interface locally:

```bash
streamlit run app.py
```
This will open the dashboard in your default browser (usually at `http://localhost:8501`).

---

## 📊 Streamlit Dashboard Layout

* **🔮 Decision Calculator**: Input custom parameters (Age, Income, Credit Score, DTI, etc.) on the credit application form to get real-time decisions, a risk gauge, cluster assignment, and a 2D PCA mapping.
* **👥 Customer Segments**: Explore cluster demographics, sizes, and default rates in bar/scatter plots and a rotating 3D scatter plot.
* **📊 Model Performance**: Inspect model validation stats (Precision, Recall, F1, ROC-AUC) side-by-side, along with Confusion Matrices and Feature Importance rankings.
* **🔬 Feature Dimensionality (PCA)**: Analyze explained variance ratios and examine feature loading weights to see which parameters influence PCA dimensions.

---

## 🌐 Git Push & Streamlit Deployment

### Push Code to GitHub
1. Initialize git and commit the files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Loan Approval Prediction System complete ML pipeline and Streamlit dashboard"
   ```
2. Create a repository on GitHub (e.g., `Loan-Approval-Prediction-System`).
3. Link the remote and push:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/Loan-Approval-Prediction-System.git
   git branch -M main
   git push -u origin main
   ```

### Deploy to Streamlit Community Cloud
1. Go to [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New App**.
3. Select your repository (`Loan-Approval-Prediction-System`), branch (`main`), and main file path (`app.py`).
4. Click **Deploy**. Streamlit will automatically read `requirements.txt`, install dependencies, run the application, and provide a public URL!