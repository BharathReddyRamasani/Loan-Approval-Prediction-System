import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os

# Set page config for a premium wide-layout dashboard
st.set_page_config(
    page_title="FinGuard | Intelligent Credit Decisions",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS to inject dark-mode friendly accents and glassmorphism styling
st.markdown("""
<style>
    /* Styling for custom metric cards */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Glowing borders for Decision output */
    .approved-box {
        background-color: rgba(16, 185, 129, 0.1);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }
    .rejected-box {
        background-color: rgba(239, 68, 68, 0.1);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
    }
    
    /* Header typography styling */
    .title-container {
        padding: 20px 0px;
        border-bottom: 1px solid #334155;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load models and cached assets
@st.cache_resource
def load_ml_assets():
    """Loads and caches the model, scaler, clustering, and PCA estimators. Trains them if missing."""
    model_path = "models/preprocessor.pkl"
    if not os.path.exists(model_path):
        try:
            # We use st.write/spinner inside cache_resource which is allowed for notification
            with st.spinner("🤖 Pre-trained models not found. Running the machine learning pipeline to train them on the fly (takes ~15s)..."):
                import train_pipeline
                train_pipeline.main()
        except Exception as e:
            st.error(f"Failed to train models on the fly: {e}")
            return None, None, None, None, None
            
    try:
        preprocessor = joblib.load("models/preprocessor.pkl")
        lr_model = joblib.load("models/logistic_regression.pkl")
        rf_model = joblib.load("models/random_forest.pkl")
        kmeans_model = joblib.load("models/kmeans.pkl")
        pca_model = joblib.load("models/pca.pkl")
        return preprocessor, lr_model, rf_model, kmeans_model, pca_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None

@st.cache_data
def load_metrics_data():
    """Loads and caches evaluation metrics and profiles computed during training."""
    try:
        with open("models/metrics.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics summary: {e}")
        return None

@st.cache_data
def load_pca_subset():
    """Loads a pre-computed PCA coordinates subset of training data for background visualizer."""
    if os.path.exists("models/pca_vis_subset.csv"):
        return pd.read_csv("models/pca_vis_subset.csv")
    return None

# Perform mapping of clusters to business profiles
def get_segment_interpretation(cluster_id, profiles):
    """Maps K-Means cluster ID to detailed borrower profiles."""
    if profiles is None:
        return {"label": f"Segment {cluster_id}", "description": "No description available.", "color": "blue"}
    
    # Locate profile details for this cluster
    row = next((item for item in profiles if item['Cluster'] == cluster_id), None)
    if not row:
        return {"label": f"Segment {cluster_id}", "description": "No description available.", "color": "blue"}
        
    income = row.get('Income', 50000)
    credit_score = row.get('CreditScore', 600)
    interest_rate = row.get('InterestRate', 12)
    default_rate = row.get('Default', 0.1) * 100
    
    # Apply segment labeling rules consistent with training
    if credit_score > 680 and income > 80000:
        return {
            'label': "Premium Low-Risk Borrowers",
            'description': "High-income individuals with excellent credit history. They secure lower interest rates and represent the lowest default risk segment.",
            'color': "green",
            'badge': "🟢 Premium Segment",
            'avg_income': income,
            'avg_credit': credit_score,
            'avg_interest': interest_rate,
            'default_rate': default_rate
        }
    elif credit_score < 580 or default_rate > 18:
        return {
            'label': "Subprime High-Risk Borrowers",
            'description': "Borrowers with lower credit scores and higher default probability. They face high interest rates and represent the highest default risk segment.",
            'color': "red",
            'badge': "🔴 Subprime Segment",
            'avg_income': income,
            'avg_credit': credit_score,
            'avg_interest': interest_rate,
            'default_rate': default_rate
        }
    elif interest_rate > 15:
        return {
            'label': "High-Interest / Moderate Risk",
            'description': "Borrowers paying high interest rates but maintaining average income and credit. Moderate default risk.",
            'color': "orange",
            'badge': "🟡 High-Interest Segment",
            'avg_income': income,
            'avg_credit': credit_score,
            'avg_interest': interest_rate,
            'default_rate': default_rate
        }
    elif income < 40000:
        return {
            'label': "Entry-Level / Low-Income Borrowers",
            'description': "Younger or lower-income borrowers with small-to-moderate loan requirements. Moderate default risk due to limited financial buffer.",
            'color': "blue",
            'badge': "🔵 Entry-Level Segment",
            'avg_income': income,
            'avg_credit': credit_score,
            'avg_interest': interest_rate,
            'default_rate': default_rate
        }
    else:
        return {
            'label': "Standard Balanced Borrowers",
            'description': "Average income, stable credit score, and moderate interest rates. Represents the bank's core steady customer base.",
            'color': "gray",
            'badge': "⚪ Standard Segment",
            'avg_income': income,
            'avg_credit': credit_score,
            'avg_interest': interest_rate,
            'default_rate': default_rate
        }

# Main App Logic
def main():
    # Sidebar Metadata & Logo
    st.sidebar.markdown("""
    <div style='text-align: center; padding-bottom: 10px;'>
        <h1 style='font-size: 1.8rem; margin: 0; color: #3b82f6;'>🏦 FinGuard AI</h1>
        <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>Credit Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.divider()
    
    # Load assets
    preprocessor, lr_model, rf_model, kmeans_model, pca_model = load_ml_assets()
    metrics = load_metrics_data()
    pca_subset = load_pca_subset()
    
    if preprocessor is None or lr_model is None or rf_model is None:
        st.warning("⚠️ Warning: Pre-trained models not found in the `models/` directory. Please run the training pipeline `python train_pipeline.py` first to proceed.")
        st.stop()
        
    st.sidebar.success("✅ Models Loaded Successfully")
    st.sidebar.markdown("""
    ### System Architecture
    - **Decision Classifier**: Logistic Regression (Balanced)
    - **Risk Scorer**: Random Forest Classifier
    - **Segmentation Engine**: K-Means Clustering ($K=4$)
    - **Reduction Engine**: PCA (3 Principal Components)
    """)
    st.sidebar.divider()
    st.sidebar.markdown("""
    **Dataset**: Kaggle Loan Default Dataset  
    **Volume**: 255,347 borrower profiles  
    **Features**: 17 original dimensions  
    """)
    
    # Header Banner
    st.markdown("""
    <div class="title-container">
        <h1 style='margin:0; font-size:2.5rem; font-weight:700;'>🏦 FinGuard Loan Decision Portal</h1>
        <p style='margin:5px 0 0 0; color:#94a3b8; font-size:1.1rem;'>Professional credit risk assessment platform powered by Logistic Regression, Random Forest, K-Means Clustering, and Principal Component Analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tab Layout
    tab_calc, tab_segments, tab_performance, tab_pca = st.tabs([
        "🔮 Decision Calculator", 
        "👥 Customer Segments", 
        "📊 Model Performance", 
        "🔬 Feature Dimensionality (PCA)"
    ])
    
    # ----------------------------------------------------
    # TAB 1: DECISION CALCULATOR
    # ----------------------------------------------------
    with tab_calc:
        st.subheader("Predictive Credit Assessment Sandbox")
        st.markdown("Fill out the applicant details below to run real-time classification, default risk scoring, and customer segment profiling.")
        
        # Split layout into input form and results display
        col_inputs, col_results = st.columns([5, 4])
        
        with col_inputs:
            st.markdown("##### 📝 Applicant Information Form")
            
            with st.expander("👤 Demographic & Social Metrics", expanded=True):
                c1, c2, c3 = st.columns(3)
                age = c1.slider("Age (Years)", min_value=18, max_value=90, value=38)
                education = c2.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
                marital = c3.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                
                c4, c5 = st.columns(2)
                dependents = c4.selectbox("Has Dependents?", ["No", "Yes"])
                mortgage = c5.selectbox("Has Mortgage?", ["No", "Yes"])
                
            with st.expander("💼 Financial & Employment Profiles", expanded=True):
                c1, c2, c3 = st.columns(3)
                income = c1.number_input("Annual Income ($)", min_value=5000, max_value=1000000, value=65000, step=5000)
                employed = c2.slider("Months Employed", min_value=0, max_value=120, value=48)
                emp_type = c3.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
                
                c4, c5 = st.columns(2)
                credit_lines = c4.slider("Number of Credit Lines", min_value=1, max_value=20, value=3)
                credit_score = c5.slider("Credit Score (FICO)", min_value=300, max_value=850, value=680)
                
            with st.expander("💵 Loan Details", expanded=True):
                c1, c2, c3 = st.columns(3)
                loan_amount = c1.number_input("Loan Amount Requested ($)", min_value=1000, max_value=1000000, value=45000, step=5000)
                loan_term = c2.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=2)
                interest_rate = c3.slider("Interest Rate (%)", min_value=1.0, max_value=35.0, value=11.5, step=0.1)
                
                c4, c5 = st.columns(2)
                dti = c4.slider("Debt-To-Income (DTI) Ratio", min_value=0.01, max_value=0.99, value=0.28, step=0.01)
                loan_purpose = c5.selectbox("Loan Purpose", ["Home", "Business", "Education", "Auto", "Other"])
                
                cosigner = st.selectbox("Has Co-signer?", ["No", "Yes"])
            
            # Map input elements to match our data processor structure
            user_input_dict = {
                'Age': age,
                'Income': income,
                'LoanAmount': loan_amount,
                'CreditScore': credit_score,
                'MonthsEmployed': employed,
                'NumCreditLines': credit_lines,
                'InterestRate': interest_rate,
                'LoanTerm': loan_term,
                'DTIRatio': dti,
                'Education': education,
                'EmploymentType': emp_type,
                'MaritalStatus': marital,
                'HasMortgage': mortgage,
                'HasDependents': dependents,
                'LoanPurpose': loan_purpose,
                'HasCoSigner': cosigner
            }
            
        with col_results:
            st.markdown("##### ⚡ Real-Time Decision Analytics")
            
            # Execute Pipeline on Current Input
            # Preprocess the single row
            input_processed = preprocessor.transform_single_dict(user_input_dict)
            
            # 1. Classification (Logistic Regression)
            lr_pred = lr_model.predict(input_processed)[0]
            # y=1 represents Default, so y=0 represents Safe (Approved)
            is_approved = (lr_pred == 0)
            
            # 2. Risk Estimation (Random Forest Probability)
            default_prob = rf_model.predict_proba(input_processed)[0][1]
            risk_pct = default_prob * 100
            
            # Map risk level
            if risk_pct < 15:
                risk_label = "LOW DEFAULT RISK"
                risk_color = "#10b981"
            elif risk_pct < 35:
                risk_label = "MODERATE DEFAULT RISK"
                risk_color = "#f59e0b"
            else:
                risk_label = "HIGH DEFAULT RISK"
                risk_color = "#ef4444"
                
            # 3. K-Means Segment
            cluster_id = kmeans_model.predict(input_processed)[0]
            segment_info = get_segment_interpretation(cluster_id, metrics.get('cluster_profiles') if metrics else None)
            
            # 4. PCA Coordinates
            pca_coords = pca_model.transform(input_processed)[0]
            
            # Display Approval Decision
            if is_approved:
                st.markdown(f"""
                <div class="approved-box">
                    <span style="font-size:0.9rem; color:#10b981; font-weight:700; text-transform:uppercase; letter-spacing:0.1em;">Final Decision</span>
                    <h2 style="color:#10b981; margin:5px 0 0 0; font-size:2.8rem; font-weight:800;">LOAN APPROVED</h2>
                    <p style="color:#a7f3d0; font-size:0.95rem; margin:8px 0 0 0;">Applicant exhibits safe credit indicators according to the Logistic Regression model.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="rejected-box">
                    <span style="font-size:0.9rem; color:#ef4444; font-weight:700; text-transform:uppercase; letter-spacing:0.1em;">Final Decision</span>
                    <h2 style="color:#ef4444; margin:5px 0 0 0; font-size:2.8rem; font-weight:800;">LOAN REJECTED</h2>
                    <p style="color:#fecaca; font-size:0.95rem; margin:8px 0 0 0;">Applicant fails safety thresholds. Risk of default exceeds permissible limits.</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display Risk Gauge (Random Forest Risk Prediction)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_pct,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Risk Score: {risk_label}", 'font': {'size': 16, 'color': risk_color}},
                number = {'suffix': "%", 'font': {'size': 32}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                    'bar': {'color': risk_color},
                    'bgcolor': "#1e293b",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 15], 'color': 'rgba(16, 185, 129, 0.1)'},
                        {'range': [15, 35], 'color': 'rgba(245, 158, 11, 0.1)'},
                        {'range': [35, 100], 'color': 'rgba(239, 68, 68, 0.1)'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f8fafc"},
                margin=dict(l=20, r=20, t=40, b=10),
                height=220
            )
            st.plotly_chart(fig_gauge, width="stretch")
            
            # Customer Segment Profiling (K-Means Output)
            st.markdown(f"""
            <div class="metric-card" style="border-left: 5px solid {segment_info['color']};">
                <div class="metric-label">👥 Customer Segment Archetype</div>
                <div class="metric-value" style="font-size:1.4rem; color:#f8fafc;">{segment_info['label']}</div>
                <div style="font-size:0.85rem; color:#94a3b8; margin-top:8px; line-height:1.4;">
                    {segment_info['description']}
                </div>
                <div style="margin-top: 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.8rem; border-top: 1px solid #334155; padding-top: 10px;">
                    <div>Average Income: <b>${segment_info.get('avg_income', 0):,.0f}</b></div>
                    <div>Average FICO: <b>{segment_info.get('avg_credit', 0):.0f}</b></div>
                    <div>Average Interest: <b>{segment_info.get('avg_interest', 0):.1f}%</b></div>
                    <div style="color:#ef4444;">Segment Default Rate: <b>{segment_info.get('default_rate', 0):.1f}%</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Applicant PCA Coordinates relative projection plot
            if pca_subset is not None:
                st.markdown("<br>##### 🔬 Projection in Dimensionality Reduction Space")
                st.markdown("<p style='color:#94a3b8; font-size:0.8rem; margin: -5px 0 10px 0;'>Your applicant (glowing star ⭐) is projected onto the 2D PCA space derived from 1,000 background borrowers.</p>", unsafe_allow_html=True)
                
                # Plot PCA projections
                fig_pca_proj = px.scatter(
                    pca_subset, 
                    x='PC1', 
                    y='PC2', 
                    color='Cluster',
                    color_continuous_scale='Turbo',
                    hover_data=['Age', 'Income', 'LoanAmount', 'CreditScore'],
                    opacity=0.3,
                    labels={'Cluster': 'Segment'}
                )
                
                # Add the current applicant as a custom large star marker
                fig_pca_proj.add_trace(go.Scatter(
                    x=[pca_coords[0]],
                    y=[pca_coords[1]],
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=15,
                        color='#ffffff',
                        line=dict(color='#3b82f6', width=2)
                    ),
                    name='Current Applicant',
                    hovertext=f"Current Applicant:<br>FICO: {credit_score}<br>Income: ${income:,.0f}<br>Loan: ${loan_amount:,.0f}"
                ))
                
                fig_pca_proj.update_layout(
                    paper_bgcolor='rgba(15, 23, 42, 0.4)',
                    plot_bgcolor='rgba(15, 23, 42, 0.4)',
                    font={'color': "#f8fafc"},
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=260,
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='Component 1'),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title='Component 2')
                )
                st.plotly_chart(fig_pca_proj, width="stretch")
            
    # ----------------------------------------------------
    # TAB 2: CUSTOMER SEGMENTS (K-MEANS)
    # ----------------------------------------------------
    with tab_segments:
        st.subheader("Customer Segmentation & Demographics Analysis")
        st.markdown("We segment the 255k borrower database using K-Means Clustering ($K=4$) to capture distinct customer profiles.")
        
        if metrics and 'cluster_profiles' in metrics:
            profiles = metrics['cluster_profiles']
            
            # Create interactive segment selection to show profiles
            st.markdown("##### 👥 Segment Profiles Summary")
            
            # Render Cards for each Segment
            cols = st.columns(4)
            for i, p_row in enumerate(profiles):
                seg_detail = get_segment_interpretation(i, profiles)
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card" style="border-top: 4px solid {seg_detail['color']}; height: 100%;">
                        <div style="font-weight: 700; color: {seg_detail['color']}; font-size: 1.1rem; margin-bottom: 5px;">
                            {seg_detail['badge']}
                        </div>
                        <p style="font-size: 0.8rem; color: #94a3b8; min-height: 50px; margin-bottom:10px;">
                            {seg_detail['description']}
                        </p>
                        <hr style="border-color:#334155; margin: 8px 0;">
                        <table style="width:100%; font-size:0.8rem; color:#e2e8f0; line-height:1.6;">
                            <tr><td><b>FICO Score:</b></td><td style="text-align:right;">{p_row['CreditScore']:.0f}</td></tr>
                            <tr><td><b>Avg Income:</b></td><td style="text-align:right;">${p_row['Income']:,.0f}</td></tr>
                            <tr><td><b>Loan Amount:</b></td><td style="text-align:right;">${p_row['LoanAmount']:,.0f}</td></tr>
                            <tr><td><b>Interest Rate:</b></td><td style="text-align:right;">{p_row['InterestRate']:.2f}%</td></tr>
                            <tr><td><b>DTI Ratio:</b></td><td style="text-align:right;">{p_row['DTIRatio']:.2f}</td></tr>
                            <tr style="color:{seg_detail['color']}; font-weight:700;">
                                <td><b>Default Rate:</b></td><td style="text-align:right;">{p_row['Default']*100:.1f}%</td></tr>
                            <tr><td><b>Volume Size:</b></td><td style="text-align:right;">{p_row['Size']:,}</td></tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Interactive Radar or Comparison Charts
            st.markdown("<br>##### 📊 Segment Dimension Comparisons", unsafe_allow_html=True)
            
            prof_df = pd.DataFrame(profiles)
            
            # Map segment labels to df for readable labels in plot
            prof_df['SegmentName'] = [get_segment_interpretation(cid, profiles)['label'] for cid in prof_df['Cluster']]
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Comparison of Income vs Loan Amount
                fig_inc_loan = px.bar(
                    prof_df,
                    x='SegmentName',
                    y=['Income', 'LoanAmount'],
                    barmode='group',
                    title='Average Annual Income vs Loan Amount Requested by Segment',
                    labels={'value': 'USD ($)', 'variable': 'Metric', 'SegmentName': 'Borrower Segment'},
                    color_discrete_sequence=['#3b82f6', '#10b981']
                )
                fig_inc_loan.update_layout(
                    paper_bgcolor='rgba(30, 41, 59, 0.2)',
                    plot_bgcolor='rgba(30, 41, 59, 0.2)',
                    font={'color': "#f8fafc"},
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_inc_loan, width="stretch")
                
            with col_chart2:
                # Correlation between FICO and default rates by cluster
                fig_fico_def = px.scatter(
                    prof_df,
                    x='CreditScore',
                    y='InterestRate',
                    size='Size',
                    color='SegmentName',
                    title='Interest Rate vs Credit Score (Bubble size represents Segment Size)',
                    labels={'CreditScore': 'Average FICO Score', 'InterestRate': 'Average Interest Rate (%)', 'SegmentName': 'Segment'},
                    color_discrete_sequence=['#10b981', '#ef4444', '#f59e0b', '#3b82f6']
                )
                fig_fico_def.update_layout(
                    paper_bgcolor='rgba(30, 41, 59, 0.2)',
                    plot_bgcolor='rgba(30, 41, 59, 0.2)',
                    font={'color': "#f8fafc"},
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_fico_def, width="stretch")
                
            # 3D Cluster Visualizer using the PCA subset
            if pca_subset is not None:
                st.markdown("<br>##### 👥 3D Cluster Topology Visualization", unsafe_allow_html=True)
                st.markdown("<p style='color:#94a3b8; font-size:0.85rem; margin: -5px 0 10px 0;'>Visualizing K-Means clusters inside the 3-dimensional principal component space. Drag to rotate and zoom.</p>", unsafe_allow_html=True)
                
                pca_subset['SegmentName'] = [get_segment_interpretation(cid, profiles)['label'] for cid in pca_subset['Cluster']]
                
                fig_3d = px.scatter_3d(
                    pca_subset,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color='SegmentName',
                    opacity=0.6,
                    size_max=8,
                    hover_data=['Age', 'Income', 'LoanAmount', 'CreditScore'],
                    color_discrete_sequence=['#10b981', '#ef4444', '#f59e0b', '#3b82f6'],
                    labels={'SegmentName': 'Segment'}
                )
                fig_3d.update_layout(
                    scene=dict(
                        xaxis=dict(backgroundcolor="rgba(15, 23, 42, 0.8)", gridcolor="#334155", showbackground=True, zerolinecolor="#334155"),
                        yaxis=dict(backgroundcolor="rgba(15, 23, 42, 0.8)", gridcolor="#334155", showbackground=True, zerolinecolor="#334155"),
                        zaxis=dict(backgroundcolor="rgba(15, 23, 42, 0.8)", gridcolor="#334155", showbackground=True, zerolinecolor="#334155")
                    ),
                    paper_bgcolor='rgba(30, 41, 59, 0.2)',
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=500,
                    legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1)
                )
                st.plotly_chart(fig_3d, width="stretch")
                
    # ----------------------------------------------------
    # TAB 3: MODEL PERFORMANCE & METRICS
    # ----------------------------------------------------
    with tab_performance:
        st.subheader("Model Validation & Explainability Metrics")
        st.markdown("We assess and compare the validation metrics for **Logistic Regression** and **Random Forest** models side-by-side.")
        
        if metrics:
            lr_m = metrics['logistic_regression_metrics']
            rf_m = metrics['random_forest_metrics']
            
            # Display Validation Cards Side-By-Side
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("""
                <div class="metric-card" style="border-top: 4px solid #3b82f6;">
                    <div class="metric-label" style="color:#3b82f6; font-size:1.1rem; font-weight:700;">Logistic Regression (Decision Classifier)</div>
                    <p style="font-size:0.8rem; color:#94a3b8; margin: 4px 0 12px 0;">Optimized using inverse-class frequency weights to establish the binary Approved/Rejected threshold.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics Table
                sub_c1, sub_c2, sub_c3, sub_c4 = st.columns(4)
                sub_c1.metric("Accuracy", f"{lr_m['accuracy']:.2%}")
                sub_c2.metric("Precision", f"{lr_m['precision']:.2%}")
                sub_c3.metric("Recall", f"{lr_m['recall']:.2%}")
                sub_c4.metric("F1 Score", f"{lr_m['f1_score']:.2%}")
                
            with c2:
                st.markdown("""
                <div class="metric-card" style="border-top: 4px solid #10b981;">
                    <div class="metric-label" style="color:#10b981; font-size:1.1rem; font-weight:700;">Random Forest Classifier (Risk Scorer)</div>
                    <p style="font-size:0.8rem; color:#94a3b8; margin: 4px 0 12px 0;">Trained with ensemble methods to yield calibrated risk probabilities (0% - 100% risk output).</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics Table
                sub_c1, sub_c2, sub_c3, sub_c4 = st.columns(4)
                sub_c1.metric("Accuracy", f"{rf_m['accuracy']:.2%}")
                sub_c2.metric("Precision", f"{rf_m['precision']:.2%}")
                sub_c3.metric("Recall", f"{rf_m['recall']:.2%}")
                sub_c4.metric("F1 Score", f"{rf_m['f1_score']:.2%}")
                
            st.markdown("<br>##### 🔍 Feature Explainability Analysis", unsafe_allow_html=True)
            st.markdown("<p style='color:#94a3b8; font-size:0.85rem; margin: -5px 0 10px 0;'>What drives credit risk and loan default in the bank's decision model? The charts below show feature importance for both classifiers.</p>", unsafe_allow_html=True)
            
            col_imp1, col_imp2 = st.columns(2)
            
            with col_imp1:
                # RF Importance
                rf_imp_df = pd.DataFrame(metrics['rf_importance']).head(12)
                fig_rf_imp = px.bar(
                    rf_imp_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Random Forest Feature Importance (Top 12 Features)',
                    labels={'importance': 'Information Gain / Importance', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig_rf_imp.update_layout(
                    paper_bgcolor='rgba(30, 41, 59, 0.2)',
                    plot_bgcolor='rgba(30, 41, 59, 0.2)',
                    font={'color': "#f8fafc"},
                    margin=dict(l=20, r=20, t=50, b=20),
                    yaxis={'categoryorder': 'total ascending'},
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_rf_imp, width="stretch")
                
            with col_imp2:
                # LR Importance
                lr_imp_df = pd.DataFrame(metrics['lr_importance']).head(12)
                fig_lr_imp = px.bar(
                    lr_imp_df,
                    x='coefficient',
                    y='feature',
                    orientation='h',
                    title='Logistic Regression Feature Coefficients (Positive = Default Indicator, Negative = Safety Indicator)',
                    labels={'coefficient': 'Model Coefficient (β)', 'feature': 'Feature'},
                    color='coefficient',
                    color_continuous_scale='RdYlGn_r'
                )
                fig_lr_imp.update_layout(
                    paper_bgcolor='rgba(30, 41, 59, 0.2)',
                    plot_bgcolor='rgba(30, 41, 59, 0.2)',
                    font={'color': "#f8fafc"},
                    margin=dict(l=20, r=20, t=50, b=20),
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_lr_imp, width="stretch")
                
            # Confusion Matrix displays
            st.markdown("<br>##### 🔲 Confusion Matrices", unsafe_allow_html=True)
            
            cm_lr = np.array(lr_m['confusion_matrix'])
            cm_rf = np.array(rf_m['confusion_matrix'])
            
            col_cm1, col_cm2 = st.columns(2)
            
            with col_cm1:
                # Plotly heatmap for LR confusion matrix
                fig_cm_lr = px.imshow(
                    cm_lr,
                    labels=dict(x="Predicted Class", y="Actual Class"),
                    x=['Approved (No Default)', 'Rejected (Default)'],
                    y=['Actual Approved', 'Actual Defaulted'],
                    text_auto=True,
                    title='Logistic Regression Confusion Matrix',
                    color_continuous_scale='Blues'
                )
                fig_cm_lr.update_layout(
                    paper_bgcolor='rgba(30, 41, 59, 0.2)',
                    font={'color': "#f8fafc"},
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_cm_lr, width="stretch")
                
            with col_cm2:
                # Plotly heatmap for RF confusion matrix
                fig_cm_rf = px.imshow(
                    cm_rf,
                    labels=dict(x="Predicted Class", y="Actual Class"),
                    x=['Approved (No Default)', 'Rejected (Default)'],
                    y=['Actual Approved', 'Actual Defaulted'],
                    text_auto=True,
                    title='Random Forest Confusion Matrix',
                    color_continuous_scale='Greens'
                )
                fig_cm_rf.update_layout(
                    paper_bgcolor='rgba(30, 41, 59, 0.2)',
                    font={'color': "#f8fafc"},
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_cm_rf, width="stretch")
                
    # ----------------------------------------------------
    # TAB 4: PCA DETAIL
    # ----------------------------------------------------
    with tab_pca:
        st.subheader("Feature Dimensionality Reduction (PCA) Analysis")
        st.markdown("We perform Principal Component Analysis (PCA) to reduce 20+ scaled and one-hot encoded variables into orthogonal components. This helps us visualize clusters and discover the principal vectors of borrower variance.")
        
        if metrics and 'pca_explained_variance' in metrics:
            pca_var = metrics['pca_explained_variance']
            pca_loadings = pd.DataFrame(metrics['pca_loadings']).T
            
            col_var, col_load = st.columns([4, 5])
            
            with col_var:
                st.markdown("##### 📈 Explained Variance")
                st.markdown("Cumulative variance explains how much total borrower profile information is captured by the PCA projection.")
                
                # Construct dataframe for Plotly
                pcs = [f"PC{i+1}" for i in range(len(pca_var['individual']))]
                var_df = pd.DataFrame({
                    'Principal Component': pcs,
                    'Individual Variance': pca_var['individual'],
                    'Cumulative Variance': pca_var['cumulative']
                })
                
                # Plot explained variance
                fig_var = go.Figure()
                fig_var.add_trace(go.Bar(
                    x=var_df['Principal Component'],
                    y=var_df['Individual Variance'] * 100,
                    name='Individual Component',
                    marker_color='#3b82f6',
                    opacity=0.8
                ))
                fig_var.add_trace(go.Scatter(
                    x=var_df['Principal Component'],
                    y=var_df['Cumulative Variance'] * 100,
                    name='Cumulative Explained',
                    line=dict(color='#10b981', width=3, dash='solid'),
                    marker=dict(size=8, color='#10b981')
                ))
                fig_var.update_layout(
                    title='Explained Variance Ratio by Principal Components',
                    xaxis_title='Components',
                    yaxis_title='Percentage (%)',
                    paper_bgcolor='rgba(30, 41, 59, 0.2)',
                    plot_bgcolor='rgba(30, 41, 59, 0.2)',
                    font={'color': "#f8fafc"},
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_var, width="stretch")
                
            with col_load:
                st.markdown("##### 🔬 Feature Loadings (Weights)")
                st.markdown("Loadings represent the coefficients (contributions) of original variables to the principal components.")
                
                # Select component to view loadings
                pc_choice = st.selectbox("Select Component to Analyze Loadings:", ["PC1", "PC2", "PC3"])
                
                # Filter and prepare loadings
                pc_loading_data = pca_loadings[[pc_choice]].reset_index()
                pc_loading_data.columns = ['Feature', 'Weight']
                # Sort by absolute weight
                pc_loading_data['AbsWeight'] = pc_loading_data['Weight'].abs()
                pc_loading_data = pc_loading_data.sort_values(by='AbsWeight', ascending=False).head(15)
                
                # Plot loadings
                fig_loadings = px.bar(
                    pc_loading_data,
                    x='Weight',
                    y='Feature',
                    orientation='h',
                    color='Weight',
                    color_continuous_scale='Coolwarm',
                    title=f'Top 15 Feature Loading Coefficients for {pc_choice}',
                    labels={'Weight': 'Loading Weight Coefficient', 'Feature': 'Original Feature'}
                )
                fig_loadings.update_layout(
                    paper_bgcolor='rgba(30, 41, 59, 0.2)',
                    plot_bgcolor='rgba(30, 41, 59, 0.2)',
                    font={'color': "#f8fafc"},
                    margin=dict(l=20, r=20, t=50, b=20),
                    yaxis={'categoryorder': 'total ascending'},
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_loadings, width="stretch")

if __name__ == "__main__":
    main()
