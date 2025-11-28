import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
import joblib
import os

st.set_page_config(page_title="Credit Risk AI Dashboard", layout="wide", page_icon="🏦")


@st.cache_resource
def load_saved_project():
    try:
        data = joblib.load('aibis_projekt_daten.pkl')
        return data['model'], data['X_test'], data['y_test'], data['X_full']
    except FileNotFoundError:
        return None, None, None, None


@st.cache_resource
def load_cluster_brain():
    try:
        return joblib.load('cluster_brain.pkl')
    except FileNotFoundError:
        return None

model, X_test, y_test, X_full = load_saved_project()
cluster_brain = load_cluster_brain()

# Initialize SHAP Explainer for Risk (cached)
explainer = shap.TreeExplainer(model)

# --- SIDEBAR ---
st.sidebar.title("Passau Finance AI")

st.sidebar.markdown("---")

page = st.sidebar.radio("Select View:", 
                        ["1. Executive Overview", 
                         "2. Customer Segmentation", 
                         "3. Loan Officer Interface"])

st.sidebar.markdown("---")
if page == "2. Customer Segmentation":
    st.sidebar.info("Model: **K-Means Clustering**")
    st.sidebar.info("Data: **PCA Reduced**")
else:
    st.sidebar.info("Model: **Random Forest**")
    st.sidebar.info("Threshold: **0.30**")

# ==========================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==========================================
if page == "1. Executive Overview":
    st.title("📊 Strategic Risk Portfolio Overview")
    st.markdown("Performance monitoring of the deployed Credit Risk AI Model.")

    # 1. KPIs
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.3).astype(int)
    
    cm = confusion_matrix(y_test, y_pred)
    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Recall (Risk Detection)", f"{recall:.1%}", "Target: >80%")
    col2.metric("Prevented Defaults", f"{cm[1, 1]} / {cm[1, 1]+cm[1, 0]}", "Loans identified")
    col3.metric("Model Accuracy", f"{accuracy:.1%}", "Trade-off for safety")
    
    st.markdown("---")

    # 2. Plots
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Confusion Matrix")
        st.write("Visualizing True Positives vs. False Alarms.")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                   xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(fig_cm)

    with col_right:
        st.subheader("ROC Curve")
        st.write("Trade-off Analysis (Precision vs. Recall).")
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
        ax_roc.plot([0, 1], [0, 1], color='navy', linestyle='--')
        
        idx_03 = (np.abs(thresholds - 0.3)).argmin()
        ax_roc.scatter(fpr[idx_03], tpr[idx_03], marker='o', color='red', s=100, label='Selected Threshold 0.3')
        
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

    # 3. Global SHAP
    st.subheader("Global Feature Importance (XAI)")
    st.write("Which factors drive the model's decisions globally?")
    
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values_risk = shap_values[1]
    else:
        shap_values_risk = shap_values[:, :, 1]
        
    fig_shap, ax_shap = plt.subplots()
    shap.summary_plot(shap_values_risk, X_test, plot_type="bar", show=False)
    st.pyplot(fig_shap)

# ==========================================
# PAGE 2: CUSTOMER SEGMENTATION 
# ==========================================
elif page == "2. Customer Segmentation":
    st.title("🧩 Customer Market Segments")
    st.markdown("Unsupervised Analysis of the customer base using **K-Means**.")
    
    if cluster_brain is None:
        st.error("⚠️ 'cluster_brain.pkl' not found.")
        st.info("Please run the 'colleague_clustering.ipynb' notebook to generate the file.")
    else:
        # Daten entpacken
        plot_df = cluster_brain['pca_data']
        shap_values_global = cluster_brain['shap_values_global']
        raw_features = cluster_brain['raw_features']
        
        t1, t2 = st.tabs(["Cluster Map (PCA)", "Cluster Meaning (SHAP)"])
        
        with t1:
            st.subheader("Customer Segments Map")
            st.markdown("Projection of customers into 2D space.")
            
            fig_cluster = plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=plot_df, x="PC1", y="PC2", 
                hue="Cluster", palette="viridis", s=100, alpha=0.8
            )
            plt.title("Identified Customer Groups")
            plt.legend(title="Cluster ID")
            st.pyplot(fig_cluster)
            
            st.info("""
            **Interpretation:**
            * **Cluster 0:** Standard Customers
            * **Cluster 1:** High Value / Investors
            * **Cluster 2:** Savers (Conservative)
            """)
            
        with t2:
            st.subheader("What defines the clusters?")
            st.markdown("Global SHAP analysis for the selected cluster.")
            
            # WICHTIG: Sicherstellen, dass Cluster als int vorliegen für die Auswahl
            unique_clusters = sorted(plot_df['Cluster'].unique())
            cluster_id_view = st.selectbox("Select Cluster to Analyze:", unique_clusters)
            
            fig_s, ax = plt.subplots()
            # Wir nutzen die gespeicherte Liste von SHAP-Werten
            # Falls shap_values_global eine Liste ist, nehmen wir das Element am Index
            if isinstance(shap_values_global, list):
                sv_plot = shap_values_global[cluster_id_view]
            else:
                # Fallback falls es doch ein Array ist
                sv_plot = shap_values_global[:, :, cluster_id_view]
                
            shap.summary_plot(sv_plot, features=raw_features, show=False)
            st.pyplot(fig_s)

# ==========================================
# PAGE 3: LOAN OFFICER INTERFACE
# ==========================================
elif page == "3. Loan Officer Interface":
    st.title("🕵️‍♂️ Loan Application Assessment")
    st.markdown("AI-supported decision making for individual loan applications.")

    # Select Customer
    st.info("Select a test case below to simulate a real-time prediction.")
    customer_id = st.selectbox("Select Applicant ID:", options=range(len(X_test)))
    
    # Get Data
    customer_data = X_test.iloc[[customer_id]]
    
    # Live Prediction
    risk_prob = model.predict_proba(customer_data)[0][1]
    threshold = 0.3
    is_risk = risk_prob >= threshold
    
    st.markdown("### AI Recommendation")
    
    c1, c2 = st.columns(2)
    with c1:
        if is_risk:
            st.error(f"🔴 **REJECT APPLICATION**")
            st.markdown(f"High risk of default detected.")
        else:
            st.success(f"🟢 **APPROVE APPLICATION**")
            st.markdown(f"Applicant meets creditworthiness criteria.")
            
    with c2:
        st.metric("Probability of Default (PD)", f"{risk_prob:.1%}", f"Threshold: 30%")

    st.markdown("---")
    
    # Explainability
    st.subheader("Why did the AI decide this?")
    st.write("Local explanation of positive (red) and negative (blue) factors:")
    
    # Prepare Waterfall Plot
    shap_values_single = explainer.shap_values(customer_data)
    if isinstance(shap_values_single, list):
        sv = shap_values_single[1][0]
        ev = explainer.expected_value[1]
    else:
        sv = shap_values_single[0, :, 1]
        ev = explainer.expected_value[1]
        
    fig_waterfall = plt.figure(figsize=(8, 4))
    shap_object = shap.Explanation(values=sv, 
                                   base_values=ev, 
                                   data=customer_data.iloc[0], 
                                   feature_names=X_full.columns)
    
    shap.plots.waterfall(shap_object, show=False)
    st.pyplot(fig_waterfall, bbox_inches='tight')
    
    # Show Raw Data
    with st.expander("View Applicant's Raw Data"):
        st.dataframe(customer_data.T)
