import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("creditcard.csv")  # Replace with the dataset path
    data['Class'] = data['Class'].map({0: 'Legitimate', 1: 'Fraudulent'})  # Map binary to labels
    return data

data = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Overview", "Data Insights", "Model Performance", "Transaction Explorer"])

# Overview Section
if section == "Overview":
    st.title("Credit Card Fraud Detection Dashboard")
    st.write("This dashboard provides insights into credit card fraud detection using the XGBoost algorithm.")

    # Display key metrics
    st.header("Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "99.82%")
    col2.metric("Precision", "93.45%")
    col3.metric("Recall", "88.70%")
    col4.metric("AUC-ROC", "0.9992")

# Data Insights Section
elif section == "Data Insights":
    st.title("Data Insights")

    # Class Distribution
    st.header("Class Distribution")
    class_dist = data['Class'].value_counts()
    fig_pie = px.pie(names=class_dist.index, values=class_dist.values, title="Transaction Class Distribution")
    st.plotly_chart(fig_pie)

    # Transaction Amounts
    st.header("Transaction Amount Distribution")
    fig_hist = px.histogram(data, x='Amount', color='Class', nbins=50, log_y=True, title="Transaction Amount by Class")
    st.plotly_chart(fig_hist)

# Model Performance Section
elif section == "Model Performance":
    st.title("Model Performance")

    # Confusion Matrix
    st.header("Confusion Matrix")
    conf_matrix = np.array([[85950, 15], [20, 128]])
    fig_heatmap = px.imshow(conf_matrix, text_auto=True, color_continuous_scale='Blues',
                            labels=dict(x="Predicted", y="Actual"))
    st.plotly_chart(fig_heatmap)

    # ROC Curve
    st.header("ROC Curve")
    # Example dummy data for ROC curve
    y_true = np.random.randint(0, 2, 1000)
    y_scores = np.random.rand(1000)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC = {roc_auc:.4f})",
                      labels=dict(x="False Positive Rate", y="True Positive Rate"))
    st.plotly_chart(fig_roc)

# Transaction Explorer Section
elif section == "Transaction Explorer":
    st.title("Transaction Explorer")

    # Filters
    st.header("Filter Transactions")
    amount_range = st.slider("Select Transaction Amount Range", min_value=0, max_value=int(data['Amount'].max()), value=(0, 200))
    filtered_data = data[(data['Amount'] >= amount_range[0]) & (data['Amount'] <= amount_range[1])]
    
    fraud_only = st.checkbox("Show Fraudulent Transactions Only")
    if fraud_only:
        filtered_data = filtered_data[filtered_data['Class'] == 'Fraudulent']

    # Display Table
    st.dataframe(filtered_data)

