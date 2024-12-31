import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
import io

# Set page configuration
st.set_page_config(page_title="Advanced EDA App", layout="wide")

# Title and description
st.title(" Advanced EDA App")
st.markdown("Upload your dataset to explore and analyze it with advanced visualization options.")

def hide_streamlit_style():
    hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

hide_streamlit_style()

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Step 1: Dataset Preview
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    # Dataset Info
    if st.checkbox("Show Dataset Info"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

    # Dataset Shape
    if st.checkbox("Show Dataset Shape"):
        st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    # Dataset Summary
    if st.checkbox("Show Summary (describe)"):
        st.write("### Summary Statistics:")
        st.write(df.describe())

    # Step 2: Display Categorical and Numerical Features
    st.write("### Categorical and Numerical Features")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    st.write(f"**Categorical Features:** {categorical_cols}")
    st.write(f"**Numerical Features:** {numerical_cols}")

    # Step 3: Preprocessing Options
    st.write("### Preprocessing Options")

    # Standardization
    standardize = st.checkbox("Standardize Numeric Features")
    if standardize:
        scaler = StandardScaler()
        if len(numerical_cols) > 0:
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            st.write("### Dataset after Standardization:")
            st.dataframe(df.head())
        else:
            st.warning("No numeric features to standardize.")

    # Encode Categorical Features
    encode_categorical = st.checkbox("Encode Categorical Features")
    if encode_categorical:
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
        st.write("### Dataset after Encoding Categorical Features:")
        st.dataframe(df.head())

    # Step 4: Handle Missing Values
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]
    if len(missing_columns) > 0:
        st.write("### Columns with Missing Values")
        st.write(missing_columns)
        fill_method = st.selectbox("Fill missing values using:", ["None", "Mean", "Median", "Custom Value"])
        if fill_method == "Mean":
            df.fillna(df.mean(), inplace=True)
        elif fill_method == "Median":
            df.fillna(df.median(), inplace=True)
        elif fill_method == "Custom Value":
            custom_value = st.text_input("Enter custom value:")
            if custom_value:
                df.fillna(custom_value, inplace=True)
        st.write("### Dataset after Handling Missing Values:")
        st.dataframe(df.head())

    # Step 5: Correlation Heatmap
    if st.checkbox("Show Correlation Heatmap"):
        if len(numerical_cols) > 1:
            st.write("### Correlation Heatmap:")
            corr_matrix = df[numerical_cols].corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            st.pyplot(plt)
        else:
            st.warning("Not enough numerical columns for a correlation heatmap.")

    # Step 6: Histograms
    if st.checkbox("Show Histograms"):
        st.write("### Histograms:")
        for col in numerical_cols:
            fig = px.histogram(df, x=col, nbins=30, title=f"Histogram for {col}")
            st.plotly_chart(fig)

# Footer
st.write("End of Advanced EDA App 🚀")
