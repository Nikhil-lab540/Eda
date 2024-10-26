import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error, confusion_matrix, classification_report

# Set page configuration
st.set_page_config(page_title="Advanced EDA App with Random Forest", layout="wide")

# Set up the Streamlit app with a title and description
st.title("🌲 Advanced EDA App with Random Forest 🌲")
st.markdown("Upload your dataset and perform detailed EDA with flexible modeling options, including Random Forest classification and regression.")

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

# Upload the dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Step 1: Dataset Preview
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    # Automatically select the last column as the default target variable
    default_target_col = df.columns[-1]

    # Step 2: Ask if the user wants to perform Classification or Regression
    task_type = st.radio("What type of problem are you working on?", ("Classification", "Regression"))

    # Step 3: Target variable selection based on the task type, with default as the last column
    if task_type == "Classification":
        st.write("You selected **Classification**. Please choose the target (label) column for classification.")
        target_col = st.selectbox("Select the target column", df.columns, index=len(df.columns)-1)
    elif task_type == "Regression":
        st.write("You selected **Regression**. Please choose the target column for regression.")
        target_col = st.selectbox("Select the target column", df.select_dtypes(include=['int64', 'float64']).columns, index=len(df.columns)-1)

    # Step 4: Feature Selection
    st.write("### Select Features for Modeling")
    selected_features = st.multiselect("Select features to include in the model", df.columns.tolist(), default=df.columns[:-1].tolist())

    # Step 5: Preprocessing Options
    st.write("### Preprocessing Options")
    
    # Standardization
    standardize = st.checkbox("Standardize Numeric Features")
    if standardize:
        scaler = StandardScaler()
        df[selected_features] = scaler.fit_transform(df[selected_features])
        st.write("### Updated Dataset after Standardization")
        st.dataframe(df.head())

    # Encode categorical features
    encode_categorical = st.checkbox("Encode Categorical Features")
    if encode_categorical:
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col])
        st.write("### Updated Dataset after Encoding Categorical Features")
        st.dataframe(df.head())

    # Show missing value information before handling
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]
    if len(missing_columns) > 0:
        st.write("### Columns with Missing Values")
        st.write(missing_columns)

    # Fill missing values
    fill_missing_values = st.checkbox("Fill Missing Values")
    if fill_missing_values and len(missing_columns) > 0:
        fill_method = st.selectbox("Select fill method", ["Mean", "Median", "Custom Value"])
        if fill_method == "Mean":
            df.fillna(df.mean(), inplace=True)
        elif fill_method == "Median":
            df.fillna(df.median(), inplace=True)
        else:
            custom_value = st.number_input("Enter custom value")
            df.fillna(custom_value, inplace=True)
        st.write("### Updated Dataset after Filling Missing Values")
        st.dataframe(df.head())
    
    # Drop highly correlated features
    drop_corr_features = st.checkbox("Drop Highly Correlated Features")
    if drop_corr_features:
        corr_matrix = df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
        df.drop(to_drop, axis=1, inplace=True)
        st.write(f"Dropped highly correlated features: {to_drop}")
        st.write("### Updated Dataset after Dropping Highly Correlated Features")
        st.dataframe(df.head())

    # Step 6: Visualization Options
    st.write("### Visualization Options")
    
    # Correlation Heatmap
    show_corr_heatmap = st.checkbox("Show Correlation Heatmap")
    if show_corr_heatmap:
        st.write("### Correlation Heatmap:")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        st.pyplot(plt)

    # Histograms
    show_histograms = st.checkbox("Show Histograms")
    if show_histograms:
        st.write("### Histograms")
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            fig = px.histogram(df, x=col, nbins=30, title=f"Histogram for {col}")
            st.plotly_chart(fig)



    # Step 7: Model Hyperparameters
    st.write("### Random Forest Hyperparameters")
    n_estimators = st.slider("Number of trees (n_estimators)", 50, 500, 100)
    max_depth = st.slider("Max depth of trees (max_depth)", 5, 50, 10)

    # Step 8: Proceed with Random Forest modeling
    if target_col:
        st.write(f"### Selected Target Column: {target_col}")

        # Automatically drop the target column from the features
        X = df[selected_features]
        y = df[target_col]

        if task_type == "Classification" and len(df[target_col].unique()) <= 10:
            proceed = st.button("Proceed with Random Forest Classification")
        elif task_type == "Regression":
            proceed = st.button("Proceed with Random Forest Regression")

        if proceed:
            # Train a Random Forest model based on the task type
            if task_type == "Classification":
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"### Model Accuracy: {accuracy:.2f}")
                
                # Display classification metrics
                st.write("### Classification Report")
                st.text(classification_report(y_test, y_pred))
                

                
            elif task_type == "Regression":
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                st.write(f"### Model R-squared: {r2:.2f}")
                st.write(f"### Mean Absolute Error: {mae:.2f}")
                st.write(f"### Root Mean Squared Error: {rmse:.2f}")

            # Display feature importance
            feature_importance = pd.DataFrame({
                'Feature': selected_features,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            st.write("### Feature Importance")
            st.dataframe(feature_importance)



# Footer with progress spinner
st.write("End of Advanced EDA App 🚀")
