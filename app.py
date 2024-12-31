import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, mean_squared_error, classification_report
import xgboost as xgb
import io

# Set page configuration
st.set_page_config(page_title="Advanced EDA App with Model Selection", layout="wide")

# Set up the Streamlit app with a title and description
st.title("🌲 Advanced EDA App with Model Selection 🌲")
st.markdown("Upload your dataset and perform detailed EDA with flexible modeling options, including Random Forest, Decision Trees, Logistic Regression, XGBoost, and more.")

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

    # Dataset Info
    if st.checkbox("Show Dataset Info"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

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

    # Automatically select the last column as the default target variable
    default_target_col = df.columns[-1]

    # Step 3: Ask if the user wants to perform Classification or Regression
    task_type = st.radio("What type of problem are you working on?", ("Classification", "Regression"))

    # Step 4: Target variable selection based on the task type, with default as the last column
    if task_type == "Classification":
        target_col_options = df.columns
    else:
        target_col_options = numerical_cols

    if len(target_col_options) > 0:
        target_col = st.selectbox("Select the target column", target_col_options, index=min(len(target_col_options)-1, 0))
    else:
        st.error("No suitable columns available to select as a target.")
        target_col = None

    # Step 5: Feature Selection
    st.write("### Select Features for Modeling")
    selected_features = st.multiselect("Select features to include in the model", df.columns.tolist(), default=df.columns[:-1].tolist())

    # Step 6: Preprocessing Options
    st.write("### Preprocessing Options")

    # Standardization
    standardize = st.checkbox("Standardize Numeric Features")
    if standardize:
        scaler = StandardScaler()
        numeric_cols = df[selected_features].select_dtypes(include=['int64', 'float64']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        st.write("### Updated Dataset after Standardization")
        st.dataframe(df.head())

    # Encode categorical features
    encode_categorical = st.checkbox("Encode Categorical Features")
    if encode_categorical:
        le = LabelEncoder()
        for col in df[selected_features].select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col])
        st.write("### Updated Dataset after Encoding Categorical Features")
        st.dataframe(df.head())

    # Step 7: Visualization Options
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

    # Step 8: Model Selection
    st.write("### Select the Model")
    
    if task_type == "Classification":
        model_choice = st.selectbox("Choose a Classification Model", 
                                    ("Random Forest", "Logistic Regression", "Decision Tree", "XGBoost"))
    else:
        model_choice = st.selectbox("Choose a Regression Model", 
                                    ("Random Forest", "Linear Regression", "Decision Tree", "XGBoost"))

    # Step 9: Hyperparameters based on model choice
    st.write("### Model Hyperparameters")

    if model_choice in ["Random Forest", "XGBoost", "Decision Tree"]:
        if model_choice in ["Random Forest", "XGBoost"]:
            n_estimators = st.slider("Number of trees (n_estimators)", 50, 500, 100)
        max_depth = st.slider("Max depth of trees (max_depth)", 5, 50, 10)

    # Step 10: Proceed with modeling
    if target_col:
        st.write(f"### Selected Target Column: {target_col}")

        # Automatically drop the target column from the features
        X = df[selected_features]
        y = df[target_col]

        # Handle categorical features in X
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.warning(f"Encoding categorical features: {categorical_cols}")
            le = LabelEncoder()
            for col in categorical_cols:
                X[col] = le.fit_transform(X[col])

        proceed = st.button(f"Proceed with {model_choice}")

        if proceed:
            # Spinner for long process
            with st.spinner(f'Training {model_choice} model...'):
                # Train the model based on the selected task and model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                if task_type == "Classification":
                    if model_choice == "Random Forest":
                        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                    elif model_choice == "Logistic Regression":
                        model = LogisticRegression(max_iter=500)
                    elif model_choice == "Decision Tree":
                        model = DecisionTreeClassifier(max_depth=max_depth)
                    elif model_choice == "XGBoost":
                        model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"### Model Accuracy: {accuracy:.2f}")
                    st.write("### Classification Report")
                    st.text(classification_report(y_test, y_pred))

                elif task_type == "Regression":
                    if model_choice == "Random Forest":
                        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                    elif model_choice == "Linear Regression":
                        model = LinearRegression()
                    elif model_choice == "Decision Tree":
                        model = DecisionTreeRegressor(max_depth=max_depth)
                    elif model_choice == "XGBoost":
                        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    st.write(f"### Model R-squared: {r2:.2f}")
                    st.write(f"### Mean Absolute Error: {mae:.2f}")
                    st.write(f"### Root Mean Squared Error: {rmse:.2f}")

# Footer with progress spinner
st.write("End of Advanced EDA App 🚀")
