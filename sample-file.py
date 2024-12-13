import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# App Title
st.title('Synergy Prediction of Potential Drug Candidates')

# Upload training and testing datasets
st.header("Upload Your Datasets")
train_file = st.file_uploader("Upload Training Dataset (CSV)", type=["csv"])
test_file = st.file_uploader("Upload Testing Dataset (CSV)", type=["csv"])

# Proceed if both files are uploaded
if train_file is not None and test_file is not None:
    # Read datasets
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    st.write("Training Dataset")
    st.dataframe(train_data.head())
    
    st.write("Testing Dataset")
    st.dataframe(test_data.head())
    
    # Assuming the last column is the target variable
    target_column = st.selectbox(
        "Select the Target Column (for classification):", train_data.columns
    )
    
    if st.button("Train and Test Model"):
        # Splitting features and target
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]
        
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        
        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Test model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Display results
        st.write("Model Accuracy:", accuracy)
        st.write("Predictions on Test Data")
        st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": predictions}))

# Add Instructions
else:
    st.info("Please upload both training and testing datasets to proceed.")
