import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# App title
st.title('Synergy Prediction of Potential Drug Candidates')

# Section to upload the training data
st.header('Upload Your Training Data Set Here')
uploaded_training_file = st.file_uploader("Choose a CSV file for training", type="csv", key="train")

if uploaded_training_file is not None:
    st.write("Training file uploaded successfully!")
    
    # Read and display the training data
    training_data = pd.read_csv(uploaded_training_file)
    st.write(training_data)

    # Ask the user to specify the target column
    target_column = st.selectbox("Select the target column (output)", training_data.columns)

    # Button to train the Random Forest Model
    if st.button("Train Random Forest Model"):
        # Validate if the target column exists in the dataset
        if target_column not in training_data.columns:
            st.error(f"Target column '{target_column}' not found in the uploaded training data.")
        else:
            # Prepare the data for training
            X = training_data.drop(columns=[target_column])  # Features
            y = training_data[target_column]  # Target

            # Split the data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and train the Random Forest model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions on the validation set
            val_predictions = model.predict(X_val)

            # Calculate the accuracy
            accuracy = accuracy_score(y_val, val_predictions)

            # Display the training results
            st.success(f"Random Forest Model trained successfully! Validation Accuracy: {accuracy:.2f}")

            # Save the trained model for later use
            st.session_state["trained_model"] = model

# Section to upload the test data
st.header('Upload Your Test Data Set Here')
uploaded_test_file = st.file_uploader("Choose a CSV file for testing", type="csv", key="test")

if uploaded_test_file is not None:
    st.write("Test file uploaded successfully!")
    
    # Read and display the test data
    test_data = pd.read_csv(uploaded_test_file)
    st.write(test_data)

    # Button to make predictions on the test data
    if st.button("Make Predictions on Test Data"):
        if "trained_model" in st.session_state:
            model = st.session_state["trained_model"]
            # Make predictions on the test data
            test_predictions = model.predict(test_data)

            # Add predictions as a new column in the test data
            test_data['Predictions'] = test_predictions

            # Display the predictions
            st.write("Predictions made successfully!")
            st.write(test_data)

            # Option to download the results
            csv = test_data.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
        else:
            st.error("Model is not trained yet. Please train the model first.")
