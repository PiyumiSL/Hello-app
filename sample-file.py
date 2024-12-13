import streamlit as st
import pandas as pd

st.title('Simple Classification Model: Heart Disease Prediction')

# Upload training dataset
st.header('Upload Your Training Data')
training_file = st.file_uploader("Choose a training CSV file", type="csv", key="train")

def train_model(data):
    # Simple logic-based model: classify based on Troponin Level and ECG
    def simple_classifier(row):
        if row['Troponin Level (ng/mL)'] > 0.1 or row['ECG (0=Normal, 1=Abnormal)'] == 1:
            return 1  # Predict Heart Disease
        return 0  # Predict No Heart Disease

    return simple_classifier

if training_file is not None:
    st.write("Training file uploaded successfully!")
    training_data = pd.read_csv(training_file)
    st.write(training_data)

    # Button to train the model
    if st.button("Train Model"):
        # Train a simple model
        classifier = train_model(training_data)
        st.success("Model trained successfully!")

        # Save the classifier for later use
        st.session_state['classifier'] = classifier

# Upload test dataset
st.header('Upload Your Test Data')
test_file = st.file_uploader("Choose a test CSV file", type="csv", key="test")

if test_file is not None:
    st.write("Test file uploaded successfully!")
    test_data = pd.read_csv(test_file)
    st.write(test_data)

    # Button to make predictions
    if st.button("Make Predictions"):
        if 'classifier' in st.session_state:
            classifier = st.session_state['classifier']

            # Apply the model to the test data
            test_data['Predictions'] = test_data.apply(classifier, axis=1)
            st.write("Predictions made successfully!")
            st.write(test_data)

            # Option to download the results
            csv = test_data.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
        else:
            st.error("Model is not trained yet. Please upload training data and train the model first.")
