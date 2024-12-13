import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.write('Hello world!')
st.title('Synergy Prediction of Potential Drug Candidates')

# Upload training data set
st.header('Upload Your Training Data Set Here')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    data = pd.read_csv(uploaded_file)
    st.write(data)

    # Submit Data button
    if st.button("Submit Data"):
        data.to_csv("uploaded_training_data.csv", index=False)
        st.success("Data saved successfully for future use!")

        # Button to train the Random Forest Model
        if st.button("Train Random Forest Model"):
            st.info("Training Random Forest Model...")
            
            target_column = st.text_input("Enter the target column name", value="target")

            if target_column not in data.columns:
                st.error(f"Column '{target_column}' not found in the uploaded data.")
            else:
                # Prepare the data for training
                X = data.drop(columns=[target_column])  # Features
                y = data[target_column]  # Target variable
                
                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Initialize the Random Forest model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                # Train the model
                model.fit(X_train, y_train)

                # Make predictions on the test set
                predictions = model.predict(X_test)
                
                # Calculate the accuracy
                accuracy = accuracy_score(y_test, predictions)
                
                # Display the accuracy
                st.success(f"Random Forest Model trained successfully! Accuracy: {accuracy:.2f}")

                # Save the trained model
                st.session_state['model'] = model
                st.session_state['features'] = X.columns.tolist()

# Upload test data set
st.header('Upload Your Test Data Set Here')
uploaded_test_file = st.file_uploader("Choose a test CSV file", type="csv", key="test")

if uploaded_test_file is not None:
    st.write("Test file uploaded successfully!")
    test_data = pd.read_csv(uploaded_test_file)
    st.write(test_data)

    # Button to make predictions on the test data
    if st.button("Make Predictions on Test Data"):
        if 'model' in st.session_state:
            model = st.session_state['model']
            features = st.session_state['features']
            
            if not all(feature in test_data.columns for feature in features):
                st.error("Test data does not contain the required features.")
            else:
                test_data_subset = test_data[features]
                test_predictions = model.predict(test_data_subset)
                test_data['Predictions'] = test_predictions
                
                st.write("Predictions made successfully!")
                st.write(test_data)
        else:
            st.error("Model is not trained yet. Please train the model first.")
