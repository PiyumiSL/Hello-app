import streamlit as st

st.write('Hello world!')
st.title('Synergy prediction of potential drug candidates')

st.header('Upload your training data set here')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    # Display the content of the file (optional)
    import pandas as pd
    data = pd.read_csv(uploaded_file)
    st.write(data)

 # Submit Data button
    if st.button("Submit Data"):
        # Save the uploaded data to a CSV file for future work
        data.to_csv("uploaded_training_data.csv", index=False)
        st.success("Data saved successfully for future use!")

      # Button to train the Random Forest Model
        if st.button("Train Random Forest Model"):
            # Here you would include your model training code
            st.success("Random Forest Model training initiated!")  # Placeholder message

            if target_column not in data.columns:
                st.error(f"Column '{target_column}' not found in the uploaded data.")
            else:
                # Prepare data for training
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


# Upload test data set
st.header('Upload Your Test Data Set Here')
uploaded_test_file = st.file_uploader("Choose a test CSV file", type="csv", key="test")

if uploaded_test_file is not None:
    st.write("Test file uploaded successfully!")
    
    # Read the uploaded test CSV file
    test_data = pd.read_csv(uploaded_test_file)
    st.write(test_data)

    # Button to make predictions on the test data
    if st.button("Make Predictions on Test Data"):
        if 'model' in locals():
            # Ensure the model is defined
            test_predictions = model.predict(test_data)
            test_data['Predictions'] = test_predictions
            
            st.write("Predictions made successfully!")
            st.write(test_data)
        else:
            st.error("Model is not trained yet. Please train the model first.")
