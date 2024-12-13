import streamlit as st
st.write('Hello world!')
st.title('SYNERGY PREDICTION OF POTENTIAL DRUG CANDIDATES')
st.header('Upload your training data set here')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.write("File uploaded successfully!")
    # Display the content of the file (optional)
    import pandas as pd
    data = pd.read_csv(uploaded_file)
    st.write(data)

st.header('Upload your test data set here')
test_file = st.file_uploader("Choose a test CSV file", type="csv", key="test")

if test_file is not None:
    st.write("Test data uploaded successfully!")
    test_data = pd.read_csv(test_file)
    st.write(test_data)

if st.button("Run Random Forest Model"):
    if training_file is not None and test_file is not None:
        with st.spinner('Simulation Running...'):
            time.sleep(1)  # Simulate delay
            
            # Preprocessing the data
            st.write("Processing the training data...")
            X_train = training_data.iloc[:, :-1]  # Features (all columns except the last)
            y_train = training_data.iloc[:, -1]   # Target (last column)
            
            st.write("Processing the test data...")
            X_test = test_data  # Features (all columns in the test dataset)
            
            # Train the Random Forest model
            st.write("Training the model...")
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            
            # Make predictions on the test data
            st.write("Making predictions...")
            predictions = model.predict(X_test)
            
            # Display predictions
            st.success('Simulation Completed!')
            st.write("Predictions for the test data:")
            st.write(predictions)
    else:
        st.error("Please upload both training and test datasets to run the model.")
