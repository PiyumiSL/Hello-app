st.write('Hello world!')
st.title('Synergy prediction of potential drug candidates')

# Initialize file variables
training_file = None
test_file = None

# Upload training data section
st.header('Upload your training data set here')
training_file = st.file_uploader("Choose a training CSV file", type="csv", key="training")

if training_file is not None:
    st.write("Training data uploaded successfully!")
    training_data = pd.read_csv(training_file)
    st.write("Training Data Preview:")
    st.write(training_data)
    
    # Ask user to select the target column
    target_column = st.selectbox("Select the target column (output labels):", training_data.columns)

# Upload test data section
st.header('Upload your test data set here')
test_file = st.file_uploader("Choose a test CSV file", type="csv", key="test")

if test_file is not None:
    st.write("Test data uploaded successfully!")
    test_data = pd.read_csv(test_file)
    st.write("Test Data Preview:")
    st.write(test_data)

# Button to run the model
if st.button("Run Random Forest Model"):
    if training_file is not None and test_file is not None:
        with st.spinner('Simulation Running...'):
            time.sleep(1)  # Simulate delay
            
            # Preprocessing the data
            st.write("Processing the training data...")
            X_train = training_data.drop(columns=[target_column])  # All columns except target
            y_train = training_data[target_column]                # Selected target column
            
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
