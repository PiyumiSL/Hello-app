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

# Train the Random Forest model using the saved data
        st.write("Training the Random Forest model...")
        
        # Assuming the last column is the target variable
        target_column = data.columns[-1]
        X = data.drop(columns=[target_column])  # Features
        y = data[target_column]  # Target
        
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Random Forest model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Display model training completion
        st.success("Random Forest model trained successfully!")

        # Optionally, make predictions on the test set
        predictions = model.predict(X_test)
        st.write("Predictions on the test set:")
        st.write(predictions)
