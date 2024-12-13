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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the data from the uploaded CSV file
file_path = '/mnt/data/Data collection.csv'  # Adjust this if needed

# Initialize df as None
df = None

try:
    # Try to read the CSV file
    df = pd.read_csv(file_path)
    print(df.head())  # Print the first few rows of the DataFrame
except FileNotFoundError:
    print(f"File not found at: {file_path}")
except pd.errors.EmptyDataError:
    print("The file is empty.")
except pd.errors.ParserError:
    print("Error parsing the file.")
except Exception as e:
    print(f"An error occurred: {e}")

# Check if df is defined and not None
if df is not None:
    # Assuming 'df' is your DataFrame from the previous cell
    data = df

    # Separate the features (X) and the target (y)
    X = data[['Troponin Level (ng/mL)', 'ECG (0=Normal, 1=Abnormal)', 'Blood Pressure (mmHg)']]
    y = data['Heart Disease (1=Present, 0=Absent)']

    # Split the data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can adjust n_estimators

    # Train the model
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(classification_rep)
else:
    print("Dataframe 'df' could not be created. Please check the file.")
