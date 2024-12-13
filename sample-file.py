
import streamlit as st
import pandas as pd
import numpy as np
import random

# App title
st.title('Synergy Prediction of Potential Drug Candidates')

# Utility functions to create a simple Random Forest
def calculate_entropy(y):
    """Calculate entropy of a label array."""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Add small constant to avoid log(0)
    return entropy

def split_data(X, y, feature_index, threshold):
    """Split dataset based on a feature index and threshold."""
    left_indices = X[:, feature_index] <= threshold
    right_indices = ~left_indices
    return X[left_indices], y[left_indices], X[right_indices], y[right_indices]

def find_best_split(X, y):
    """Find the best feature and threshold to split the data."""
    best_gain = -1
    best_feature_index = None
    best_threshold = None
    current_entropy = calculate_entropy(y)

    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            _, y_left, _, y_right = split_data(X, y, feature_index, threshold)
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            left_weight = len(y_left) / len(y)
            right_weight = len(y_right) / len(y)
            gain = current_entropy - (left_weight * calculate_entropy(y_left) + right_weight * calculate_entropy(y_right))

            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = threshold

    return best_feature_index, best_threshold

class DecisionTree:
    """A simple decision tree for classification."""
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None
        self.feature_index = None  # Initialize feature_index and threshold
        self.threshold = None

    def fit(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            self.tree = np.bincount(y).argmax()  # Leaf node with majority class
            return

        feature_index, threshold = find_best_split(X, y)
        if feature_index is None:
            self.tree = np.bincount(y).argmax()  # Leaf node
            return

        self.feature_index = feature_index
        self.threshold = threshold

        X_left, y_left, X_right, y_right = split_data(X, y, feature_index, threshold)

        self.left = DecisionTree(self.max_depth)
        self.left.fit(X_left, y_left, depth + 1)

        self.right = DecisionTree(self.max_depth)
        self.right.fit(X_right, y_right, depth + 1)

    def predict(self, X):
        if isinstance(self.tree, int):
            return self.tree  # Leaf node

        # Check if feature_index and threshold are defined
        if self.feature_index is not None and self.threshold is not None:
            if X[self.feature_index] <= self.threshold:
                return self.left.predict(X)
            else:
                return self.right.predict(X)
        else:
            return self.tree  # Return the tree value if it's a leaf

class RandomForest:
    """A simple random forest implementation."""
    def __init__(self, n_trees, max_depth, sample_size):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), self.sample_size, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            tree = DecisionTree(self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(x) for tree in self.trees for x in X])  # Correctly iterate over trees and samples
        predictions = predictions.reshape(self.n_trees, len(X))
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_votes

# Section to upload the training data
st.header('Upload Your Training Data Set Here')
uploaded_training_file = st.file_uploader("Choose a CSV file for training", type="csv", key="train")

if uploaded_training_file is not None:
    st.write("Training file uploaded successfully!")
    training_data = pd.read_csv(uploaded_training_file)
    st.write(training_data)

    target_column = st.selectbox("Select the target column (output)", training_data.columns)
    if st.button("Train Random Forest Model"):
        if target_column not in training_data.columns:
            st.error(f"Target column '{target_column}' not found.")
        else:
            X = training_data.drop(columns=[target_column]).values
            y = training_data[target_column].values

            # Train the Random Forest
            model = RandomForest(n_trees=10, max_depth=3, sample_size=int(0.8 * len(X)))
            model.fit(X, y)
            st.success("Random Forest Model trained successfully!")

            # Save the trained model for predictions
            st.session_state["trained_model"] = model

# Section to upload the test data
st.header('Upload Your Test Data Set Here')
uploaded_test_file = st.file_uploader("Choose a CSV file for testing", type="csv", key="test")

if uploaded_test_file is not None:
    st.write("Test file uploaded successfully!")
    test_data = pd.read_csv(uploaded_test_file)
    st.write(test_data)

    if st.button("Make Predictions on Test Data"):
        if "trained_model" in st.session_state:
            model = st.session_state["trained_model"]
            X_test = test_data.values

            predictions = model.predict(X_test)
            test_data["Predictions"] = predictions
            st.write("Predictions made successfully!")
            st.write(test_data)
        else:
            st.error("Model is not trained yet. Please train the model first.")

# Function to compute accuracy
            def accuracy_score(y_true, y_pred):
            """Calculate the accuracy of predictions."""
            return np.mean(y_true == y_pred)

# Add this section after making predictions in your Streamlit app
            if st.button("Make Predictions on Test Data"):
            if "trained_model" in st.session_state:
            model = st.session_state["trained_model"]
            X_test = test_data.drop(columns=[target_column]).values

            predictions = model.predict(X_test)
            test_data["Predictions"] = predictions
            st.write("Predictions made successfully!")
            st.write(test_data)

        # Calculate accuracy
            acc = accuracy_score(test_data[target_column].values, predictions)
            st.write(f"Accuracy of the model: {acc:.2f}")

            else:
            st.error("Model is not trained yet. Please train the model first.")

