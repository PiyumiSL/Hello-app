import streamlit as st
import pandas as pd
import numpy as np

# App title
st.title('Predicting the Presence of Heart Diseases')

# Utility functions
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
        self.feature_index = None
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

        if self.feature_index is not None and self.threshold is not None:
            if X[self.feature_index] <= self.threshold:
                return self.left.predict(X)
            else:
                return self.right.predict(X)
        else:
            return self.tree

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
        predictions = np.array([tree.predict(x) for tree in self.trees for x in X])
        predictions = predictions.reshape(self.n_trees, len(X))
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_votes

class SupportVectorMachine:
    """A simple Support Vector Machine implementation."""
    def __init__(self, learning_rate=0.01, n_iters=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weights) - self.bias
        return np.sign(approx)

class ANNModel:
    """Simple Artificial Neural Network (ANN) with one hidden layer."""
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=100):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        # Encode y to zero-based index if necessary
        unique_classes = np.unique(y)
        y_encoded = np.searchsorted(unique_classes, y)

        # One-hot encoding of y
        y_one_hot = np.zeros((y_encoded.size, self.output_size))
        y_one_hot[np.arange(y_encoded.size), y_encoded] = 1

        for _ in range(self.epochs):
            # Forward pass
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_layer_output = self.sigmoid(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
            output_layer_output = self.sigmoid(output_layer_input)

            # Backward pass
            output_error = y_one_hot - output_layer_output
            output_delta = output_error * self.sigmoid_derivative(output_layer_output)

            hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta = hidden_error * self.sigmoid_derivative(hidden_layer_output)

            # Update weights and biases
            self.weights_hidden_output += np.dot(hidden_layer_output.T, output_delta) * self.learning_rate
            self.weights_input_hidden += np.dot(X.T, hidden_delta) * self.learning_rate
            self.bias_output += np.sum(output_delta, axis=0) * self.learning_rate
            self.bias_hidden += np.sum(hidden_delta, axis=0) * self.learning_rate

    def predict(self, X):
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_output
        output_layer_output = self.sigmoid(output_layer_input)

        return np.argmax(output_layer_output, axis=1)

# Section to upload the training data
st.header('Upload Your Training Data Set Here')
uploaded_training_file = st.file_uploader("Choose a CSV file for training", type="csv", key="train")

if uploaded_training_file is not None:
    st.write("Training file uploaded successfully!")
    training_data = pd.read_csv(uploaded_training_file)
    st.write(training_data)

    target_column = st.selectbox("Select the target column (output)", training_data.columns)
    model_type = st.selectbox("Select the model to train", ["Random Forest(RF)", "Support Vector Machine(SVM)","Artificial Neural Network(ANN)"])

if st.button("Train Model"):  # Ensure the colon is present
    # Perform the training process
    st.write("Training model...")

    if target_column not in training_data.columns:
        st.error(f"Target column '{target_column}' not found.")
    else:
        X = training_data.drop(columns=[target_column]).values
        y = training_data[target_column].values

        if model_type == "Random Forest(RF)":
            model = RandomForest(n_trees=10, max_depth=5, sample_size=int(len(X) * 0.8))
            model.fit(X, y)
            st.success("Random Forest model trained successfully!")

        elif model_type == "Support Vector Machine(SVM)":
            model = SupportVectorMachine()
            model.fit(X, y)
            st.success("SVM model trained successfully!")

        elif model_type == "Artificial Neural Network(ANN)":
            # Assuming the output size corresponds to the number of unique classes in y
            output_size = len(np.unique(y))
            model = ANNModel(input_size=X.shape[1], hidden_size=5, output_size=output_size)
            model.fit(X, y)
            st.success("ANN model trained successfully!")
