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
