class LogisticRegression:
    def __init__(self, learning_rate=5e-4, reg_lambda=1e-4, batch_size=200, epochs=100):
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.batch_size = batch_size
        self.epochs = epochs

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def initialize_weights(self, input_size, num_classes):
        self.weights = np.random.normal(0, 1, (input_size, num_classes))
        self.bias = np.zeros((1, num_classes))

    def train(self, X_train, y_train):
        num_samples, num_features = X_train.shape
        num_classes = y_train.shape[1]
        self.initialize_weights(num_features, num_classes)

        for epoch in range(self.epochs):
            for i in range(0, num_samples, self.batch_size):
                X_batch = X_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]

                logits = np.dot(X_batch, self.weights) + self.bias
                y_pred = self.softmax(logits)
                loss = self.cross_entropy_loss(y_batch, y_pred)

                grad_weights = np.dot(X_batch.T, (y_pred - y_batch)) / X_batch.shape[0]
                grad_bias = np.sum(y_pred - y_batch, axis=0, keepdims=True) / X_batch.shape[0]
                self.weights -= self.learning_rate * (grad_weights + self.reg_lambda * self.weights)
                self.bias -= self.learning_rate * grad_bias

            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        logits = np.dot(X, self.weights) + self.bias
        return np.argmax(self.softmax(logits), axis=1)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Usage:
# Assuming read_dataset function is used to load the MNIST binary files.
dataset_path = "data/mnist"
X_train, y_train, X_test, y_test = read_dataset(dataset_path)

# Train logistic regression
log_reg = LogisticRegression()
log_reg.train(X_train, y_train)
log_reg.evaluate(X_test, y_test)
