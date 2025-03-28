import numpy as np

class FFNN:
    def __init__(self, layer_sizes, activations, weight_init='zero', init_params=None, seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.num_layers = len(layer_sizes) - 1
        self.weights = []

        if init_params is None:
            init_params = {}

        for i in range(self.num_layers):
            input_size, output_size = layer_sizes[i], layer_sizes[i+1]
            if weight_init == 'zero':
                W = np.zeros((input_size + 1, output_size))
            elif weight_init == 'random_uniform':
                lower, upper = init_params.get('lower', -0.5), init_params.get('upper', 0.5)
                W = np.random.uniform(lower, upper, (input_size + 1, output_size))
            elif weight_init == 'random_normal':
                mean, var = init_params.get('mean', 0), init_params.get('variance', 0.01)
                W = np.random.normal(mean, np.sqrt(var), (input_size + 1, output_size))
            self.weights.append(W)

    def activation(self, x, func, derivative=False):
        if func == 'linear':
            return x if not derivative else np.ones_like(x)
        elif func == 'relu':
            return np.maximum(0, x) if not derivative else (x > 0).astype(float)
        elif func == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig if not derivative else sig * (1 - sig)
        elif func == 'tanh':
            tanh_x = np.tanh(x)
            return tanh_x if not derivative else 1 - tanh_x**2
        elif func == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def loss(self, y_pred, y_true, loss_func="mse", derivative=False):
        if loss_func == "mse":
            return np.mean((y_pred - y_true)**2) if not derivative else (y_pred - y_true)
        elif loss_func == "binary_crossentropy":
            return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)) \
                if not derivative else (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-8)
        elif loss_func == "categorical_crossentropy":
            return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0] \
                if not derivative else y_pred - y_true

    def forward_propagation(self, X):
        A = X
        self.A_cache = [X]  
        self.Z_cache = []  

        for i in range(self.num_layers):
            A = np.hstack([np.ones((A.shape[0], 1)), A])  # Tambahkan bias
            Z = np.dot(A, self.weights[i])
            A = self.activation(Z, self.activations[i])
            self.Z_cache.append(Z)
            self.A_cache.append(A)

        return A  

    def backward_propagation(self, y_true, loss_func, learning_rate):
        m = y_true.shape[0]
        dA = self.loss(self.A_cache[-1], y_true, loss_func, derivative=True)

        for i in reversed(range(self.num_layers)):
            dZ = dA * self.activation(self.Z_cache[i], self.activations[i], derivative=True)
            dW = np.dot(np.hstack([np.ones((self.A_cache[i].shape[0], 1)), self.A_cache[i]]).T, dZ) / m
            dA = np.dot(dZ, self.weights[i][1:].T)  

            self.weights[i] -= learning_rate * dW  

    def train(self, X, y, epochs=1000, learning_rate=0.01, loss_func="mse"):
        history = []

        for epoch in range(epochs):
            y_pred = self.forward_propagation(X)
            loss_value = self.loss(y_pred, y, loss_func)
            self.backward_propagation(y, loss_func, learning_rate)

            history.append(loss_value)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss_value:.5f}")

        return history
