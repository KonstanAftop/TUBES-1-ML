import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

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
                lower, upper = init_params.get('lower', 0.5), init_params.get('upper', 0.5)
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
            x = np.atleast_2d(x)  
            x_max = np.max(x, axis=1, keepdims=True)  
            x_stable = x - x_max  
            exp_x = np.exp(x_stable)
            softmax_output = exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-12)  

            return softmax_output.squeeze()  



    def loss(self, y_pred, y_true, loss_func="mse", derivative=False,regularization=None, lambda_reg=0.0):
        if loss_func == "mse":
            return np.mean((y_pred - y_true) ** 2) if not derivative else (y_pred - y_true)
        elif loss_func == "binary_cross_entropy":
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) if not derivative else (y_pred - y_true) / (y_pred * (1 - y_pred))
        elif loss_func == "categorical_cross_entropy":
            return -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1)) if not derivative else (y_pred - y_true)
        
        if regularization:
            reg_term = 0
            for W in self.weights:
                W_no_bias = W[1:, :] 
                if regularization == "l2":
                    reg_term += np.sum(W_no_bias ** 2)
                elif regularization == "l1":
                    reg_term += np.sum(np.abs(W_no_bias))
            loss_value += lambda_reg * reg_term

    def forward_propagation(self, X):
        A = X
        self.A_cache = [X]
        self.Z_cache = []

        for i in range(self.num_layers):
            A = np.hstack([np.ones((A.shape[0], 1)), A])
            Z = np.dot(A, self.weights[i])
            A = self.activation(Z, self.activations[i])
            self.Z_cache.append(Z)
            self.A_cache.append(A)

        return A

    def backward_propagation(self, y_true, loss_func, learning_rate, regularization=None, lambda_reg=0.0):
        m = y_true.shape[0]
        dA = self.loss(self.A_cache[-1], y_true, loss_func, derivative=True)

        self.gradients = []
        for i in reversed(range(self.num_layers)):
            dZ = dA * self.activation(self.Z_cache[i], self.activations[i], derivative=True)
            A_prev = np.hstack([np.ones((self.A_cache[i].shape[0], 1)), self.A_cache[i]])
            dW = np.dot(A_prev.T, dZ) / m

           
            if regularization == "l2":
                dW[1:, :] += lambda_reg * self.weights[i][1:, :]
            elif regularization == "l1":
                dW[1:, :] += lambda_reg * np.sign(self.weights[i][1:, :])

            dA = np.dot(dZ, self.weights[i][1:].T)
            self.weights[i] -= learning_rate * dW
            self.gradients.insert(0, dW)




    def train(self, X, y, batch_size=4, epochs=1000, learning_rate=0.01, loss_func="mse", X_val=None, y_val=None, verbose=True,regularization=None, lambda_reg=0.0):
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            pbar = tqdm(total=X.shape[0], desc=f"Epoch {epoch+1}/{epochs}", unit="inst")

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                y_pred = self.forward_propagation(X_batch)
                self.backward_propagation(y_batch, loss_func, learning_rate,regularization, lambda_reg)

               
                pbar.update(len(X_batch))

            pbar.close()

            train_loss = self.loss(self.forward_propagation(X), y, loss_func, regularization=regularization, lambda_reg=lambda_reg)
            history['train_loss'].append(train_loss)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self.loss(self.forward_propagation(X_val), y_val, loss_func)
                history['val_loss'].append(val_loss)

            if verbose:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.5f}, Val Loss = {val_loss:.5f}")
                else:
                    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.5f}")

        return history

    def infer(self, X):
        """
        Melakukan inferensi (prediksi) pada input X menggunakan model yang telah dilatih.
        
        Parameters:
            X (numpy array): Data input dengan bentuk (n_samples, n_features)

        Returns:
            numpy array: Hasil prediksi model
        """
        return self.forward_propagation(X)

    def plot_network(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        layer_x_positions = np.linspace(0, 1, len(self.layer_sizes))
        max_neurons = max(self.layer_sizes)
        
        for i, (x, num_neurons) in enumerate(zip(layer_x_positions, self.layer_sizes)):
            y_positions = np.linspace(0, 1, num_neurons) if num_neurons > 1 else [0.5]
            for y in y_positions:
                ax.scatter(x, y, s=300, color='blue', edgecolors='black')
                if i > 0:
                    prev_y_positions = np.linspace(0, 1, self.layer_sizes[i-1]) if self.layer_sizes[i-1] > 1 else [0.5]
                    for prev_y in prev_y_positions:
                        ax.plot([layer_x_positions[i-1], x], [prev_y, y], 'gray', lw=0.5)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title("Feedforward Neural Network Structure")
        plt.show()

    def plot_weight_distribution(self, layers, log_scale=False):
        for i in layers:
            plt.figure(figsize=(6, 4))
            plt.hist(self.weights[i].flatten(), bins=20, log=log_scale)
            plt.title(f"Weight Distribution - Layer {i}")
            plt.xlabel("Weight Values")
            plt.ylabel("Frequency")
            plt.show()

    def plot_gradient_distribution(self, layers, log_scale=False):
        for i in layers:
            plt.figure(figsize=(6, 4))
            plt.hist(self.gradients[i].flatten(), bins=20, log=log_scale)
            plt.title(f"Gradient Distribution - Layer {i}")
            plt.xlabel("Gradient Values")
            plt.ylabel("Frequency")
            plt.show()

    def save(self, filename="model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename="model.pkl"):
        with open(filename, "rb") as f:
            return pickle.load(f)