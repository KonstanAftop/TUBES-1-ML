import numpy as np

class FFNN:
    def __init__(self, layer_sizes, activations, weight_init='zero', init_params=None, seed=42):
        np.random.seed(seed)
        self.layer_sizes=layer_sizes
        self.activations=activations
        self.num_layers=len(layer_sizes)-1
        self.weights=[]
        
        if init_params is None:
            init_params = {}

        for i in range(self.num_layers):
            input_size, output_size = layer_sizes[i], layer_sizes[i+1]
            if weight_init=='zero':
                W=np.zeros((input_size+1, output_size))
            elif weight_init=='random_uniform':
                lower,upper =init_params.get('lower', -0.5), init_params.get('upper', 0.5)
                W=np.random.uniform(lower,upper,(input_size+1, output_size))
            elif weight_init=='random_normal':
                mean, var=init_params.get('mean' , 0) , init_params.get('variance', 0.01)
                W=np.random.normal(mean, np.sqrt(var), (input_size+1, output_size))
            self.weights.append(W)
            
    def activation(self, x, func):
        if func == 'linear' :
            return x
        elif func == 'relu':
            return np.maximum(0,x)
        elif func == 'sigmoid' :
            return 1/(1+np.exp(-x))
        elif func == 'tanh':
            return np.tanh(x)
        elif func == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_propagation(self,X):
        A = X
        for i in range(self.num_layers):
            A = np.hstack([A, np.ones((A.shape[0], 1))]) 
            Z = np.dot(A, self.weights[i])  
            A = self.activation(Z, self.activations[i])
        return A
    
