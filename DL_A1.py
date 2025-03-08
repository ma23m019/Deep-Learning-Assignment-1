######################################################## QUESTION 1 ######################################################## 
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Define class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot one sample image for each class
fig, axes = plt.subplots(5, 2, figsize=(10, 10))
axes = axes.flatten()

for i in range(10):
    # Find the first image in the training set for each class
    image_index = np.where(y_train == i)[0][0]
    image = x_train[image_index]

    # Plot the image
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(class_names[i])
    axes[i].axis('off')

plt.tight_layout()
plt.show()

######################################################## QUESTION 2 ######################################################## 

import numpy as np
import pandas as pd

class FeedForwardNN:
    def __init__(self, input_size, hidden_layers, output_size, weight_type='random', activation_function='relu', optimizer='sgd', learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, epochs = 10):
        """
        Initializes the Feedforward Neural Network.

        Parameters:
        - input_size: Number of input features.
        - hidden_layers: List containing the number of neurons in each hidden layer.
        - output_size: Number of output neurons (e.g. number of classes for classification).
        - weight_type: Type of weight initialization ('random' or 'Xavier').
        - activation_function: Activation function to use ('relu', 'tanh', 'sigmoid').
        - optimizer: Optimization algorithm for updating weights ('sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam')
        - learning_rate: Learning rate for weight updates (default: 0.01).
        - beta1: Momentum decay factor for Adam and Momentum optimizers (default: 0.9).
        - beta2: Squared gradient decay factor for RMSprop and Adam optimizers (default: 0.999).
        - epsilon: Small value to prevent division by zero in optimizers (default: 1e-8).
        - epochs: Number of training iterations (default: 10).

        Initializes:
        - Weights using the specified initialization method.
        - Biases as zero vectors for each layer.
        - Optimizer parameters (if applicable).
        """        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weight_type = weight_type
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.epochs = epochs
                
        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()
    
    def initialize_weights(self):
        """
        Initializes the weights of the neural network.
        If 'Xavier' initialization is selected, weights are scaled based on the number of neurons.
        Otherwise, small random values are used for initialization.
        """        
        weights = []
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            if self.weight_type == 'Xavier':
                weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) / np.sqrt(layer_sizes[i])
            else:  # Random initialization
                weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            weights.append(weight)
        return weights
    
    def initialize_biases(self):
        """
        Initializes biases as zero matrices for all layers.
        """        
        biases = [np.zeros((1, size)) for size in self.hidden_layers + [self.output_size]]
        return biases

    def initialize_optimizer_params(self):
        """
        Initializes optimization-related parameters for various optimizers.
        
        Initializes:
        - u_w: History terms for weight updates (used in Momentum and Nestrerov).
        - u_b: History terms for bias updates.
        - squared_grads_w: Squared gradients for weight updates (used in RMSprop, Adam and Nadam).
        - squared_grads_b: Squared gradients for bias updates.
        - m_w: First moment estimates for weights (used in Adam and Nadam).
        - m_b: First moment estimates for biases.
        - v_w: Second moment estimates for weights (used in Adam and Nadam).
        - v_b: Second moment estimates for biases.
        - t: Time step counter for Adam and Nadam optimizers.
    
        This method is called before training to ensure that all necessary optimizer parameters are initialized to zero.
        """
        self.u_w = [np.zeros_like(w) for w in self.weights]
        self.u_b = [np.zeros_like(b) for b in self.biases]
        self.squared_grads_w = [np.zeros_like(w) for w in self.weights]
        self.squared_grads_b = [np.zeros_like(b) for b in self.biases]
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0  # Time step for Adam and Nadam
        
    def activation(self, x):
        """
        Applies the selected activation function to the input.
        Supported activation functions:
        - ReLU (Rectified Linear Unit)
        - Tanh (Hyperbolic Tangent)
        - Sigmoid (Logistic function)
        """      
        if self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        elif self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
    
    def activation_derivative(self, x):
        """
        Computes the derivative of the activation function.
        """      
        if self.activation_function == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_function == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_function == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
    
    def softmax(self, x):
        """
        Computes the softmax activation function for multi-class classification.
        """      
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Performs forward propagation through the network.
        Stores activations and weighted sums for use in backpropagation.
        """      
        self.activations = []
        self.z_values = []
        
        a = X  # Input layer activation
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z) # Store weighted sum
            if i < len(self.weights) - 1:
                a = self.activation(z) # Apply activation function for hidden layers
            else:
                a = self.softmax(z) # Apply softmax for output layer
            self.activations.append(a) # Store activation
        
        return a
    
    def compute_loss(self, y_true, y_pred):
        """
        Computes cross-entropy loss for classification.
        Adds a small constant (1e-8) to prevent log(0) errors.
        """      
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    
    def compute_accuracy(self, X, y_true):
        """
        Computes the accuracy of the model on given data.
        Accuracy is calculated as the percentage of correct predictions.
        """    
        y_pred = self.forward(X)  # Get model predictions
        y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
        y_true_classes = np.argmax(y_true, axis=1)  # Convert one-hot encoded labels to class labels
        accuracy = np.mean(y_pred_classes == y_true_classes)  # Compute accuracy
        return accuracy

    def update_weights(self, dW, dB, layer_idx):
        """
        Updates the weights and biases of a specific layer using the chosen optimization algorithm.
    
        Parameters:
        - dW: Gradient of the loss with respect to the weights of the layer.
        - dB: Gradient of the loss with respect to the biases of the layer.
        - layer_idx: Index of the layer being updated.
    
        This method updates the model's weights and biases for the given layer using the selected optimizer.
        """
        
        if self.optimizer == 'sgd':
            self.weights[layer_idx] -= self.learning_rate * dW
            self.biases[layer_idx] -= self.learning_rate * dB
        
        elif self.optimizer == 'momentum':
            self.u_w[layer_idx] = self.beta1 * self.u_w[layer_idx] + dW
            self.u_b[layer_idx] = self.beta1 * self.u_b[layer_idx] + dB
            self.weights[layer_idx] -= self.learning_rate * self.u_w[layer_idx]
            self.biases[layer_idx] -= self.learning_rate * self.u_b[layer_idx]       
        
        elif self.optimizer == 'nesterov':
            # Lookahead step: Temporarily shift weights
            lookahead_weights = self.weights[layer_idx] - self.beta1 * self.u_w[layer_idx]
            lookahead_biases = self.biases[layer_idx] - self.beta1 * self.u_b[layer_idx]
    
            # Compute new velocity with gradients at lookahead position
            self.u_w[layer_idx] = self.beta1 * self.u_w[layer_idx] + (1 - self.beta1) * dW
            self.u_b[layer_idx] = self.beta1 * self.u_b[layer_idx] + (1 - self.beta1) * dB
    
            # Apply updated history
            self.weights[layer_idx] = lookahead_weights - self.learning_rate * self.u_w[layer_idx]
            self.biases[layer_idx] = lookahead_biases - self.learning_rate * self.u_b[layer_idx]
            
        elif self.optimizer == 'rmsprop':
            self.squared_grads_w[layer_idx] = self.beta2 * self.squared_grads_w[layer_idx] + (1 - self.beta2) * (dW ** 2)
            self.squared_grads_b[layer_idx] = self.beta2 * self.squared_grads_b[layer_idx] + (1 - self.beta2) * (dB ** 2)
            self.weights[layer_idx] -= self.learning_rate * dW / (np.sqrt(self.squared_grads_w[layer_idx]) + self.epsilon)
            self.biases[layer_idx] -= self.learning_rate * dB / (np.sqrt(self.squared_grads_b[layer_idx]) + self.epsilon)
        
        elif self.optimizer == 'adam':
            self.t += 1
            self.m_w[layer_idx] = self.beta1 * self.m_w[layer_idx] + (1 - self.beta1) * dW
            self.m_b[layer_idx] = self.beta1 * self.m_b[layer_idx] + (1 - self.beta1) * dB
            self.v_w[layer_idx] = self.beta2 * self.v_w[layer_idx] + (1 - self.beta2) * (dW ** 2)
            self.v_b[layer_idx] = self.beta2 * self.v_b[layer_idx] + (1 - self.beta2) * (dB ** 2)
            
            m_hat_w = self.m_w[layer_idx] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_b[layer_idx] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_w[layer_idx] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[layer_idx] / (1 - self.beta2 ** self.t)
            
            self.weights[layer_idx] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            self.biases[layer_idx] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        
        elif self.optimizer == 'nadam':
            self.t += 1
            self.m_w[layer_idx] = self.beta1 * self.m_w[layer_idx] + (1 - self.beta1) * dW
            self.m_b[layer_idx] = self.beta1 * self.m_b[layer_idx] + (1 - self.beta1) * dB
            self.v_w[layer_idx] = self.beta2 * self.v_w[layer_idx] + (1 - self.beta2) * (dW ** 2)
            self.v_b[layer_idx] = self.beta2 * self.v_b[layer_idx] + (1 - self.beta2) * (dB ** 2)
            
            m_hat_w = self.m_w[layer_idx] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m_b[layer_idx] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_w[layer_idx] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[layer_idx] / (1 - self.beta2 ** self.t)
        
            nadam_m_w = (self.beta1 * m_hat_w) + ((1 - self.beta1) * dW) / (1 - self.beta1 ** self.t)
            nadam_m_b = (self.beta1 * m_hat_b) + ((1 - self.beta1) * dB) / (1 - self.beta1 ** self.t)
        
            self.weights[layer_idx] -= self.learning_rate * nadam_m_w / (np.sqrt(v_hat_w) + self.epsilon)
            self.biases[layer_idx] -= self.learning_rate * nadam_m_b / (np.sqrt(v_hat_b) + self.epsilon)

        
    def backward(self, X, y_true):
        """
        Performs backpropagation to update weights and biases using gradient descent.
        Gradients are computed for each layer and weights are adjusted accordingly.
        """    
        m = X.shape[0]
        dZ = self.activations[-1] - y_true  # Compute gradient for output layer
            
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[i-1].T, dZ) / m if i > 0 else np.dot(X.T, dZ) / m
            dB = np.sum(dZ, axis=0, keepdims=True) / m
            self.update_weights(dW, dB, i)
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
                dZ = dA * self.activation_derivative(self.z_values[i-1])
    
    def train(self, X, y):
        """
        Trains the neural network using gradient descent.
        Logs the loss and accuracy at regular intervals.
        """      
        for epoch in range(epochs):
            y_pred = self.forward(X)  # Forward pass
            loss = self.compute_loss(y, y_pred)  # Compute loss
            self.backward(X, y)  # Backward pass
            accuracy = self.compute_accuracy(X, y)  # Compute accuracy
                        
            if epoch % 10 == 0:  # Print progress every 10 epochs
                print(f"Epoch {epoch}: Loss {loss:.4f}, Accuracy {accuracy:.4f}")
