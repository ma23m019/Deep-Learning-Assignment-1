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
    def __init__(self, input_size, hidden_layers, output_size, weight_type='random', activation_function='relu'):
        """
        Initializes the Feedforward Neural Network.

        Parameters:
        - input_size: Number of input features.
        - hidden_layers: List containing the number of neurons in each hidden layer.
        - output_size: Number of output neurons (e.g. number of classes for classification).
        - weight_type: Type of weight initialization ('random' or 'Xavier').
        - activation_function: Activation function to use ('relu', 'tanh', 'sigmoid').

        The weights and biases are initialized using the specified method.
        """        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weight_type = weight_type
        self.activation_function = activation_function
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
        else:
            raise ValueError("Unsupported activation function")
    
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
        else:
            raise ValueError("Unsupported activation function")
    
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
    
    def backward(self, X, y_true, learning_rate):
        """
        Performs backpropagation to update weights and biases using gradient descent.
        Gradients are computed for each layer and weights are adjusted accordingly.
        """    
        m = X.shape[0]
        dZ = self.activations[-1] - y_true  # Compute gradient for output layer
        dW = np.dot(self.activations[-2].T, dZ) / m
        dB = np.sum(dZ, axis=0, keepdims=True) / m
        
        self.weights[-1] -= learning_rate * dW  # Update weights for output layer
        self.biases[-1] -= learning_rate * dB

        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            dA = np.dot(dZ, self.weights[i + 1].T)  # Compute gradient for activation
            dZ = dA * self.activation_derivative(self.z_values[i])  # Apply activation derivative
            dW = np.dot(self.activations[i - 1].T, dZ) / m if i > 0 else np.dot(X.T, dZ) / m
            dB = np.sum(dZ, axis=0, keepdims=True) / m
            
            self.weights[i] -= learning_rate * dW  # Update weights
            self.biases[i] -= learning_rate * dB  # Update biases
    
    def train(self, X, y, epochs=100, learning_rate=0.01):
        """
        Trains the neural network using gradient descent.
        Logs the loss and accuracy at regular intervals.
        """      
        for epoch in range(epochs):
            y_pred = self.forward(X)  # Forward pass
            loss = self.compute_loss(y, y_pred)  # Compute loss
            self.backward(X, y, learning_rate)  # Backward pass
            accuracy = self.compute_accuracy(X, y)  # Compute accuracy
                        
            if epoch % 10 == 0:  # Print progress every 10 epochs
                print(f"Epoch {epoch}: Loss {loss:.4f}, Accuracy {accuracy:.4f}")
