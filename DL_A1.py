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

######################################################## QUESTION 2, QUESTION 3 ######################################################## 

import numpy as np
import pandas as pd

class FeedForwardNN:
    def __init__(self, input_size, hidden_layers, output_size, weight_type='random', activation_function='relu', optimizer='sgd', 
                 learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, epochs = 10, batch_size=None):
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
        self.batch_size = batch_size
                
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
    

    def train(self, X, y, X_val, y_val):
        """
        Trains the neural network.
        Logs the loss and accuracy at each epoch.
        """
        m = X.shape[0]

        # If batch_size is not set, use full dataset size
        if self.batch_size is None:
            self.batch_size = m  # Full-batch training

        for epoch in range(self.epochs):
            # Shuffle data at the start of each epoch
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled, y_shuffled = X[indices], y[indices]

            for i in range(0, m, self.batch_size):  # Use self.batch_size
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                y_pred = self.forward(X_batch)  # Forward pass
                self.backward(X_batch, y_batch)  # Backpropagation

            # Compute training loss and accuracy on full dataset
            y_pred_full = self.forward(X)
            loss = self.compute_loss(y, y_pred_full)
            accuracy = self.compute_accuracy(X, y)
            wandb.log({"train_accuracy": accuracy})
            wandb.log({"train_loss": loss})

            # Compute loss and accuracy on validation dataset
            y_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, y_pred)
            val_accuracy = self.compute_accuracy(X_val, y_val)
            wandb.log({"val_accuracy": val_accuracy})
            wandb.log({"val_loss": val_loss})


######################################################## QUESTION 4 ######################################################## 
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist

# Load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Normalize images
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
X_val = X_val.reshape(X_val.shape[0], -1) / 255.0

# One-hot encoding labels
y_train = np.eye(10)[y_train]
y_val = np.eye(10)[y_val]
y_test = np.eye(10)[y_test]

# Define the hyperparameter sweep configuration
sweep_config = {
    'method': 'random',  # Random search for broad exploration  (or 'Bayesian')
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'epochs': {'values': [5, 10]},
        'hidden_layers': {'values': [3, 4, 5]},
        'layer_size': {'values': [32, 64, 128]},
        'l2_lambda': {'values': [0, 0.0005, 0.5]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_type': {'values': ['random', 'Xavier']},
        'activation': {'values': ['sigmoid', 'tanh', 'relu']}
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="Deep learning Assignment 1")

def sweep_train():
    # Initialize wandb run
    wandb.init(project="Deep learning Assignment 1", reinit=True)
    config = wandb.config

    # Initialize the network
    model = FeedForwardNN(
            input_size = 784,  # Flattened image size
            hidden_layers = [config.layer_size] * config.hidden_layers,
            output_size = 10,
            weight_type = config.weight_type,
            activation_function = config.activation,
            optimizer = config.optimizer,
            learning_rate = config.learning_rate,
            epochs = config.epochs,
            batch_size = config.batch_size,
            l2_lambda = config.l2_lambda
        )

    # Train the network
    model.initialize_optimizer_params()

    # Train model
    model.train(X_train, y_train, X_val, y_val)

    wandb.finish()

# Run the sweep
wandb.agent(sweep_id, sweep_train, count=20)  # Running 20 trials


######################################################## QUESTION 7 ######################################################## 

from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load Fashion MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Normalize images
X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test = X_test.reshape(X_test.shape[0], -1) / 255.0
X_val = X_val.reshape(X_val.shape[0], -1) / 255.0

# One-hot encoding labels
y_train = np.eye(10)[y_train]
y_val = np.eye(10)[y_val]
y_test = np.eye(10)[y_test]

model = FeedForwardNN(input_size = 784, hidden_layers = [128] * 4, output_size = 10, weight_type = 'Xavier',
                      activation_function = "relu", optimizer = "adam", learning_rate = 0.0001, epochs = 10, batch_size = 160, l2_lambda = 0)

# Train the network
model.initialize_optimizer_params()

# Train model
model.train(X_train, y_train, X_val, y_val)

def plot_confusion_matrix(model, X_test, y_test, class_names):
    # Get predictions
    y_pred_probs = model.forward(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

# compute test accuracy
print(model.compute_accuracy(X_test, y_test))

# plot the confusion matrix
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plot_confusion_matrix(model, X_test, y_test, class_names)


######################################################## QUESTION 10 ######################################################## 

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

X_train = x_train.reshape(x_train.shape[0], -1) / 255.0
X_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# One-hot encoding labels
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# CONFIG 1
model_1 = FeedForwardNN(input_size = 784, hidden_layers = [128] * 4, output_size = 10, weight_type = 'Xavier',
                      activation_function = "relu", optimizer = "adam", learning_rate = 0.0001, epochs = 10, batch_size = 160, l2_lambda = 0)

model_1.initialize_optimizer_params()
model_1.train(X_train, y_train, X_test, y_test)

print(model_1.compute_accuracy(X_test, y_test))

# CONFIG 2
model_2 = FeedForwardNN(input_size = 784, hidden_layers = [64] * 4, output_size = 10, weight_type = 'Xavier',
                      activation_function = "relu", optimizer = "nadam", learning_rate = 0.001, epochs = 10, batch_size = 32, l2_lambda = 0)
model_2.initialize_optimizer_params()
model_2.train(X_train, y_train, X_test, y_test)
print(model_2.compute_accuracy(X_test, y_test))

# CONFIG 3
model_3 = FeedForwardNN(input_size = 784, hidden_layers = [128] * 5, output_size = 10, weight_type = 'Xavier',
                      activation_function = "relu", optimizer = "nadam", learning_rate = 0.0001, epochs = 10, batch_size = 320, l2_lambda = 0)
model_3.initialize_optimizer_params()
model_3.train(X_train, y_train, X_test, y_test)
print(model_3.compute_accuracy(X_test, y_test))
