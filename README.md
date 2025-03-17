# Deep-Learning-Assignment-1

This repository contains the code for Deep Learning Assignment 1, which involves implementing a feedforward neural network from scratch and experimenting with various optimization techniques, activation functions, and hyperparameters. The code is organized into different sections corresponding to the questions in the assignment.

## Links

- **Wandb Report**: [View the Wandb Report](https://wandb.ai/ma23m019-indian-institute-of-technology-madras/Deep%20learning%20Assignment%201/reports/Deep-Learning-Assignment-1--VmlldzoxMTY0OTMzNg?accessToken=ukz6nyb4k5dk4autycnq6yuse2kq9sbf8lnedfinvz0cce18g1e9lwbppz0ivqwi)
- **GitHub Repository**: [GitHub Repo](https://github.com/ma23m019/Deep-Learning-Assignment-1)

## Code Organization

The code is organized into several sections, each corresponding to a specific question in the assignment:

### Question 1: Data Visualization
- **Objective**: Visualize samples from the Fashion-MNIST dataset.
- **Code**: The code loads the Fashion-MNIST dataset, plots one sample image for each class, and logs the images to Wandb.

### Questions 2 & 3: Feedforward Neural Network Implementation
- **Objective**: Implement a feedforward neural network from scratch.
- **Code**: The `FeedForwardNN` class is defined, which includes methods for forward propagation, backward propagation, weight initialization, and training. The class supports various activation functions (`relu`, `tanh`, `sigmoid`) and optimizers (`sgd`, `momentum`, `nesterov`, `rmsprop`, `adam`, `nadam`).

### Question 4: Hyperparameter Tuning with Wandb Sweeps
- **Objective**: Perform hyperparameter tuning using Wandb sweeps.
- **Code**: A hyperparameter sweep configuration is defined, and the `sweep_train` function is used to train the model with different combinations of hyperparameters. The results are logged to Wandb.

### Question 7: Confusion Matrix and Test Accuracy
- **Objective**: Evaluate the trained model on the test set and plot the confusion matrix.
- **Code**: The model is trained on the Fashion-MNIST dataset, and the test accuracy is computed. A confusion matrix is plotted to visualize the model's performance.

### Question 10: MNIST Dataset Experimentation
- **Objective**: Experiment with different configurations of the feedforward neural network on the MNIST dataset.
- **Code**: Three different configurations of the `FeedForwardNN` are trained on the MNIST dataset, and their test accuracies are compared.
