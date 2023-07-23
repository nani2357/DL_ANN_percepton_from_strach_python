import numpy as np
import os
import joblib


class Perceptron:
    def __init__(self, eta: float=None, epochs: int=None):
        # Initialize the student's knowledge (weights) with small random numbers
        # We use a 3-dimensional vector because we have two inputs and one bias
        self.weights = np.random.randn(3) * 1e-4 
        training = (eta is not None) and (epochs is not None)
        if training:
            print(f"initial weights before training: \n{self.weights}\n")
        # Set the student's learning speed (eta) and the number of times they'll review the textbook (epochs)
        self.eta = eta
        self.epochs = epochs
    
    def _z_outcome(self, inputs, weights):
        # The student makes a raw guess based on their current knowledge and the object they're looking at
        # This is done by taking the dot product of the inputs and the weights
        return np.dot(inputs, weights)
    
    def activation_function(self, z):
        # The student finalizes their guess
        # If the raw guess (z) is greater than 0, they guess 1; otherwise, they guess 0
        return np.where(z > 0, 1, 0)
    
    def fit(self, X, y):
        # The student starts a study session
        # X is the set of examples and y is the correct labels for those examples
        self.X = X
        self.y = y
        
        # Add a bias term to the inputs
        # This is like adding a constant term to a linear equation to allow it to fit the data better
        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
        print(f"X with bias: \n{X_with_bias}")
        
        for epoch in range(self.epochs):
            print("--"*10)
            print(f"for epoch >> {epoch}")
            print("--"*10)
            
            # The student makes guesses for all the examples in the textbook
            # They do this by calculating the z outcome and then applying the activation function
            z = self._z_outcome(X_with_bias, self.weights)
            y_hat = self.activation_function(z)
            print(f"predicted value after forward pass: \n{y_hat}")
            
            # The student checks if their guesses are right and calculates their mistakes
            # The error is the difference between the correct labels and the guesses
            self.error = self.y - y_hat
            print(f"error: \n{self.error}")
            
            # The student updates their knowledge based on their mistakes
            # They do this by adding the dot product of the inputs and the error to the weights
            # The learning rate (eta) controls the size of the update
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)
            print(f"updated weights after epoch: {epoch + 1}/{self.epochs}: \n{self.weights}")
            print("##"*10)
            
    def predict(self, X):
        # The student classifies new objects based on their updated knowledge
        # They do this in the same way as in the fit method, but without updating the weights
        X_with_bias = np.c_[X, -np.ones((len(X), 1))]
        z = self._z_outcome(X_with_bias, self.weights)
        return self.activation_function(z)
    
    def total_loss(self):
        # Calculate the student's score based on their mistakes
        # The total loss is the sum of the absolute values of the errors
        total_loss = np.sum(self.error)
        print(f"\ntotal loss: {total_loss}\n")
        return total_loss
    
    def _create_dir_return_path(self, model_dir, filename):
        # Helper function to create a directory and return the path to a file in that directory
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, filename)
    
    def save(self, filename, model_dir=None):
        # Save the student's knowledge to a diary
        # The knowledge is saved as a binary file using the joblib library
        if model_dir is not None:
            model_file_path = self._create_dir_return_path(model_dir, filename)
            joblib.dump(self, model_file_path)
        else:
            model_file_path = self._create_dir_return_path("model", filename)
            joblib.dump(self, model_file_path)
    
    def load(self, filepath):
        # Read the student's knowledge from the diary
        # The knowledge is loaded from a binary file using the joblib library
        return joblib.load(filepath)
