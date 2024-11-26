#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.weights = np.random.rand(2)  # Randomly initialize weights for two inputs
        self.bias = np.random.rand(1)[0]  # Randomly initialize bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        linear_output = np.dot(inputs, self.weights) + self.bias
        return self.step_function(linear_output)

    def train(self, X, y):
        for epoch in range(self.epochs):
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                error = target - prediction
                # Update weights and bias
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

# Training dataset for (inputs, target)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # Target output (AND gate behavior)

# Initialize and train the perceptron
perceptron = Perceptron(learning_rate=0.1, epochs=100)
perceptron.train(X, y)

# Test the perceptron
for inputs in X:
    print(f"Input: {inputs}, Prediction: {perceptron.predict(inputs)}")


# In[ ]:




