# -*- coding: utf-8 -*-
'''
Class 0: Setosa
Class 1: Versicolor
Class 2: Virginica
'''

# Import the necessary libraries/modules
import mlflow  # Import MLflow library for experiment tracking
mlflow.set_tracking_uri("http://localhost:8090")  # Set the tracking URI for MLflow server
mlflow.set_experiment("my_iris_experiment_one")  # Set the experiment name for MLflow

import tensorflow as tf  # Import TensorFlow library
import numpy as np  # Import NumPy library
import joblib  # Import Joblib library for object serialization
from sklearn.datasets import load_iris  # Import Iris dataset from scikit-learn
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for feature scaling

mlflow.autolog()  # Enable automatic logging of TensorFlow metrics to MLflow

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()  # Initialize a StandardScaler object
X_train = scaler.fit_transform(X_train)  # Fit the scaler to training data and transform it
X_test = scaler.fit_transform(X_test)  # Fit the scaler to testing data and transform it

# Save the scaler object to a file
joblib.dump(scaler, 'scaler.pkl')  # Serialize and save the scaler object to 'scaler.pkl' file

# Build the neural network model using TensorFlow Keras Sequential API
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),  # Input layer with 64 neurons and ReLU activation, input shape of (4,) for 4 features
        tf.keras.layers.Dense(3, activation='softmax')  # Output layer with 3 neurons and softmax activation
    ]
)

# Compile the model
model.compile(
    optimizer='adam',  # Adam optimizer for optimization
    loss='sparse_categorical_crossentropy',  # Sparse categorical crossentropy loss function for multi-class classification
    metrics=['accuracy']  # Metric to monitor during training, accuracy in this case
)

# Train the model
model.fit(
    X_train, y_train,  # Training data
    epochs=10,  # Number of epochs for training
    validation_data=(X_test, y_test)  # Validation data
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Save the trained model
# model.save("model.keras")
