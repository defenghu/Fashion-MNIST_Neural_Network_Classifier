import os

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from tqdm import tqdm


# Function to convert labels into one-hot vectors
def convert_to_one_hot(labels: np.ndarray) -> np.ndarray:
    one_hot_labels = np.zeros((labels.size, labels.max() + 1))
    rows = np.arange(labels.size)
    one_hot_labels[rows, labels] = 1
    return one_hot_labels


# Function to plot the loss and accuracy history
def plot_loss_and_accuracy(loss_history: list, accuracy_history: list, filename='plot'):
    num_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, num_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, num_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


# Function to scale the train and test data
def scale_data(train_data, test_data):
    max_value = np.concatenate((train_data, test_data)).max()
    scaled_train_data = train_data / max_value
    scaled_test_data = test_data / max_value
    return scaled_train_data, scaled_test_data


# Function to initialize weights using Xavier initialization
def xavier_initialization(num_inputs, num_outputs):
    total_nodes = num_inputs * num_outputs
    limit = np.sqrt(6 / (num_inputs + num_outputs))
    weights = np.random.uniform(-limit, limit, (num_inputs, num_outputs))
    return weights


# Function to compute mean squared error
def mean_squared_error(predicted, actual):
    return np.mean((predicted - actual) ** 2)


# Function to compute derivative of mean squared error
def derivative_mean_squared_error(predicted, actual):
    return 2 * (predicted - actual)


# Function to compute sigmoid activation
def sigmoid_activation(input_data):
    return 1 / (1 + np.exp(-input_data))


# Function to compute derivative of sigmoid activation
def derivative_sigmoid_activation(input_data):
    return (1 / (1 + np.exp(-input_data))) * (1 - (1 / (1 + np.exp(-input_data))))


# Function to train the model
def train_model(model, learning_rate, train_data, train_labels, batch_size=100):
    num_samples = train_data.shape[0]
    for i in range(0, num_samples, batch_size):
        model.backpropagation(train_data[i:i + batch_size], train_labels[i:i + batch_size], learning_rate)


# Function to compute accuracy of the model
def compute_accuracy(model, train_data, train_labels):
    model.forward_propagation(train_data)
    predicted_labels = np.argmax(model.forward_step, axis=1)
    actual_labels = np.argmax(train_labels, axis=1)
    model.accuracy = np.mean(predicted_labels == actual_labels)
    model.loss = np.mean((predicted_labels - actual_labels) ** 2)


# Class for one-layer neural network
class OneLayerNeuralNetwork:
    def __init__(self, num_features, num_classes):
        self.weights = xavier_initialization(num_features, num_classes)
        self.biases = xavier_initialization(1, num_classes)
        self.forward_step = None
        self.accuracy = None
        self.loss = None

    # Function for forward propagation
    def forward_propagation(self, input_data):
        forward_step = np.dot(input_data, self.weights) + self.biases
        self.forward_step = sigmoid_activation(forward_step)

    # Function for backpropagation
    def backpropagation(self, input_data, labels, learning_rate):
        self.forward_propagation(input_data)
        error = derivative_mean_squared_error(self.forward_step, labels) * derivative_sigmoid_activation(
            np.dot(input_data, self.weights) + self.biases)
        gradient_weights = np.dot(input_data.T, error) / input_data.shape[0]
        gradient_biases = np.mean(error, axis=0)
        self.weights -= learning_rate * gradient_weights
        self.biases -= learning_rate * gradient_biases


# Class for two-layer neural network
class TwoLayerNeuralNetwork:
    def __init__(self, num_features, num_classes):
        num_hidden_nodes = 64
        self.weights = [xavier_initialization(num_features, num_hidden_nodes),
                        xavier_initialization(num_hidden_nodes, num_classes)]
        self.biases = [xavier_initialization(1, num_hidden_nodes), xavier_initialization(1, num_classes)]
        self.forward_step = None
        self.accuracy = None
        self.loss = None

    # Function for forward propagation
    def forward_propagation(self, input_data):
        for i in range(2):
            input_data = sigmoid_activation(input_data @ self.weights[i] + self.biases[i])
        self.forward_step = input_data

    # Function for backpropagation
    def backpropagation(self, input_data, labels, learning_rate):
        self.forward_propagation(input_data)
        predicted_labels = self.forward_step
        num_samples = input_data.shape[0]
        biases = np.ones((1, num_samples))
        loss_gradient_1 = 2 * learning_rate / num_samples * (
                (predicted_labels - labels) * predicted_labels * (1 - predicted_labels))
        output_first_layer = sigmoid_activation(np.dot(input_data, self.weights[0]) + self.biases[0])
        loss_gradient_0 = np.dot(loss_gradient_1, self.weights[1].T) * output_first_layer * (1 - output_first_layer)
        self.weights[0] -= np.dot(input_data.T, loss_gradient_0)
        self.weights[1] -= np.dot(output_first_layer.T, loss_gradient_1)
        self.biases[0] -= np.dot(biases, loss_gradient_0)
        self.biases[1] -= np.dot(biases, loss_gradient_1)


if __name__ == '__main__':

    # Check if data directory exists, if not, create it
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Downloading train dataset.')
        train_data_url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        train_data_request = requests.get(train_data_url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(train_data_request.content)
        print('Train dataset downloaded.')

        print('Downloading test dataset.')
        test_data_url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        test_data_request = requests.get(test_data_url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(test_data_request.content)
        print('Test dataset downloaded.')

    # Read train and test data
    raw_train_data = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test_data = pd.read_csv('../Data/fashion-mnist_test.csv')

    # Separate features and labels
    train_features = raw_train_data[raw_train_data.columns[1:]].values
    test_features = raw_test_data[raw_test_data.columns[1:]].values

    train_labels = convert_to_one_hot(raw_train_data['label'].values)
    test_labels = convert_to_one_hot(raw_test_data['label'].values)
    '''
    # Stage 1/7: Scale the data and initialize the model
    scaled_train_features, scaled_test_features = scale_data(train_features, test_features)
    print([scaled_train_features[2, 778], scaled_test_features[0, 774]], end=" ")
    print(xavier_initialization(2, 3).flatten().tolist(), end=" ")
    sample_values = [-1, 0, 1, 2]
    print(sigmoid_activation(sample_values))

    # Stage 2/7: Forward propagation
    scaled_train_features, scaled_test_features = scale_data(train_features, test_features)
    one_layer_model = OneLayerNeuralNetwork(scaled_train_features.shape[1], train_labels.shape[1])
    one_layer_model.forward_propagation(scaled_train_features)
    print(one_layer_model.forward_step[:2].flatten().tolist())

    # Stage 3/7: Backpropagation
    array1 = np.array([-1, 0, 1, 2])
    array2 = np.array([4, 3, 2, 1])
    print(mean_squared_error(array1, array2).flatten().tolist(), end=" ")
    print(derivative_mean_squared_error(array1, array2).flatten().tolist(), end=" ")
    print(derivative_sigmoid_activation(array1).flatten().tolist(), end=" ")
    scaled_train_features, scaled_test_features = scale_data(train_features, test_features)
    one_layer_model = OneLayerNeuralNetwork(scaled_train_features.shape[1], train_labels.shape[1])
    learning_rate = 0.1
    one_layer_model.forward_propagation(scaled_train_features[:2])
    one_layer_model.backpropagation(scaled_train_features[:2], train_labels[:2], learning_rate)
    one_layer_model.forward_propagation(scaled_train_features[:2])
    print(mean_squared_error(one_layer_model.forward_step[:2], train_labels[:2]).flatten().tolist())

    # Stage 4/7: Training and evaluation of one-layer model
    scaled_train_features, scaled_test_features = scale_data(train_features, test_features)
    one_layer_model = OneLayerNeuralNetwork(scaled_train_features.shape[1], train_labels.shape[1])
    learning_rate = 0.5
    compute_accuracy(one_layer_model, scaled_train_features, train_labels)
    initial_accuracy = one_layer_model.accuracy.flatten().tolist()
    accuracy_history = []
    loss_history = []
    for i in tqdm(range(20)):
        train_model(one_layer_model, learning_rate, scaled_train_features, train_labels)
        compute_accuracy(one_layer_model, scaled_test_features, test_labels)
        accuracy_history.append(one_layer_model.accuracy)
        loss_history.append(one_layer_model.loss)
    final_accuracy = np.array(accuracy_history).flatten().tolist()
    print(initial_accuracy, final_accuracy)
    plot_loss_and_accuracy(loss_history, accuracy_history)

    # Stage 5/7: Forward propagation of two-layer model
    scaled_train_features, scaled_test_features = scale_data(train_features, test_features)
    two_layer_model = TwoLayerNeuralNetwork(scaled_train_features.shape[1], train_labels.shape[1])
    two_layer_model.forward_propagation(scaled_train_features[:2])
    print(two_layer_model.forward_step.flatten().tolist())

    # Stage 6/7: Backpropagation of two-layer model
    scaled_train_features, scaled_test_features = scale_data(train_features, test_features)
    learning_rate = 0.1
    two_layer_model = TwoLayerNeuralNetwork(scaled_train_features.shape[1], train_labels.shape[1])
    two_layer_model.forward_propagation(scaled_train_features[:2])
    two_layer_model.backpropagation(scaled_train_features[:2], train_labels[:2], learning_rate)
    two_layer_model.forward_propagation(scaled_train_features[:2])
    print(mean_squared_error(two_layer_model.forward_step[:2], train_labels[:2]).flatten().tolist())
    '''
    # Stage 7/7: Training and evaluation of two-layer model
    scaled_train_features, scaled_test_features = scale_data(train_features, test_features)
    two_layer_model = TwoLayerNeuralNetwork(scaled_train_features.shape[1], train_labels.shape[1])
    learning_rate = 0.5
    accuracy_history = []
    loss_history = []
    for i in tqdm(range(20)):
        train_model(two_layer_model, learning_rate, scaled_train_features, train_labels)
        compute_accuracy(two_layer_model, scaled_test_features, test_labels)
        accuracy_history.append(two_layer_model.accuracy)
        loss_history.append(two_layer_model.loss)
    print(np.array(accuracy_history).flatten().tolist())
    plot_loss_and_accuracy(loss_history, accuracy_history)
