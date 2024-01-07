# Fashion-MNIST Neural Network Classifier

This project is a Python-based implementation of a neural network model for classification tasks. It uses the
Fashion-MNIST dataset, which is a dataset of Zalando's article images, consisting of a training set of 60,000 examples
and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

## Key Features

1. __Data Preprocessing:__ The project includes functions for downloading the Fashion-MNIST dataset, scaling the data,
   and converting labels into one-hot vectors.
2. __Neural Network Models:__ The project implements both a one-layer and a two-layer neural network model. The weights
   of the models are initialized using the Xavier initialization method.
3. __Activation Functions:__ The sigmoid activation function is used in the neural network models, with the derivative
   of the sigmoid function also implemented for use in backpropagation.
4. __Loss Function:__ The mean squared error function is used as the loss function, with its derivative also implemented
   for use in backpropagation.
5. __Training and Evaluation:__ The project includes functions for training the models using batch gradient descent,
   computing the accuracy of the models, and plotting the loss and accuracy history.
6. __Data Visualization:__ The project uses matplotlib to plot the loss and accuracy history during the training
   process.

## Getting Started

To get a local copy up and running, follow these simple steps:

1. Clone the repository:

```sh
git clone https://github.com/defenghu/Fashion-MNIST_Neural_Network_Classifier.git
```

2. Install required packages:

```sh
pip install -r requirements. txt
```

3. Run the script:

```sh
python test/neural_network.py
```

## Usage

The main script of the project is designed to be run from the command line. It first checks if the Fashion-MNIST dataset
is available, and if not, it downloads the dataset. It then preprocesses the data, initializes the models, and trains
them. The script also evaluates the models on the test set and plots the loss and accuracy history.

The project also includes commented-out code for different stages of the process, which can be uncommented and run to
understand the intermediate steps of the process.

## Dependencies

The project requires the following Python libraries:

- numpy
- pandas
- matplotlib
- os
- requests
- tqdm
- time

## Future Work

Future improvements to the project could include implementing additional features such as regularization, different
optimization methods, and other types of neural network architectures. Additionally, the project could be extended to
work with other datasets and tasks.

## License

This project is open-sourced under the MIT License. This license permits reuse within proprietary software provided that
all copies of the licensed software include a copy of the MIT License terms and the copyright notice.

## Acknowledgments

This project would not have been possible without the following resources and contributions:

- The Fashion-MNIST dataset, which was provided by Zalando Research for academic use and machine learning benchmarking.
- The Python open-source community for providing an extensive collection of libraries which were instrumental in the
  development of this project, including NumPy, Pandas, Matplotlib, and others.

I also extend my gratitude to the maintainers and contributors of the various Python libraries and tools that were used
in the development of this project. Their hard work and dedication to open-source development have greatly facilitated
my progress.