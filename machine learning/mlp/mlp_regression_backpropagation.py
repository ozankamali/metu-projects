import pickle
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x: np.array):
    return 1/(1+np.exp(-x))


class MLPRegressor:
    def __init__(self, learning_rate: float, epoch_number: int):
        # Please do not make any changes in this constructor

        # Weights are initialized temporarily for size clarification
        # These values are overwritten via the initialize_parameter function call belove.
        self.W = np.zeros((2, 3))  # of size (2, 3)
        self.W_bias = np.zeros((1, 3))  # of size (1, 3), row vector (1x3) matrix
        self.GAMMA = np.zeros((3, 1))  # of size (3, 1)
        self.GAMMA_bias = np.zeros((1, 1)) # of size (1, 1), row vector (1x1) matrix

        self.learning_rate = learning_rate
        self.epoch_number = epoch_number

        self.initialize_parameters()

    def initialize_parameters(self):
        # Please do not make any changes in this function

        # In order to be able to obtain the same results during training
        # Start with fixed initial weights
        self.W, self.W_bias, self.GAMMA, self.GAMMA_bias = pickle.load(open("../datasets/part1_regression_initial_weights.dat", "rb"))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: two-dimensional data numpy array fed to the network, size: (# of data instances, # of features)
        :return: two numpy arrays as the outputs of the hidden and output layer, the forward propagation result for the given data, sizes: (# of data instances, 3) (hidden layer output), (# of data instances, 1) (output layer output)

        Please implement the forward propagation procedure for the given data instances here
        This function should return two numpy arrays of sizes (# of data instances, 3) (hidden layer output), (# of data instances, 1) (output layer output)
        """
        
        hidden_layer_output = sigmoid(np.dot(x, self.W) + self.W_bias)
        output_layer_output = np.dot(hidden_layer_output, self.GAMMA) + self.GAMMA_bias
        return hidden_layer_output, output_layer_output

    def train(self, data_instances: np.ndarray, labels: np.ndarray):
        # Please only implement the part asked in this function
        # And please do not make any other changes on the already provided code pieces
        for iteration_count in range(1, self.epoch_number + 1):
            for data_index in range(len(data_instances)):
                x = data_instances[data_index].reshape(1, -1)  # convert into a row vector, (1x2) matrix
                label = labels[data_index][0]  
                hidden_layer_output, output_layer_output = self.forward(x)
                
                """
                    Please calculate the weight update rules for W, W_bias, GAMMA and GAMMA_bias matrices here
                    using the "x", "hidden_layer_output", "output_layer_output" and "label" variables defined above
                    
                    The amount of weight changes should be stored in "W_update", "W_bias_update", "GAMMA_update", "GAMMA_bias_update" variables.
                """
                
                derivative_W = -2 * (label - output_layer_output[0, 0]) * (hidden_layer_output * (1 - hidden_layer_output)) * self.GAMMA.flatten()
                W_update = np.dot(x.T, derivative_W)
                W_bias_update = derivative_W * 1 

                derivative_G = -2 * (label - output_layer_output[0, 0])
                GAMMA_update = derivative_G * hidden_layer_output.T
                GAMMA_bias_update = derivative_G * 1

                # After finding update values we are performing the weight updates
                self.W = self.W - self.learning_rate*W_update
                self.W_bias = self.W_bias - self.learning_rate*W_bias_update

                self.GAMMA = self.GAMMA - self.learning_rate*GAMMA_update
                self.GAMMA_bias = self.GAMMA_bias - self.learning_rate*GAMMA_bias_update

            # After each epoch on the dataset, calculate the mean squarred loss with the dataset.

            total_loss_value = 0.0
            for data_index in range(len(data_instances)):
                x = data_instances[data_index].reshape(1, -1)
                label = labels[data_index][0]  # Convert label to scalar
                _, output = self.forward(x)
                total_loss_value += (label - output[0, 0])**2

            mean_mse = total_loss_value / len(data_instances)
            print(f"Epoch Number: {iteration_count} - Training Mean SE: {mean_mse:.3f}")


# Loading data and training the model
X, L = pickle.load(open("../datasets/part1_regression_dataset.dat", "rb"))
mlp = MLPRegressor(learning_rate=0.01, epoch_number=250)
mlp.train(X, L)
