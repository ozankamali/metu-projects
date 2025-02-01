import pickle
import numpy as np

def sigmoid(x: np.array):
    return 1/(1+np.exp(-x))


def softmax(x: np.array):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class MLPClassifier:
    def __init__(self, learning_rate: float, epoch_number: int):
        # Please do not make any changes in this constructor

        # Weights are initialized temporarily for size clarification
        # These values are overwritten via the initialize_parameter function call belove.
        self.W = np.zeros((4, 3))  # of size (4, 3)
        self.W_bias = np.zeros((1, 3))  # of size (1, 3), a row vector (1x3 matrix)
        self.GAMMA = np.zeros((3, 3))  # of size (3, 3)
        self.GAMMA_bias = np.zeros((1, 3))  # of size (1, 3), a row vector (1x3) matrix

        self.learning_rate = learning_rate
        self.epoch_number = epoch_number

        self.initialize_parameters()

    def initialize_parameters(self):
        # Please do not make any changes in this function

        # In order to be able to obtain the same results during training
        # Start with fixed initial weights
        self.W, self.W_bias, self.GAMMA, self.GAMMA_bias = pickle.load(open("../datasets/part1_classification_initial_weights.dat", "rb"))


    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: two-dimensional data numpy array fed to the network, size: (# of data instances, # of features)
        :return: two numpy arrays as the outputs of the hidden and output layers, the forward propagation result for the given data, sizes: (# of data instances, 3) (hidden layer output), (# of data instances, 3) (output layer output)

        Please implement the forward propagation procedure for the given data instances here
        This function should return two numpy arrays of sizes (# of data instances, 3) (hidden layer output)  and (# of data instances, 3) (output layer output)
        """
        hidden_layer_output = sigmoid(np.dot(x, self.W) + self.W_bias) #1x3
        output_layer_output = softmax(np.dot(hidden_layer_output, self.GAMMA) + self.GAMMA_bias) #1x3
        return hidden_layer_output, output_layer_output

    def train(self, data_instances: np.ndarray, labels: np.ndarray):
        # Please only implement the part asked in this function
        # And please do not make any other changes on the already provided code pieces
        for iteration_count in range(1, self.epoch_number+1):
            for data_index in range(len(data_instances)):
                x = data_instances[data_index].reshape(1, -1) # convert into a row vector, (1x4) matrix
                label = labels[data_index]

                hidden_layer_output, output_layer_output = self.forward(x)
                """
                    Please calculate the weight update rules for W, W_bias and GAMMA and GAMMA_bias matrices here
                    using the "x", "hidden_layer_output", "output_layer_output" and "label" variables defined above
                    
                    The amount of weight changes should be stored in "W_update", "W_bias_update", "GAMMA_update", "GAMMA_bias_update" variable.
                """
                
                derivative_softmax = output_layer_output - label
                GAMMA_update = np.dot(hidden_layer_output.T, derivative_softmax) #must be 3x3 = 3x1 * 1x3 with 
                GAMMA_bias_update = derivative_softmax * 1

                derivative_W = np.dot(derivative_softmax, self.GAMMA.T) * hidden_layer_output * (1 - hidden_layer_output)
                W_update = np.dot(x.T, derivative_W)
                W_bias_update = derivative_W * 1

                # After finding update values we are performing the weight updates
                self.W = self.W - self.learning_rate*W_update
                self.W_bias = self.W_bias - self.learning_rate*W_bias_update

                self.GAMMA = self.GAMMA - self.learning_rate*GAMMA_update
                self.GAMMA_bias = self.GAMMA_bias - self.learning_rate*GAMMA_bias_update

            # After each epoch on the dataset, calculate the Mean Cross Entropy loss and accuracy with the dataset.
            correct, wrong = 0, 0
            total_loss_value = 0.0
            for data_index in range(len(data_instances)):
                x = data_instances[data_index].reshape(1, -1)  # convert into a row vector, 1x4 matrix
                label = labels[data_index]
                _, output = self.forward(x)
                loss = 0
                max_index = None
                max_value = -float('inf')
                for i in range(len(output[0])):
                    loss += label[i]*np.log(output[0][i])
                    if output[0][i] > max_value:
                        max_value = output[0][i]
                        max_index = i
                total_loss_value += -loss
                if label[max_index] == 1:
                    correct += 1
                else:
                    wrong += 1
            mean_ce = total_loss_value/len(data_instances)
            accuracy = correct/(correct+wrong)*100
            print(f"Epoch Number: {iteration_count} - Training Mean CE: {mean_ce:.3f} - Training Accuracy: {accuracy:.2f}")


X, L = pickle.load(open("../datasets/part1_classification_dataset.dat", "rb"))
mlp = MLPClassifier(learning_rate=0.01, epoch_number=120)
mlp.train(X, L)
