import torch
import torch.nn as nn
import numpy as np
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: ", device)


# Load the best hyperparameters
with open('best_hyperparameters.pkl', 'rb') as f:
    best_hyperparameters = pickle.load(f)

best_neurons = best_hyperparameters['config']['#_neurons']
best_iterations = best_hyperparameters['config']['#_iterations']
best_activation_fn = best_hyperparameters['config']['activation_function']
best_lr = best_hyperparameters['config']['learning_rate']

print("Best Number of Neurons:", best_neurons)
print("Best Number of Iterations:", best_iterations)
print("Best Activation Function:", best_activation_fn)
print("Best Learning Rate:", best_lr)


class MLPModel(nn.Module):
    def __init__(self, best_neuron, best_activation_f):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(784, best_neuron)
        self.layer2 = nn.Linear(best_neuron, 10)
        self.activation_function = best_activation_f

    def forward(self, x):
        hidden_layer_output = self.activation_function(self.layer1(x))
        output_layer = self.layer2(hidden_layer_output)
        return output_layer

x_train, y_train = pickle.load(open("../datasets/part3_train_dataset.dat", "rb"))
x_validation, y_validation = pickle.load(open("../datasets/part3_validation_dataset.dat", "rb"))
x_test, y_test = pickle.load(open("../datasets/part3_test_dataset.dat", "rb"))


x_train = x_train / 255.0
x_validation = x_validation / 255.0
x_test = x_test / 255.0

x_train = torch.from_numpy(x_train).float().to(device)  
y_train = torch.from_numpy(y_train).long().to(device)    

x_validation = torch.from_numpy(x_validation).float().to(device)  
y_validation = torch.from_numpy(y_validation).long().to(device)    

x_test = torch.from_numpy(x_test).float().to(device)  
y_test = torch.from_numpy(y_test).long().to(device)    


x_combined = torch.cat((x_train, x_validation), dim=0)
y_combined = torch.cat((y_train, y_validation), dim=0)

ITERATION = best_iterations

test_accuracies = []

for repet in range(1, 11):
    nn_model = MLPModel(best_neurons, best_activation_fn).to(device)  
    nn_model.train()  
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=best_lr)

    soft_max_function = torch.nn.Softmax(dim=1)

    for iteration in range(1, ITERATION + 1):
        optimizer.zero_grad()
        predictions = nn_model(x_combined)
        loss_value = loss_function(predictions, y_combined)
        loss_value.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss = loss_function(predictions, y_combined)
            predictions = nn_model(x_validation)
            probability_score_values = soft_max_function(predictions)
    
        print("Iteration : %d Training Loss : %f " % (iteration, train_loss.item()))

    nn_model.eval()
    with torch.no_grad():
        predictions = nn_model(x_test)
        test_loss = loss_function(predictions, y_test)
        print("Test - Loss %.2f" % (test_loss.item()))

        correct_predictions = (predictions.argmax(dim=1) == y_test).sum().item()
        test_accuracy = correct_predictions / y_test.size(0)
        test_accuracies.append(test_accuracy)
        print("Test Accuracy #%d %.2f%%" % (repet, test_accuracy * 100))


test_accuracies_mean = np.mean(test_accuracies)
test_accuracies_std = np.std(test_accuracies)

confidence_interval = (
    float(test_accuracies_mean - 1.96 * (test_accuracies_std / np.sqrt(len(test_accuracies)))), 
    float(test_accuracies_mean + 1.96 * (test_accuracies_std / np.sqrt(len(test_accuracies))))
)

print("Accuracy Mean for Best HP: ", test_accuracies_mean)
print("Confidence Interval for Best HP: ", confidence_interval)
