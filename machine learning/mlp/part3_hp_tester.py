import torch
import torch.nn as nn
import numpy as np
import pickle
import copy
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: ", device)

# training data is already shuffled
x_train, y_train = pickle.load(open("../datasets/part3_train_dataset.dat", "rb"))
x_validation, y_validation = pickle.load(open("../datasets/part3_validation_dataset.dat", "rb"))
x_test, y_test = pickle.load(open("../datasets/part3_test_dataset.dat", "rb"))


x_train = torch.from_numpy((x_train / 255.0).astype(np.float32)).to(device)
x_test = torch.from_numpy((x_test / 255.0).astype(np.float32)).to(device)
x_validation = torch.from_numpy((x_validation / 255.0).astype(np.float32)).to(device)


y_train = torch.from_numpy(y_train).to(torch.long).to(device)
y_validation = torch.from_numpy(y_validation).to(torch.long).to(device)
y_test = torch.from_numpy(y_test).to(torch.long).to(device)

hyperparameters = {
    "#_neurons": [64, 128],
    "#_iterations": [75, 150],
    "activation_function": [nn.ReLU(), nn.Tanh(), nn.Sigmoid()],
    "learning_rate": [0.001, 0.01]
}

configurations = list(itertools.product(
    hyperparameters["#_neurons"],
    hyperparameters["#_iterations"],
    hyperparameters["activation_function"],
    hyperparameters["learning_rate"]
))

class MLPModel(nn.Module):
    def __init__(self, neuron_size, activation_f):
        super(MLPModel, self).__init__()
        self.layer1 = nn.Linear(784, neuron_size)
        self.layer2 = nn.Linear(neuron_size, 10)
        self.activation_function = activation_f

    def forward(self, x):
        hidden_layer_output = self.activation_function(self.layer1(x))
        output_layer = self.layer2(hidden_layer_output)
        return output_layer

general_accuracies = []

for (neurons, iterations, activation_fn, lr) in configurations:
    print("Starting with Hyperparameter Settings: \n", "hidden neurons: ", neurons, ", epochs: ", iterations, ", activation function: ", activation_fn, ", learning rate: ", lr)
    
    test_accuracies = []

    for repet in range(1, 11):
        
        nn_model = MLPModel(neurons, activation_fn).to(device)
        nn_model.train() 
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)

        soft_max_function = torch.nn.Softmax(dim=1)

        ITERATION = iterations

        for iteration in range(1, ITERATION + 1):
            optimizer.zero_grad()
            
            predictions = nn_model(x_train)
            loss_value = loss_function(predictions, y_train)
            
            loss_value.backward()
            optimizer.step()

            with torch.no_grad():
                train_prediction = nn_model(x_train)
                train_loss = loss_function(train_prediction, y_train)
                

                predictions = nn_model(x_validation)
                probability_score_values = soft_max_function(predictions)
                validation_loss = loss_function(predictions, y_validation)

            print("Iteration : %d Training Loss : %f - Validation Loss %f" % (iteration, train_loss.item(), validation_loss.item()))

        nn_model.eval()
        with torch.no_grad():
            predictions = nn_model(x_test)
            test_loss = loss_function(predictions, y_test)
            print("Test - Loss %.2f" % (test_loss))


            correct_predictions = (predictions.argmax(dim=1) == y_test).sum().item()
            test_accuracy = correct_predictions / y_test.size(0)
            test_accuracies.append(test_accuracy)
            print("Test Accuracy #%d %.2f%%" % (repet, test_accuracy * 100))

    test_accuracies_mean = np.mean(test_accuracies)
    test_accuracies_std = np.std(test_accuracies)

    confidence_interval = (
        float(test_accuracies_mean - 1.96*(test_accuracies_std/np.sqrt(len(test_accuracies)))), 
        float(test_accuracies_mean + 1.96*(test_accuracies_std/np.sqrt(len(test_accuracies))))
    )

    print("Accuracy Mean: ", test_accuracies_mean)
    print("Confidence Interval: ", confidence_interval)

    general_accuracies.append({
        "config": {
            "#_neurons": neurons,
            "#_iterations": iterations,
            "activation_function": activation_fn,
            "learning_rate": lr
        },
        "mean_accuracy": test_accuracies_mean,
        "confidence_interval": confidence_interval
    })

best_index = np.argmax(np.array([config["mean_accuracy"] for config in general_accuracies]))
best_config = general_accuracies[best_index]

with open('best_hyperparameters.pkl', 'wb') as f:
    pickle.dump(best_config, f)

print("\nBest Configuration:")
print("Neurons per Layer:", best_config["config"]["#_neurons"])
print("Iterations:", best_config["config"]["#_iterations"])
print("Activation Function:", best_config["config"]["activation_function"])
print("Learning Rate:", best_config["config"]["learning_rate"])

print("Mean Accuracy:", best_config["mean_accuracy"])
print("Confidence Interval:", best_config["confidence_interval"])


print("\nAll Configurations Mean Accuracy and Confidence Intervals:")
for index, accuracy_data in enumerate(general_accuracies):
    config = accuracy_data["config"]
    mean_accuracy = accuracy_data["mean_accuracy"]
    confidence_interval = accuracy_data["confidence_interval"]
    
    print(f"Configuration {index + 1}:")
    print(f"  Neurons per Layer: {config['#_neurons']}")
    print(f"  Iterations: {config['#_iterations']}")
    print(f"  Activation Function: {config['activation_function']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Mean Accuracy: {mean_accuracy}")
    print(f"  Confidence Interval: {confidence_interval}\n")
