import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt


dataset, labels = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))


#scikitlearn sample code
def plot_training_data_with_decision_boundary(C, kernel, dataset, labels, ax=None, long_title=True, support_vectors=True):
    clf = SVC(C=C, kernel=kernel).fit(dataset, labels)
    clf.fit(dataset, labels)    
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    common_params = {"estimator": clf, "X": dataset, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    if support_vectors:
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    ax.scatter(dataset[:, 0], dataset[:, 1], c=labels, s=30, edgecolors="k")
    if long_title:
        ax.set_title(f"kernel:{kernel} C:{C} in SVC")
    else:
        ax.set_title(kernel)

    if ax is None:
        plt.show()

plot_training_data_with_decision_boundary(C=1.0, kernel="linear", dataset=dataset, labels=labels)
plt.savefig('dataset1_model1.png')

plot_training_data_with_decision_boundary(C=1.0, kernel="rbf", dataset=dataset, labels=labels)
plt.savefig('dataset1_model2.png')


plot_training_data_with_decision_boundary(C=5.0, kernel="linear", dataset=dataset, labels=labels)
plt.savefig('dataset1_model3.png')


plot_training_data_with_decision_boundary(C=5.0, kernel="rbf", dataset=dataset, labels=labels)
plt.savefig('dataset1_model4.png')