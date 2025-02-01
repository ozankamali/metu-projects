import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from operator import itemgetter

dataset, labels = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

params = [{'SVC__C':[1.0, 2.0, 5.0], 'SVC__kernel':['linear', 'rbf', 'sigmoid']}] 
pipe_svc = Pipeline([('scaler', StandardScaler()), ('SVC', SVC())])
n_iterations = 5
ci = []
for repet in range(n_iterations):
    clf = GridSearchCV(estimator=pipe_svc, param_grid=params, cv=StratifiedKFold(n_splits=10, shuffle=True), scoring='precision')
    clf.fit(dataset, labels)
    results = clf.cv_results_
    results = pd.DataFrame(results)

    
    results.to_csv(f'out{repet+1}.csv', columns=['params', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'split3_test_score', 'split4_test_score', 'split5_test_score', 'split6_test_score', 'split7_test_score', 'split8_test_score', 'split9_test_score', 'mean_test_score', 'std_test_score'], index=False)

df = pd.read_csv("out1.csv")
scores = []
means = []
confidence_intervals = []
for j, hp in enumerate(df['params']):
    for file in range(5):
        file = f"out{file+1}.csv"
        df = pd.read_csv(file)
        for i in range(10):
            scores.append(df[f'split{i}_test_score'][j])
    mean = np.mean(scores)
    std = np.std(scores)
    ci = 1.96 * std / np.sqrt(len(scores))
    means.append((hp, round(mean, 4)))
    confidence_intervals.append((hp, (round(mean - ci, 4), round(mean + ci, 4))))
    scores = []
print("confidence intervals: \n", confidence_intervals)
print("means: \n", means)
best_hp = max(means, key = itemgetter(1))[0]
best_mean = max(means, key = itemgetter(1))[1]
print("best parameters: \n", (best_hp, best_mean))