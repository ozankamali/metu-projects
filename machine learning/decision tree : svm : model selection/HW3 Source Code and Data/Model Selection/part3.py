import pickle
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import f1_score, make_scorer, accuracy_score
import numpy as np
from collections import Counter



dataset, labels = pickle.load(open("../datasets/part3_dataset.data", "rb"))
outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=np.random.randint(1, 1000)) # n_repeats = 10
inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=np.random.randint(1, 1000)) # n_repeats = 5
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(dataset)
dataset = scaler.transform(dataset)
 
N_REPET = 5
# Gradient Boosting Tree # 5 ITERATIONS # 2 HYPERPARAMETERS
gb_params = {'GB__loss':["log_loss"], 'GB__learning_rate':[0.1, 0.01]} # add
gb_performance = []
gb_pipeline = Pipeline([('GB', GradientBoostingClassifier())])
gb_clf = GridSearchCV(gb_pipeline, param_grid=gb_params, refit=True, cv=inner_cv, scoring="f1_micro")
for repet in range(N_REPET):
    gb_performance.append(cross_val_score(gb_clf, dataset, labels, cv=outer_cv, verbose=True)) 
    print(gb_performance)
gb_overall_performance = np.mean(gb_performance)
print(gb_overall_performance)

# MLP # 5 ITERATIONS # 2 HYPERPARAMETERS
# mlp_params = {'MLP__activation':["relu", "tanh"], 'MLP__max_iter':[100]} # add
# mlp_pipeline = Pipeline([('MLP', MLPClassifier())])
# mlp_performance = []
# mlp_overall_performance = []
# mlp_best_params = []
# mlp_pipeline = Pipeline([('MLP', MLPClassifier())])
# mlp_clf = GridSearchCV(mlp_pipeline, param_grid=mlp_params, refit=True, cv=inner_cv, scoring="f1_micro")
# Random Forest # 5 ITERATIONS # 2 HYPERPARAMETERS
# rf_params = {'RF__criterion':["gini", "entropy"], 'RF__n_estimators':[100]} # add
# rf_pipeline = Pipeline([('RF', RandomForestClassifier())])
# rf_performance = []
# rf_overall_performance = []
# rf_best_params = []
# rf_pipeline = Pipeline([('RF', RandomForestClassifier())])
# rf_clf = GridSearchCV(rf_pipeline, param_grid=rf_params, refit=True, cv=inner_cv, scoring="f1_micro")
# # Decision Tree # 1 ITERATIONS # 2 HYPERPARAMETERS
# dt_params = {'DT__criterion':["gini", "entropy"], 'DT__splitter':["best"]} # add
# dt_pipeline = Pipeline([('DT', DecisionTreeClassifier())])
# dt_performance = []
# dt_overall_performance = []
# dt_best_params = []
# dt_pipeline = Pipeline([('DT', DecisionTreeClassifier())])
# dt_clf = GridSearchCV(dt_pipeline, param_grid=dt_params, refit=True, cv=inner_cv, scoring="f1_micro")
# SVM # 1 ITERATIONS # 2 HYPERPARAMETERS
# svm_params = {'SVM__C':[1, 5], 'SVM__kernel':['rbf']} # add
# svm_pipeline = Pipeline([('SVM', SVC())])
# svm_performance = []
# svm_overall_performance = []
# svm_best_params = []
# svm_pipeline = Pipeline([('SVM', SVC())])    
# svm_clf = GridSearchCV(svm_pipeline, param_grid=svm_params, refit=True, cv=inner_cv, scoring="f1_micro")
# KNN # 1 ITERATIONS # 2 HYPERPARAMETERS
# knn_params = {'KNN__n_neighbors':[1, 5], 'KNN__weights':['uniform']} # add
# knn_pipeline = Pipeline([('KNN', KNeighborsClassifier())])
# knn_performance = []
# knn_overall_performance = []
# knn_best_params = []
# knn_pipeline = Pipeline([('KNN', KNeighborsClassifier())])
# knn_clf = GridSearchCV(knn_pipeline, param_grid=knn_params, refit=True, cv=inner_cv, scoring="f1_micro")



# for train_indices, test_indices in outer_cv.split(dataset, labels):
#     current_training_part = dataset[train_indices]
#     current_training_part_label = labels[train_indices]
    
#     # knn_clf.fit(current_training_part, current_training_part_label)
    
#     # svm_clf.fit(current_training_part, current_training_part_label)
    
#     # dt_clf.fit(current_training_part, current_training_part_label)

#     current_test_part = dataset[test_indices]
#     current_test_part_label = labels[test_indices]

#     # knn_predicted = knn_clf.predict(current_test_part)
#     # knn_overall_performance.append(f1_score(current_test_part_label, knn_predicted, average="micro"))
#     # knn_best_params.append(knn_clf.best_params_["KNN__n_neighbors"])
    

#     # svm_predicted = svm_clf.predict(current_test_part)
#     # svm_overall_performance.append(f1_score(current_test_part_label, svm_predicted, average="micro"))
#     # svm_best_params.append(svm_clf.best_params_["SVM__C"])

#     # dt_predicted = dt_clf.predict(current_test_part)
#     # dt_overall_performance.append(f1_score(current_test_part_label, dt_predicted, average="micro"))
#     # dt_best_params.append(dt_clf.best_params_["DT__criterion"])

#     # for repet in range(N_REPET):
#     #     # rf_clf.fit(current_training_part, current_training_part_label)
#     #     # rf_predicted = rf_clf.predict(current_test_part)
#     #     # rf_overall_performance.append(f1_score(current_test_part_label, rf_predicted, average="micro"))
#     #     # rf_best_params.append(rf_clf.best_params_["RF__criterion"])

#     #     # mlp_clf.fit(current_training_part, current_training_part_label)
#     #     # mlp_predicted = mlp_clf.predict(current_test_part)
#     #     # mlp_overall_performance.append(f1_score(current_test_part_label, mlp_predicted, average="micro"))
#     #     # mlp_best_params.append(mlp_clf.best_params_["MLP__activation"])
       
#     #     gb_clf.fit(current_training_part, current_training_part_label)
#     #     gb_predicted = gb_clf.predict(current_test_part)
#     #     gb_overall_performance.append(f1_score(current_test_part_label, gb_predicted, average="micro"))
#     #     gb_best_params.append(gb_clf.best_params_["GB__learning_rate"])
#     # print(len(gb_best_params))

# print("KNN: ")
# print(Counter(knn_best_params))
# std = np.std(knn_overall_performance)
# mean = np.mean(knn_overall_performance)
# d = std / np.sqrt(len(knn_overall_performance))
# ci = (mean - d, mean + d)
# print("overall mean & confidence interval: ", mean, ci)

# print("SVM: ")
# print(Counter(svm_best_params))
# std = np.std(svm_overall_performance)
# mean = np.mean(svm_overall_performance)
# d = std / np.sqrt(len(svm_overall_performance))
# ci = (mean - d, mean + d)
# print("overall mean & confidence interval: ", mean, ci)

# print("DT: ")
# print(Counter(dt_best_params))
# std = np.std(dt_overall_performance)
# mean = np.mean(dt_overall_performance)
# d = std / np.sqrt(len(dt_overall_performance))
# ci = (mean - d, mean + d)
# print("overall mean & confidence interval: ", mean, ci)

# print("RF: ")
# print(Counter(rf_best_params))
# std = np.std(rf_overall_performance)
# mean = np.mean(rf_overall_performance)
# d = std / np.sqrt(len(rf_overall_performance))
# ci = (mean - d, mean + d)
# print("overall mean & confidence interval: ", mean, ci)

# print("MLP: ")
# print(Counter(mlp_best_params))
# std = np.std(mlp_overall_performance)
# mean = np.mean(mlp_overall_performance)
# d = std / np.sqrt(len(mlp_overall_performance))
# ci = (mean - d, mean + d)
# print("overall mean & confidence interval: ", mean, ci)

print("GB: ")
std = np.std(gb_overall_performance)
mean = np.mean(gb_overall_performance)
d = std / np.sqrt(len(gb_overall_performance))
ci = (mean - d, mean + d)
print("overall mean & confidence interval: ", mean, ci)
































# # MLP # 5 ITERATIONS # 2 HYPERPARAMETERS
# n_repetitions = 5
# params = {"MINMAX__feature_range": [(-1, 1)], "MLP__criterion": ["relu", "tanh"]} 
# pipeline = Pipeline([("MINMAX", MinMaxScaler()), ("MLP", MLPClassifier())])
# # pipeline = make_pipeline(MinMaxScaler(), RandomForestClassifier())
# outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=np.random.randint(1, 1000))
# inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=np.random.randint(1, 1000))
# # f1= make_scorer(f1_score, average='micro') # for multilabeled data 
# nested_scores = []

# for repet in range(n_repetitions):
#     clf = GridSearchCV(estimator=pipeline, param_grid=params, cv=inner_cv, scoring="f1_micro") # inner loop
#     nested_score = cross_val_score(clf, dataset, labels, scoring="f1_micro", cv=outer_cv) # outer loop  
#     print(nested_score)
#     nested_scores.append(nested_score)

# mean = np.mean(nested_scores)
# std = np.std(nested_scores)
# n = len(nested_scores)
# confidence_interval = 1.96 * std / np.sqrt(n)

# print("mean f1 score for MLP w/ 5 Iterations: ", mean)
# print("confidence interval for MLP w/ 5 Iterations: ", (mean - confidence_interval, mean + confidence_interval))




# # Random Forest # 5 ITERATIONS # 2 HYPERPARAMETERS
# n_repetitions = 5
# params = {"MINMAX__feature_range": [(-1, 1)], "RF__criterion": ["gini", "entropy"]} 
# pipeline = Pipeline([("MINMAX", MinMaxScaler()), ("RF", RandomForestClassifier())])
# # pipeline = make_pipeline(MinMaxScaler(), RandomForestClassifier())
# outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=np.random.randint(1, 1000))
# inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=np.random.randint(1, 1000))
# # f1= make_scorer(f1_score, average='micro') # for multilabeled data 
# nested_scores = []

# for repet in range(n_repetitions):
#     clf = GridSearchCV(estimator=pipeline, param_grid=params, cv=inner_cv, scoring="f1_micro") # inner loop
#     nested_score = cross_val_score(clf, dataset, labels, scoring="f1_micro", cv=outer_cv) # outer loop  
#     print(nested_score)
#     nested_scores.append(nested_score)

# mean = np.mean(nested_scores)
# std = np.std(nested_scores)
# n = len(nested_scores)
# confidence_interval = 1.96 * std / np.sqrt(n)

# print("mean f1 score for Random Forest w/ 5 Iterations: ", mean)
# print("confidence interval for Random Forest w/ 5 Iterations: ", (mean - confidence_interval, mean + confidence_interval))



















# # DECISION TREE # 1 ITERATIONS # 2 HYPERPARAMETERS
# params = [{'MINMAX__feature_range': [(-1, 1)], 'DT__criterion':['gini', 'entropy'], 'DT__splitter':['best']}] 
# pipeline = Pipeline([('MINMAX', MinMaxScaler()), ('DT', DecisionTreeClassifier())])
# outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10)
# inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
# f1= make_scorer(f1_score, average='micro') # for multilabeled data 
# clf = GridSearchCV(estimator=pipeline, param_grid=params, cv=inner_cv, scoring=f1) # inner loop
# clf.fit(dataset, labels)
# hyperparameter_scores = []
# inner_cv_results = clf.cv_results_
# for i in range(len(inner_cv_results['params'])):
#     params = inner_cv_results['params'][i]
#     mean_score = inner_cv_results['mean_test_score'][i]
#     std_score = inner_cv_results['std_test_score'][i]
#     ci = 1.96 * std_score / np.sqrt(3) # 3 = n_splits
#     hyperparameter_scores.append({'params': params, 'mean_score': mean_score, 'std_score': std_score, 'confidence_interval': (mean_score - ci, mean_score + ci)})
# print("Hyperparameter Search Results: \n")
# for i in range(len(hyperparameter_scores)):
#     print(hyperparameter_scores[i])
#     print("\n")

# nested_score = cross_val_score(clf, dataset, labels, scoring=f1, cv=outer_cv) # outer loop  
# mean = np.mean(nested_score)
# std = np.std(nested_score)
# n = len(nested_score)
# confidence_interval = 1.96 * std / np.sqrt(n)
# print("mean f1 score for Decision Tree: ", mean)
# print("confidence interval for Decision Tree: ", (mean - confidence_interval, mean + confidence_interval))




# SVM # 1 ITERATIONS # 2 HYPERPARAMETERS
# params = [{'MINMAX__feature_range': [(-1, 1)], 'SVC__C':[1, 5], 'SVC__kernel':['rbf']}] 
# pipeline = Pipeline([('MINMAX', MinMaxScaler()), ('SVC', SVC())])
#
# outer_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10)
# inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)
# f1= make_scorer(f1_score, average='micro') # for multilabeled data 
#
# clf = GridSearchCV(estimator=pipeline, param_grid=params, cv=inner_cv, scoring=f1) # inner loop
# clf.fit(dataset, labels)
# hyperparameter_scores = []
# inner_cv_results = clf.cv_results_
# for i in range(len(inner_cv_results['params'])):
#     params = inner_cv_results['params'][i]
#     mean_score = inner_cv_results['mean_test_score'][i]
#     std_score = inner_cv_results['std_test_score'][i]
#     ci = 1.96 * std_score / np.sqrt(3) # 3 = n_splits
#     hyperparameter_scores.append({'params': params, 'mean_score': mean_score, 'std_score': std_score, 'confidence_interval': (mean_score - ci, mean_score + ci)})
# print("Hyperparameter Search Results: \n")
# for i in range(len(hyperparameter_scores)):
#     print(hyperparameter_scores[i])
#     print("\n")
#
# nested_score = cross_val_score(clf, dataset, labels, scoring=f1, cv=outer_cv) # outer loop  
#
# mean = np.mean(nested_score)
# std = np.std(nested_score)
# n = len(nested_score)
# confidence_interval = 1.96 * std / np.sqrt(n)
# print("mean f1 score for SVM: ", mean)
# print("confidence interval for SVM: ", (mean - confidence_interval, mean + confidence_interval))


























