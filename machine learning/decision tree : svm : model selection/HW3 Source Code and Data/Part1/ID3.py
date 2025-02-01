import numpy as np
from collections import Counter

# In the decision tree, non-leaf nodes are going to be represented via TreeNode
class TreeNode:
    def __init__(self, attribute):
        self.attribute = attribute
        # dictionary, k: subtree, key (k) an attribute value, value is either TreeNode or TreeLeafNode
        self.subtrees = {}

# In the decision tree, leaf nodes are going to be represented via TreeLeafNode
class TreeLeafNode:
    def __init__(self, data, label):
        self.data = data
        self.labels = label

class DecisionTree:
    def __init__(self, dataset: list, labels, features, criterion):
        """
        :param dataset: array of data instances, each data instance is represented via an Python array
        :param labels: array of the labels of the data instances
        :param features: the array that stores the name of each feature dimension
        :param criterion: depending on which criterion ("information gain" or "gain ratio") the splits are to be performed
        """
        self.dataset = dataset
        self.labels = labels
        self.features = features
        self.criterion = criterion
        # it keeps the root node of the decision tree
        self.root = None

        # further variables and functions can be added...
        self.used_attributes = []



    def calculate_entropy__(self, dataset, labels):
        """
        :param dataset: array of the data instances
        :param labels: array of the labels of the data instances
        :return: calculated entropy value for the given dataset
        """
        entropy_value = 0.0

        """
        Entropy calculations
        """
        label_counts = Counter(labels)
        total = len(labels)
    
        for count in label_counts.values():
            p = count / total
            if p > 0:
                entropy_value -= p * np.log2(p)
        
        return entropy_value


    def calculate_average_entropy__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an average entropy value is calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute an average entropy value is going to be calculated
        :return: the calculated average entropy value for the given attribute
        """
        average_entropy = 0.0
        
        attribute_values = set([instance[attribute] for instance in dataset])
        
        for value in attribute_values:
            subset_labels = [labels[i] for i in range(len(dataset)) if dataset[i][attribute] == value]
            p = len(subset_labels) / len(labels)
            subset_entropy = self.calculate_entropy__(None, subset_labels)
            average_entropy += p * subset_entropy
        
        return average_entropy

    def calculate_information_gain__(self, dataset, labels, attribute):
        """
        :param dataset: array of the data instances on which an information gain score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the information gain score is going to be calculated...
        :return: the calculated information gain score
        """
        information_gain = 0.0
        """
            Information gain calculations
        """
        entropy = self.calculate_entropy__(None, labels)
        avg_entropy = self.calculate_average_entropy__(dataset, labels, attribute)
        information_gain = entropy - avg_entropy
        return information_gain

    def calculate_intrinsic_information__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances on which an intrinsic information score is going to be calculated
        :param labels: array of the labels of those data instances
        :param attribute: for which attribute the intrinsic information score is going to be calculated...
        :return: the calculated intrinsic information score
        """
        intrinsic_info = 0.0
        """
            Intrinsic information calculations for a given attribute
        """

        attribute_values = set([instance[attribute] for instance in dataset])
        
        for value in attribute_values:
            subset_labels = [labels[i] for i in range(len(dataset)) if dataset[i][attribute] == value]
            if len(labels) > 0:
                p = len(subset_labels) / len(labels)
                intrinsic_info -= p * np.log2(p)
    
        return intrinsic_info
    
    def calculate_gain_ratio__(self, dataset, labels, attribute):
        """
        :param dataset: array of data instances with which a gain ratio is going to be calculated
        :param labels: array of labels of those instances
        :param attribute: for which attribute the gain ratio score is going to be calculated...
        :return: the calculated gain ratio score
        """
        """
            Your implementation
        """
        gain_ratio = 0.0
        
        intrinsic_info = self.calculate_intrinsic_information__(dataset, labels, attribute)
        information_gain = self.calculate_information_gain__(dataset, labels, attribute)
        
        if intrinsic_info > 0:
            gain_ratio = information_gain / intrinsic_info
        return gain_ratio
        


    def ID3__(self, dataset, labels, used_attributes):
        """
        Recursive function for ID3 algorithm
        :param dataset: data instances falling under the current  tree node
        :param labels: labels of those instances
        :param used_attributes: while recursively constructing the tree, already used labels should be stored in used_attributes
        :return: it returns a created non-leaf node or a created leaf node
        """
        """
        Function ID3
         Input: Example set S
         Output: Decision Tree DT
    1    If all examples in S belong to the same class c
         return a new leaf and label it with c
    2    Else
    2.1     i. Select an attribute A according to some heuristic function
    2.2     ii. Generate a new node DT with A as test
    2.3     iii.For each Value vi of A
    3    (a) Let Si = all examples in S with A = vi
    4    (b) Use ID3 to construct a decision tree DTi for example set Si
    5    (c) Generate an edge that connects DT and DT
        
        """
        
        # 1
        if (len(set(labels)) == 1):
            return TreeLeafNode(None, labels[0])
        # 2
        else:
            best_attribute = None
            best_criterion = -1.0
            # 2.1
            for i, feature in enumerate(self.features):
                if len(self.used_attributes) != 16:  # total features
                    score = None  
                    if self.criterion == "information gain" and i not in self.used_attributes:
                        score = self.calculate_information_gain__(dataset, labels, attribute=i)
                    elif self.criterion == "gain ratio" and i not in self.used_attributes:
                        score = self.calculate_gain_ratio__(dataset, labels, attribute=i)
                    
                    if score is not None: 
                        print("attribute:", self.features[i], self.criterion, score)
                        if score > best_criterion:
                            best_attribute = feature
                            best_attribute_index = i
                            best_criterion = score 
                else: 
                    return node
            print("---------------------------------------------------------------")
            print("best attribute:", best_attribute, "best score:", best_criterion)
            print("---------------------------------------------------------------")
            # 2.2
            node = TreeNode(attribute=best_attribute)
            self.used_attributes.append(best_attribute_index)
            print("used attributes", ([self.features[i] for i in self.used_attributes]))
            attribute_values = set([instance[best_attribute_index] for instance in dataset])

            # 2.3
            for value in attribute_values:    
                # 3
                subset = [dataset[i] for i in range(len(dataset)) if dataset[i][best_attribute_index] == value]
                subset_labels = [labels[i] for i in range(len(dataset)) if dataset[i][best_attribute_index] == value]
                # 4
                node.subtrees[value] = self.ID3__(subset, subset_labels, self.used_attributes.copy())
            
            return node            

    def predict(self, x):
        """
        :param x: a data instance, 1-dimensional Python array 
        :return: predicted label of x

        If a leaf node contains multiple labels in it, the majority label should be returned as the predicted label.
        """
        curr = self.root
        while type(curr) != TreeLeafNode:
            attribute_index = self.features.index(curr.attribute)
            attribute_value = x[attribute_index]
            curr = curr.subtrees[attribute_value]
        label_counts = Counter([curr.labels])
        predicted_label = label_counts.most_common(1)[0][0]
       
        return predicted_label

    def train(self):
        self.root = self.ID3__(self.dataset, self.labels, self.used_attributes)
        print("Training completed")