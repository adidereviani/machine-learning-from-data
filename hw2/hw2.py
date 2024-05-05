import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
                 0.25: 1.32,
                 0.1: 2.71,
                 0.05: 3.84,
                 0.0001: 100000},
             2: {0.5: 1.39,
                 0.25: 2.77,
                 0.1: 4.60,
                 0.05: 5.99,
                 0.0001: 100000},
             3: {0.5: 2.37,
                 0.25: 4.11,
                 0.1: 6.25,
                 0.05: 7.82,
                 0.0001: 100000},
             4: {0.5: 3.36,
                 0.25: 5.38,
                 0.1: 7.78,
                 0.05: 9.49,
                 0.0001: 100000},
             5: {0.5: 4.35,
                 0.25: 6.63,
                 0.1: 9.24,
                 0.05: 11.07,
                 0.0001: 100000},
             6: {0.5: 5.35,
                 0.25: 7.84,
                 0.1: 10.64,
                 0.05: 12.59,
                 0.0001: 100000},
             7: {0.5: 6.35,
                 0.25: 9.04,
                 0.1: 12.01,
                 0.05: 14.07,
                 0.0001: 100000},
             8: {0.5: 7.34,
                 0.25: 10.22,
                 0.1: 13.36,
                 0.05: 15.51,
                 0.0001: 100000},
             9: {0.5: 8.34,
                 0.25: 11.39,
                 0.1: 14.68,
                 0.05: 16.92,
                 0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels = data if data.ndim == 1 else data[:, -1]
    total_samples = len(labels)
    label_types, label_counts = np.unique(labels, return_counts=True)
    label_probs = label_counts / total_samples
    gini = 1 - sum(np.square(label_probs))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    labels_set = data if data.ndim == 1 else data[:, -1]
    total_samples = len(labels_set)
    label_types, label_counts = np.unique(labels_set, return_counts=True)
    label_probabilities = label_counts / total_samples
    entropy = -np.sum(label_probabilities * np.log2(label_probabilities))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy


class DecisionNode:

    def __init__(self, data, impurity_func, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.data.size == 0:
            return pred

        labels = self.data[:, -1]
        unique_labels, counts = np.unique(labels, return_counts=True)
        pred = unique_labels[np.argmax(counts)]

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.children.append(node)
        self.children_values.append(val)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.

        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in
        self.feature_importance
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if self.feature == -1 or n_total_sample == 0:
            self.feature_importance = 0
            return

        impurity_initial = self.impurity_func(self.data)
        n_node_samples = len(self.data)
        weighted_impurity = 0

        for child in self.children:
            child_weight = len(child.data) / n_total_sample
            weighted_impurity += child_weight * self.impurity_func(child.data)

        self.feature_importance = (n_node_samples / n_total_sample) * impurity_initial - weighted_impurity
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting
                  according to the feature values.
        """
        goodness = 0
        groups = {}  # groups[feature_value] = data_subset
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        feature_values = self.data[:, feature]
        unique_values = np.unique(feature_values)
        for value in unique_values:
            subset = self.data[feature_values == value]
            groups[value] = subset

        impurity_initial = self.impurity_func(self.data)
        total_samples = len(self.data)
        weighted_impurity = sum(
            (len(subset) / total_samples) * self.impurity_func(subset) for subset in groups.values())
        information_gain = impurity_initial - weighted_impurity

        if self.gain_ratio:
            split_information = -sum((len(subset) / total_samples) * np.log2(len(subset) / total_samples)
                                     for subset in groups.values() if len(subset) > 0)
            if split_information != 0:
                goodness = information_gain / split_information
        else:
            goodness = information_gain
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return goodness, groups

    def calc_chi_statistic(self, groups):
        """
        Calculate the chi-square statistic and compare it to the chi-table to decide on pruning.

        Args:
        - groups: Dictionary of groups after split, where keys are feature values and values are subsets of data.

        Returns:
        - Whether the node should be split (True if yes, False if no).
        """
        # If there is only one group then there is no need to calculate the chi statistic, as no splitting is needed
        if len(groups) < 2:
            return False

        total_counts = np.array([sum(subset[:, -1] == 'e') for subset in groups.values()]).sum(), \
            np.array([sum(subset[:, -1] == 'p') for subset in groups.values()]).sum()

        total_samples = sum(total_counts)

        chi_stat = 0

        for group in groups.values():
            n_f = sum(group[:, -1] == 'e')
            p_f = sum(group[:, -1] == 'p')
            E_0 = (total_counts[0] / total_samples) * len(group)
            E_1 = (total_counts[1] / total_samples) * len(group)

            if E_0 > 0:
                chi_stat += ((n_f - E_0) ** 2) / E_0
            if E_1 > 0:
                chi_stat += ((p_f - E_1) ** 2) / E_1

        degrees_of_freedom = (len(groups) - 1) * (2 - 1)  # Because we only have two labels

        critical_value = chi_table[degrees_of_freedom][self.chi]
        return chi_stat > critical_value

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################

        # Cut if depth exceeds threshold or if group is pure
        if self.depth >= self.max_depth or len(np.unique(self.data[:, -1])) == 1:
            self.terminal = True
            return

        # Find best splitting feature
        best_goodness = -np.inf
        best_feature = -1
        best_groups = {}

        n_features = self.data.shape[1] - 1
        for feature in range(n_features):
            goodness, groups = self.goodness_of_split(feature)
            if goodness > best_goodness:
                best_goodness = goodness
                best_feature = feature
                best_groups = groups

        # Cut if no feature was found or if the selected feature didn't divide the data or if the goodness is too small
        if best_feature == -1 or len(best_groups) < 2 or best_goodness <= 1e-10:
            self.terminal = True
            return

        self.feature = best_feature
        self.calc_feature_importance(len(self.data))

        # Calculate Chi Square statistic if needed
        splitting_needed = True
        if self.chi < 1:
            splitting_needed = self.calc_chi_statistic(best_groups)

        # Create children nodes according to the best feature
        if splitting_needed:
            for feature_value, subset in best_groups.items():
                if len(subset) > 0:
                    child_node = DecisionNode(data=subset,
                                              impurity_func=self.impurity_func,
                                              depth=self.depth + 1,
                                              chi=self.chi,
                                              max_depth=self.max_depth,
                                              gain_ratio=self.gain_ratio)
                    self.add_child(child_node, feature_value)

        if not self.children:
            self.terminal = True
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################


class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the tree
        self.impurity_func = impurity_func  # the impurity function to be used in the tree
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio  #
        self.root = None  # the root node of the tree

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset.
        You are required to fully grow the tree until all leaves are pure
        or the goodness of split is 0.

        This function has no return value
        """
        ###########################################################################

        self.root = DecisionNode(self.data, self.impurity_func, max_depth=self.max_depth, chi=self.chi, gain_ratio=self.gain_ratio)
        self.root.split()

        queue = self.root.children.copy()

        while queue:

            child_node = queue.pop()
            child_node.split()
            queue += child_node.children.copy()

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, instance):
        """
        Predict a given instance

        Input:
        - instance: an row vector from the dataset. Note that the last element
                    of this vector is the label of the instance.

        Output: the prediction of the instance.
        """
        pred = None
        ###########################################################################
        node = self.root
        while not node.terminal:
            feature_value = instance[node.feature]
            found_children_with_value = False
            for i, value in enumerate(node.children_values):
                if value == feature_value:
                    node = node.children[i]
                    found_children_with_value = True
                    break
            if not found_children_with_value:
                break  # No matching child, prediction is based on current node

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset

        Input:
        - dataset: the dataset on which the accuracy is evaluated

        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        ###########################################################################
        correct = 0
        for row in dataset:
            if self.predict(row) == row[-1]:
                correct += 1
        accuracy = (correct / len(dataset)) * 100
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return accuracy

    def depth(self):
        def get_depth(node):
            if node.terminal or not node.children:
                return 0
            return 1 + max(get_depth(child) for child in node.children)
        self.root.depth = get_depth(self.root)

        return self.root.depth


def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy
    as a function of the max_depth.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output: the training and validation accuracies per max depth
    """
    training = []
    validation = []
    root = None

    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        ###########################################################################
        curr_tree = DecisionTree(X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=True)
        curr_tree.build_tree()

        train_accuracy = curr_tree.calc_accuracy(X_train)
        val_accuracy = curr_tree.calc_accuracy(X_validation)

        training.append(train_accuracy)
        validation.append(val_accuracy)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return training, validation


def chi_pruning(X_train, X_test):
    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]  # From no pruning to maximum pruning
    chi_training_acc = []
    chi_validation_acc = []
    depth = []

    ###########################################################################
    for chi in chi_values:
        curr_tree = DecisionTree(X_train, impurity_func=calc_entropy, gain_ratio=True,  chi=chi, max_depth=1000)
        curr_tree.build_tree()

        train_acc = curr_tree.calc_accuracy(X_train)
        val_acc = curr_tree.calc_accuracy(X_test)
        tree_depth = curr_tree.depth()

        chi_training_acc.append(train_acc)
        chi_validation_acc.append(val_acc)
        depth.append(tree_depth)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of node in the tree.
    """
    ###########################################################################
    if node is None:
        return 0
    n_nodes = 1  # Count this node
    for child in node.children:  # Assuming each node has a list of children
        n_nodes += count_nodes(child)  # Recursively count children

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






