import math
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def load(filename: str, delimiter: str) -> np.ndarray:
    """
    Takes in name of file assumed to be csv. The file can be deliminited with tab or comma returns np.ndarray of data

    :param filename: (str) either full or local path to data file
    :return: np.ndarray of data
    """
    data = pd.read_csv(filename, delimiter=delimiter)
    data = data.to_numpy()
    return data


class LeafNode:

    label = None

    def __init__(self, label):
        """
        takes in leaf node label, sets label

        :param label: (str) label/class of data in node
        """
        self.label = label

    # get_label is also in function in decision node
    # here is the "base case" essentially, takes advantage of common function names
    # between classes
    def get_label(self):
        """
        Getter method, returns leaf node's label

        :return: (int) class label/class prediction
        """
        return self.label


class DecisionNode:

    parent_data = np.ndarray
    child_nodes = []
    attributes = None
    label_column_name = None
    labels = None

    def __init__(self, parent_data: np.ndarray, attributes: dict, label_column_name: int, depth):
        """
        sets up all the necessary values of the given decision node

        :param parent_data: pd.DataFrame of all data in decision node (data to decide on)
        :param attributes: dict keys are remaining possible attributes to split on
        and the values in those keys contains information regarding the type of that attribute
        (continuous or categorical) and a np.array of all the unique possible values for that attribute

        :param label_column_name: (str) name of the label/class column name in the parent data
        :param labels: dict columns are label/class names and values in each column are unique
        possible values for that particular label/class
        """
        self.depth = depth
        self.parent_data = parent_data
        self.label_column_name = label_column_name
        self.attributes = attributes
        # child_nodes of optimal split at this point
        # child_nodes is a list of values
        # each value is either a tuple of the form: (attribute split on, category/value
        # of the attribute of this particular child (0,1, or 2 for this data set), data frame with only data where
        # the value under the given splitting attribute is the aforementioned value) OR
        # the element in child_nodes list is a leaf node with a corresponding label
        self.child_nodes = self.make_subtree(parent_data)
        # for house voting data set len(self.child_nodes) is either 3 if splitting attribute is found or
        # it is 1 if no splitting attribute

    # get predicted label based on tree
    def get_label(self, data_to_label: np.ndarray):
        """
        Recursively searches the tree for the label of a given data point (vector/pd.Series)

        :param data_to_label: pd.Series vector of data to label based on calculated tree
        :return: (str) predicted label/class name of data_to_label
        """
        for child_node in self.child_nodes:
            if isinstance(child_node, LeafNode):
                return child_node.get_label()
            attribute_splitting_on = child_node[0]
            if self.attributes[attribute_splitting_on]['type'] == 'cat':
                if data_to_label[attribute_splitting_on] == child_node[1]:
                    return child_node[2].get_label(data_to_label)
            elif self.attributes[attribute_splitting_on]['type'] == 'cont':
                less_than = child_node[3]
                if (less_than):
                    if data_to_label[attribute_splitting_on] < child_node[1]:
                        return child_node[2].get_label(data_to_label)
                    else:
                        continue
                else:
                    if data_to_label[attribute_splitting_on] >= child_node[1]:
                        return child_node[2].get_label(data_to_label)
                    else:
                        continue
        # this should not be reached
        print(data_to_label)
        print(self.child_nodes)
        print("No Label to be Found")
        assert False

    # calculate gini coefficient
    def gini_value(self, node_data: np.ndarray) -> float:
        """
        Calculates the gini value of a particular node (collection of data points)

        :param node_data: np.ndarray
        :return: gini value based on counts of classes
        """
        # total data points in child node
        total = node_data.shape[0]
        # one key for each possible y-value/label
        # for each possible label value make a dictionary where the value is the number of
        # data points with that label
        labels = dict()
        for label in node_data[:, self.label_column_name]:
            if label in labels.keys():
                labels[label] += 1
            else:
                labels[label] = 1
        # verify label counts are accurate
        check_sum = 0
        for key in labels.keys():
            # below test only for this dataset
            # assert(key in [0, 1])
            check_sum += labels[key]
        # total count of each labels should equal total number of data points in child
        assert (check_sum == total)
        gini = 0
        for label in labels.keys():
            # p_i = number of data points with label i over total number of data points in child
            p = labels[label] / total
            label_contribution = (p**2)
            gini += label_contribution
        gini = 1 - gini
        # verify gini is valid quantity
        assert (1 >= gini >= 0)
        return gini

    def info_gain(self, parent_df: np.ndarray, child_nodes: list[(str, int, np.ndarray, bool)], criterion: int) -> float:
        """
        Takes in parent decision node dataFrame and all child nodes np.ndarrays and returns information
        gain calculated according to the identified criterion

        :param parent_df: np.ndarray containing all data from parent node
        :param child_nodes: list[(str, int, np.ndarray, bool|None)]list of tuples, one for each child node, first elem is the attribute name
        being split on, second element is the value of the current branch with respect to that value,
        the third item is all the data such that for the attribute being split on (elem 0) the data
        has an appropriate value for that attribute based on the passed in value (elem 1), the last element indicates whether 
        it is a greater than or less than some threshold (elem 1) child node - if None the attribute is categorical and this is a mute
        point, if True then we have a less than child node, if false we have a greater than child node.

        :param criterion: (int) if 0, use entropy to calculate information gain, if 1 use gini coefficient
        :return: (float) information gain of current split
        """
        if criterion == 0:
            criterion = self.entropy
        if criterion == 1:
            criterion = self.gini_value
        parent_entropy = criterion(parent_df)
        num_data_points_in_parent = parent_df.shape[0]
        weighted_average_entropy = 0
        for child in child_nodes:
            child_df = child[2]
            num_data_point_in_child = child_df.shape[0]
            child_entropy_contribution = (
                num_data_point_in_child/num_data_points_in_parent)*criterion(child_df)
            weighted_average_entropy += child_entropy_contribution
        info_gain = parent_entropy - weighted_average_entropy
        # handle rounding errors for info gain
        if abs(info_gain - 0) <= 1e-09:
            info_gain = 0
        # verify info gain is a legal value
        assert(0 <= info_gain)
        return info_gain

    # calculate entropy
    def entropy(self, node_data: np.ndarray) -> float:
        """
        Calculates the entropy of a particular dataset based on the class values

        :param node_data: np.ndarray of data to find entropy of
        :return: entropy of passed in data
        """
        # total data points in child node
        total = node_data.shape[0]
        # one key for each possible y-value/label
        # for each possible label value make a dictionary where the value is the number of
        # data points with that label
        labels = dict()
        for label in node_data[:,self.label_column_name]:
            if label in labels.keys():
                labels[label] += 1
            else:
                labels[label] = 1
        # verify label counts are accurate
        check_sum = 0
        for key in labels.keys():
            # below test only for this dataset
            # assert(key in [0, 1])
            check_sum += labels[key]
        # total count of each labels should equal total number of data points in child
        assert(check_sum == total)
        entropy = 0
        for label in labels.keys():
            # p_i = number of data points with label i over total number of data points in child
            p = labels[label]/total
            # contribution is 0 if p = 0 (log_2(1) = 0)
            # if else catches case where p = 0
            label_contribution = (p)*math.log2(p if p != 0 else 1)          
            entropy -= label_contribution
        # verify entropy is valid quantity
        # assert(entropy >= 0)
        return entropy

    def make_subtree(self, parent_data: np.ndarray) -> list:
        """
        This is the main function which actually constructs the decision tree. It takes in some decision
        node's data (i.e. parent_data) and finds the best attribute to split on. Then either splits on
        that attribute or stops (if info gain is 0, if one of the attribute's branches is empty, or if all
        the labels are the same).

        :param parent_data: (pd.DataFrame)
        :return: (list) If we were able to split then the list will have a tuple for each possible attribute value, think of these tuples as the child nodes for each branch, they contain the actual child decision tree as well as some meta data. They will each be of the form (attribute, attribute_value, DecisionNode(parent_data[parent_data[attribute] = attribute_value], attributes = remaining attributes, etc.)). If we could not branch then we will return a list with one element of type LeafNode representing the majority label of the parent data (current decision node).
        """
        # get all the labels of the current parent data to see if they are all identical
        # assuming self.label_column_name is the only label column
        labels = parent_data[:,self.label_column_name]
        # assume output data is array of ints indicating class (e.g. I converted loan output for N, Y to 0, 1)
        labels = np.int_(labels)
        # if all labels are identical we can split no more
        # if there are no remaining attributes we can split no further
        if np.all(labels[0] == labels) or self.depth >= 8:
            # set the majority label
            label = np.bincount(labels).argmax()
            # no further child nodes, return a leaf node
            return [LeafNode(label)]
        # find optimal splitting attribute
        max_info_gain = -np.inf
        best_attribute = None
        best_child_nodes = None

        # test subset of attributes for info gain
        num_attributes = len(list(self.attributes.keys()))
        num_attributes_to_get = math.ceil(math.sqrt(num_attributes))
        test_attributes = list(self.attributes.keys())
        # sample sqrt(len(attributes)) unique attributes to test
        sampled_attribute_indices = random.sample(range(num_attributes),num_attributes_to_get)
        attributes_to_test = []
        for i in sampled_attribute_indices:
            attributes_to_test.append(test_attributes[i])
            
        # loop over selected attributes 
        for attribute in attributes_to_test:
            # keep track of child nodes (data for potential child branches)
            child_nodes = []
            # if attribute is continuous
            if self.attributes[attribute]['type'] == 'cont':
                # implement functionality for handling continuous type attributes later
                for split_val in self.attributes[attribute]['legal_values']:
                    # get the data such that all rows are in the same category under the current attribute
                    df_greater_equal = parent_data[parent_data[:,attribute] >= split_val]
                    df_less_than = parent_data[parent_data[:,attribute] < split_val]
                    # so long as the data in the respective attribute and category yield a non_zero number of vectors
                    # test each possible set of child nodes (one test for each possible split value)
                    child_nodes = []
                    if df_greater_equal.shape[0] >= 3 and df_less_than.shape[0] >= 3:
                        # keep track of attribute being split on, the current category
                        # of that attribute, and the data that has that attribute and current category
                        child_nodes.append((attribute, split_val, df_greater_equal, False))
                        child_nodes.append((attribute, split_val, df_less_than, True))
                        # criterion = 0, split based on entropy
                        # criterion = 1, split based on gini
                        # search for criterion
                        criterion = 0
                        info_gain = self.info_gain(parent_data, child_nodes, criterion)
                        if info_gain >= max_info_gain:
                            max_info_gain = info_gain
                            best_attribute = attribute
                            best_child_nodes = child_nodes
            # if attribute is categorical
            elif self.attributes[attribute]['type'] == 'cat':
                # handle categorically valued attribute data
                # loop thru each possible category for a given attribute
                none_empty = True
                for category in self.attributes[attribute]['legal_values']:
                    # get the data such that all rows are in the same category under the current attribute
                    df_to_add = parent_data[parent_data[:,attribute] == category]
                    # so long as the data in the respective attribute and category yield a non_zero number of vectors
                    if df_to_add.shape[0] > 0:
                        # keep track of attribute being split on, the current category
                        # of that attribute, and the data that has that attribute and current category
                        child_nodes.append((attribute, category, df_to_add, None))
                    else:
                        none_empty = False
                # criterion = 0, split based on entropy
                # criterion = 1, split based on gini
                if(not none_empty):
                    continue
                criterion = 0
                info_gain = self.info_gain(parent_data, child_nodes, criterion)
                if info_gain >= max_info_gain and info_gain > 0:
                    max_info_gain = info_gain
                    best_attribute = attribute
                    best_child_nodes = child_nodes
        # if no further splitting can be done to improve model (entropy = 0) or training data doesn't have all the
        # possible values for the splitting attribute (one branch is empty) then return a leaf node
        if best_attribute is None or len(best_child_nodes) == 0:
            label = np.bincount(labels).argmax()
            return [LeafNode(label)]
        # if we reach this point we can perform a legal split and we make each child node into a decision tree from
        # which we can do more splitting
        self.depth += 1
        children_to_return = [(child[0], child[1], DecisionNode(parent_data=child[2], attributes=self.attributes,
                                                                label_column_name=self.label_column_name, depth=self.depth), child[3]) for child in best_child_nodes]
        return children_to_return


class DecisionTree:

    def __init__(self, label_column_name: int, training_data: np.ndarray, test_data: np.ndarray, attributes: dict):
        """
        Initializes parameters for decision tree

        :param all_data: np.ndarray of data on which to create decision tree
        :param label_column_name: index of the column holding the label/class data
        """
        self.training_df = training_data
        self.test_df = test_data
        self.label_column_name = label_column_name
        # Dictionary of attributes with their properites (type, legal split values)
        self.attributes = attributes
        # Works recursively down in decision node
        self.root = DecisionNode(parent_data=self.training_df, label_column_name=label_column_name,
                                 attributes=self.attributes, depth=1)

    def make_decision(self, data_to_label: np.ndarray) -> int:
        """
        Makes a prediction about a particular data series (vector)

        :param data_to_label: np.ndarray of data we want to label based on our tree
        :return: (int) predicted label value (assume classes labeled with ints)
        """
        return self.root.get_label(data_to_label)

# def test_tree(tree: DecisionTree, data_to_test_on_indicator: int) -> float:
#         """
#         Function to get the accuracy of the tree based on either training data or test data

#         :param data_to_test_on_indicator: If 0, get accuracy of tree on training data. If 1, get accuracy of tree on test data.
#         :return: (float) accuracy of based on aforementioned input
#         """
#         data_to_test_on = None
#         if data_to_test_on_indicator == 0:
#             data_to_test_on = tree.training_df
#         if data_to_test_on_indicator == 1:
#             data_to_test_on = tree.test_df
#         num_correct = 0
#         guesses = 0
#         for i in range(len(data_to_test_on.axes[0])):
#             current_row = data_to_test_on.loc[i, :]
#             decision = tree.make_decision(current_row)
#             if current_row[tree.label_column_name] == decision:
#                 num_correct += 1
#             guesses += 1
#         return num_correct/guesses

def get_attributes_and_categories(all_data: np.ndarray, label_column_name: int):
        """
        Creates a dictionary for labels to indicate label type and legal values among all data. Creates a
        dictionary for all attributes giving attributes a type (categorical or continuous) and an array of
        each attributes legal values as well (according to all data)

        :param all_data: data to create tree on
        :return: dictionary of attributes of form, attributes[attribute] = {'type' : 'cat'|'cont', 'legal_values' : np.array}
        """
        # ask if all data is continous
        cont_response = input("Answer 'yes' or 'no', are all attributes made of continous data?")
        invalid_response = True
        cont = False
        while(invalid_response):
            if cont_response != 'yes' and cont_response != 'no':
                cont_response = input("Please answer with 'yes' or 'no', are all attributes made of continous data?")
                continue
            else:
                if cont_response == 'yes':
                    cont = True
                else:
                    cont = False
                invalid_response = False

        # if not, ask if all data is categorical
        cat = False
        if not cont:
            cont_response = input("Answer 'yes' or 'no', are all attributes made of categorical data?")
            invalid_response = True
            cont = False
            while(invalid_response):
                if cont_response != 'yes' and cont_response != 'no':
                    cont_response= input("Please answer with 'yes' or 'no', are all attributes made of categorical data?")
                    continue
                else:
                    if cont_response == 'yes':
                        cat = True
                    else:
                        cat = False
                    invalid_response = False

        # assume final column is target/label/y values
        # attributes[label_column_name] will yield the target/y/label information
        attributes = dict()
        # keep track of valid label values as well, in a data structure akin to
        # the one used for attributes for consistency
        labels = dict()
        # assume there is one label column with the name self.label_column_name
        labels[label_column_name] = dict()
        labels[label_column_name]['type'] = 'cat'
        labels[label_column_name]['legal_values'] = np.unique(all_data[:,label_column_name])
        labels = labels
        # now work on attributes
        attribute_columns = list(range(0, all_data.shape[1]))
        attribute_columns.remove(label_column_name)
        # get sqrt(m) attributes, where m is the total number of attributes originally
        for i in range(len(attribute_columns)):
            col = attribute_columns[i]
            attributes[col] = dict()
            # each attribute is either categorical ('cat') or continuous ('cont')
            # if cat is true, all attributes are categorical
            if cat:
                attributes[col]['type'] = 'cat'

            # cont is true, all attributes are categorical
            elif cont:
                attributes[col]['type'] = 'cont'

            # if neither cont nor cat is true that means there is a mix of categorical and continous
            # attributes and so we need the user to specific which is which
            else:
                not_done = True
                while(not_done):
                    attr_type = input(
                        f"Please enter the type of attribute {col} (if continous enter cont if categorical enter cat): ")
                    if attr_type != "cont" and attr_type != "cat":
                        print(
                            "Please enter 'cont' for continous attributes or 'cat' for categorical attributes")
                        continue
                    attributes[col]['type'] = attr_type
                    not_done = False

            # each attribute has either legal categories or legal thresholds on which to perform splits

            # for categorical, the splitting values will be all unique categories seen
            if attributes[col]['type'] == 'cat':
                attributes[col]['legal_values'] = np.unique(all_data[:,col])

            # for continous values, the splitting numbers will be the averages between all data points
            if attributes[col]['type'] == 'cont':
                averages = np.array([])
                sorted_values =  np.unique(all_data[all_data[:,col].argsort()][:, col])
                for i in range(0, sorted_values.shape[0]-1):
                    avg = (sorted_values[i] + sorted_values[i+1])/2
                    averages = np.append(averages, avg)
                attributes[col]['legal_values'] = averages
        return attributes

class RandomForest:    
    def __init__(self, n: int, label_column_name: int, training_data: np.ndarray, attributes: dict):
        """Generate n trees, giving each bootstrap data that is sampled with replacement from the original
        training data.

        Args:
            n (int): number of trees in forest
            label_column_name (int): column index of label
            training_data (np.ndarray): data to train on
            attributes (dict): dictionary with information about attributes (format described in get_attributes_and_categories)
        """
        self.trees = []
        self.label_column_name = label_column_name
        data_size = training_data.shape[0]
        for i in tqdm(range(n)):
            # sample data with replacement (bootstrapping)
            indicies_to_get = np.random.choice(data_size, size=data_size)
            cur_training_data = training_data[indicies_to_get]
            tree = DecisionTree(label_column_name, cur_training_data, test_data=None, attributes=attributes)
            self.trees.append(tree)
    
    def get_trees(self):
        """Returns a list of all trees (Decision tree objects) in the random forest.

        Returns:
            list: list of DecisionTree objects (all trees in forest)
        """
        return self.trees

    def get_decision(self, data_to_label: np.ndarray):
        """Takes in data to label, then finds the majority label (most common label) among
        all the trees in the random forest and returns that label

        Args:
            data_to_label (np.ndarray): Data/row we are trying to label

        Returns:
            _type_ : label
        """
        classes_choice = dict()
        most_votes = -np.inf
        most_popular_choice = -1
        # get vote from each tree, take vote which the most trees vote for
        for tree in self.trees:
            cur_decision = tree.make_decision(data_to_label)
            if cur_decision in classes_choice:
                classes_choice[cur_decision] += 1
            else:
                classes_choice[cur_decision] = 1
            if classes_choice[cur_decision] > most_votes:
                most_votes = classes_choice[cur_decision]
                most_popular_choice = cur_decision
        return most_popular_choice
    
    def evaluate(self, test_data: np.ndarray, binary=False):
        """Returns statistics regarding the efficacy of the random forest

        Args:
            test_data (np.ndarray): data we are testing (many rows)
            binary (bool): if positive then we are handling the binary labeled data and I want 
            to not calculate accuracy, precision, recall for each individual class

        Returns:
            tuple(float, float, float, float): accuracy, precision, recall, f1_score  
        """
        # evaluate random forest using test_data
        # assume test_data is the last fold from a given stratified cross fold validation set
        precision = 0
        accuracy = 0
        recall = 0
        # keys are 'true_class,predicted_class' of type 'int,int'
        confusion_matrix = dict()
        classes = np.unique(test_data[:,self.label_column_name])
        num_instances = test_data.shape[0]
        num_classes = classes.shape[0]
        classes = [int(_class) for _class in classes]
        for true_class in classes:
            for predicted_class in classes:
                confusion_matrix[f"{true_class},{predicted_class}"] = 0

        # inner function to build confusion matrix using numpy .apply function
        def build_confusion_matrix(instance):
            predicted_class = int(self.get_decision(instance))
            true_class = int(instance[self.label_column_name])
            confusion_matrix[f"{true_class},{predicted_class}"] += 1

        # build confusion matrix
        np.apply_along_axis(build_confusion_matrix, 1, test_data)

        # handle easy case with 2 classes for cancer in particular
        if (binary and len(classes)==2):
            classes = np.sort(classes)
            class_1 = classes[1]
            class_0 =classes[0]
            #         class_1 class_0
            # class_1   TP       FN
            # class_0   FP       TN
            #                                     row      column
            true_positive = confusion_matrix[f"{class_1},{class_1}"]
            false_positive = confusion_matrix[f"{class_0},{class_1}"]
            false_negative = confusion_matrix[f"{class_1},{class_0}"]
            true_negative = confusion_matrix[f"{class_0},{class_0}"]
            accuracy = (true_positive + true_negative)/(true_positive+false_positive+false_negative+true_negative)
            precision = (true_positive/(true_positive+false_positive))
            recall = (true_positive/(true_positive+false_negative))
            f1_score = (2*(precision*recall))/(precision + recall)
            return accuracy, precision, recall, f1_score

        # loop over all classes to desired statistics (averaging at the end)
        for true_class in classes:
            true_positive = 0
            false_positive = 0
            false_negative = 0
            for predicted_class in classes:
                if predicted_class == true_class:
                    #         class_0 class_1 class_2 ...
                    # class_0    x       y        z   ...
                    # class_1    w       v        t   ...
                    # class_2    a       b        c   ...
                    # ....      ...     ...      ...  .
                    #                                  .
                    # 
                    #                                       row            column
                    true_positive += confusion_matrix[f"{true_class},{predicted_class}"]
                    accuracy += true_positive
                else:
                    false_negative += confusion_matrix[f"{true_class},{predicted_class}"]
                    false_positive += confusion_matrix[f"{predicted_class},{true_class}"]
            try:
                precision += (true_positive/(true_positive+false_positive))
                recall += (true_positive/(true_positive+false_negative))
            except ZeroDivisionError:
                continue
        # accuracy is just the sum of diagonals of total entries
        accuracy = accuracy/num_instances
        # average precision and recall over all classes
        precision = precision/num_classes
        recall = recall/num_classes
        # f1 score from class
        f1_score = (2*(precision*recall))/(precision + recall)
        return accuracy, precision, recall, f1_score

def stratified_cross_fold(k: int, data: np.ndarray, class_column_name: int) -> list:
    """Create k folds with correct proportions of each labels in each. Return a list of 
    such folds.

    Args:
        k (int): number of folds to create
        data (np.ndarray): all data to use to create folds of
        class_column_name (int): integer of column index for label

    Returns:
        list: returns list of folds
    """
    np.random.shuffle(data)
    classes = np.unique(data[:,class_column_name])
    # get data split into classes
    split_data = dict()
    split_data_indices = dict()
    for _class in classes:
        split_data[_class] = data[data[:,class_column_name] == _class]
    # get 1/k * amount of data in each class to put into k folds (to maintain proportions)
    folds = []
    for i in range(k):
        for _class in split_data.keys():
            if len(folds) != i:
                folds[i] = np.concatenate((folds[i],np.array_split(split_data[_class], k)[i]), axis=0)
            else:
                folds.append(np.array_split(split_data[_class], k)[i])
    return folds

def rf_plot(df: np.ndarray, label_column_name: int):
    """This is just the main function that creates all random forests and plots the desired statistics.
    
    Args:
        df (np.ndarray): all data
        label_column_name (int): column index of where labels are
    """
    attributes = get_attributes_and_categories(df, label_column_name=label_column_name)
    num_folds = 10
    folds = stratified_cross_fold(num_folds, df, label_column_name)
    num_trees = [1, 5, 10, 20, 30, 40, 50]
    scores = []
    for n_trees in num_trees:
        accuracy = 0
        precision = 0
        recall = 0
        f1_score = 0
        num_folds = len(folds)
        for i, fold in enumerate(folds):
            test = folds[i]
            training = None
            num_folds_i = 0
            for j in range(len(folds)):
                if j == i:
                    continue
                try:
                    training = np.concatenate((training, folds[j]), axis=0)
                except ValueError:
                    training = folds[j]
                num_folds_i += 1
            random_forest = RandomForest(n_trees, label_column_name, training, attributes)
            t_accuracy, t_precision, t_recall, t_f1_score = random_forest.evaluate(test)
            accuracy += t_accuracy
            precision += t_precision
            recall += t_recall
            f1_score += t_f1_score
        # Since we recorded test scores with test data as each fold (and training data as other folds), we need to divide results by the number of folds.
        accuracy = accuracy/num_folds
        precision = precision/num_folds
        recall = recall/num_folds
        f1_score = f1_score/num_folds
        scores.append((n_trees, accuracy, precision, recall, f1_score))
    
    # print scores, form is list of (ntrees, accuracy, precision, recall, f1 score) tuples
    print(scores)

    # setup x, y data for graphs
    x_values = np.array(num_trees)
    accuracies = np.array([score[1] for score in scores])
    precision = np.array([score[2] for score in scores])
    recall = np.array([score[3] for score in scores])
    f1_score = np.array([score[4] for score in scores])
    accuracy_std = np.std(accuracies)
    precision_std = np.std(precision)
    recall_std = np.std(recall)
    f1_score_std = np.std(f1_score)

    # accuracy
    plt.errorbar(x_values, accuracies, yerr=accuracy_std)
    plt.xlabel(f"Number of Trees in Forest (ntrees)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Trees")
    plt.show()

    # precision
    plt.errorbar(x_values, precision, yerr=precision_std)
    plt.xlabel(f"Number of Trees in Forest (ntrees)")
    plt.ylabel("Precision")
    plt.title("Precision vs Number of Trees")
    plt.show()

    # recall
    plt.errorbar(x_values, recall, yerr=recall_std)
    plt.xlabel(f"Number of Trees in Forest (ntrees)")
    plt.ylabel("Recall")
    plt.title("Recall vs Number of Trees")
    plt.show()

    # f1
    plt.errorbar(x_values, f1_score, yerr=f1_score_std)
    plt.xlabel(f"Number of Trees in Forest (ntrees)")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Number of Trees")
    plt.show()

    # all metrics
    plt.plot(x_values, accuracies, label="Accuracy")
    plt.plot(x_values, precision, label="Precision")
    plt.plot(x_values, recall, label="Recall")
    plt.plot(x_values, f1_score, label="F1 Score")
    plt.xlabel("Number of Trees in Forest (ntrees)")
    plt.ylabel("Percentage")
    plt.legend()
    plt.title("All Metrics vs Number of Trees")
    plt.show()



def rf_main(data_set):
    # uncomment to run code on titanic dataset
    df = load(data_set, ',')
    label_column_name = 0

    # uncomment to run code on loan dataset
    # df = load("datasets/hw3_wine.csv", '\t')
    # label_column_name = 0

    # uncomment to run code on parkinsons dataset
    # df = load("datasets/hw3_cancer.csv", '\t')
    # label_column_name = 9

    # df = load("datasets/cmc.csv", ',')
    # label_column_name = 9

    # plot data/do the hw
    rf_plot(df, label_column_name)
