import math
import numpy as np
import matplotlib.pyplot as plt


# load data from iris.csv file into numpy arrays
def load(filename: str) -> np.array:
    """
    create np.array multi dimensional of data in file, assume file is .csv

    :param filename: (str) name of file data is in
    :return: np.array of data
    """
    # assume file is of type csv
    # reviewing the iris.csv file we see no column names
    data = np.genfromtxt(filename, delimiter=',', dtype=None, encoding=None)
    data = np.array(data.tolist())
    return data


def preprocess(data: np.array):
    """
    Randomly shuffles data, splits into training and testing, normalizes, splits into data and labels

    :param data: all data to process
    :return: training_data, training_labels, test_data, test_labels
    """
    # shuffle data randomly (done before splitting)
    np.random.shuffle(data)
    # ASSUME last column is labels ************** This may not be true
    labels = data[:, -1:]
    # all other columns are features/attributes of type float
    data = data[:, :-1]
    # convert data to float data type
    data = data.astype(float)
    # good idea to look at data to make sure its clean
    # no empty rows, etc...
    # Split the data into training and testing sets
    num_data_points = np.shape(data)[0]
    # find an index to cutoff training and testing data
    # this is random because the data is already randomly shuffled around
    cutoff_index = math.floor(num_data_points*.8)
    training_data = data[:cutoff_index]
    training_labels = (labels[:cutoff_index])[:, 0]
    test_data = data[cutoff_index:]
    test_labels = labels[cutoff_index:][:, 0]
    # normalize data
    # use max and min from training data only (to prevent leakage of the test data
    # into the training data)
    maxes = np.max(training_data, axis=0)
    mins = np.min(training_data, axis=0)
    # normalize accordingly
    training_data = normalize_features(training_data, maxes=maxes, mins=mins)
    test_data = normalize_features(test_data, maxes=maxes, mins=mins)
    # type checking
    assert(type(training_data) == type(training_labels) == type(test_data) == type(test_labels))
    return training_data, training_labels, test_data, test_labels


# normalize values with function, where x is a column in data, (x - min(x))/(max(x)-min(x))
def normalize_features(data: np.ndarray, maxes: float, mins: float) -> np.ndarray:
    """
    Normalize features based on training data (inputted max, min values), according to function (x - min)/(max - min)

    :param data: data to normalize
    :param maxes: max of every feature from training data
    :param mins: min of every feature from training data
    :return: np.ndarray same data as passed in just normalized
    """
    if maxes is None:
        print("max should be from training data only")
        maxes = np.max(data, axis=0)
    if mins is None:
        print("min should be from training data only")
        mins = np.min(data, axis=0)
    # verify number of columns is same across all
    assert(len(mins) == len(maxes) == np.shape(data)[1])
    # use (x - min)/(max - min) normalization formula
    max_minus_min= maxes - mins
    data = data - mins
    data = data / max_minus_min
    return data


# do knn for passed in vector based on training data
def knn(v: np.ndarray, training_data: np.ndarray, k: int):
    """
    Find knn on vector v comparing it to k nearest training data vectors, return list of k nearest neighbors

    :param v: np.ndarray - vector/data instance to label
    :param training_data: np.ndarray - data used to get the k-nearest neighbors of v to
    :param k: int - k parameter (how many neighbors to check), verified to not be odd by train function
    :return: list of k nearest neighbors of form [(index of neighbor, distance),...]
    """
    # make sure cols/rows line up
    assert(len(v) == np.shape(training_data)[1])
    distances = []
    for index, row in enumerate(training_data):
        distances.append((index, euclidean_distance(row, v)))
    distances.sort(key = lambda x: x[1])
    return distances[:k]


# returns the euclidean distance between vectors v1 and v2
def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate euclidean distance between two points

    :param v1: np.ndarray vector 1
    :param v2: np.ndarray vector 2
    :return: (float) -> distance
    """
    return np.sqrt(np.subtract(v1, v2).dot(np.subtract(v1, v2)))


# pass in array of labels/classes and return an array of the unique ones
def get_labels(labels: np.ndarray) -> np.ndarray:
    """
    returns all unique label values

    :param labels: array of all labels
    :return: all unique labels
    """
    return np.unique(labels)
        

# train and report accuracy on within training data
def train(training_data: np.ndarray, training_labels: np.ndarray, data_to_test: np.ndarray,
          data_to_test_labels: np.ndarray, labels=None, k=3, binary=False) -> float:
    """
    Run knn on each instance in data_to_test based on training data and return the accuracy.

    :param training_data: np.ndarray data to train on
    :param training_labels: np.ndarray used to get labels of k nearest neighbors
    :param data_to_test: np.ndarray data to test knn on
    :param data_to_test_labels: np.ndarray used to get true labels of test data to compare to prediction
    :param labels: (np.ndarray)  array of all possible label values (classes)
    :param k: (int) hyper parameter, how many neighbors to look at
    :return: (float) accuracy of knn on passed in data (success/total)
    """
    # evaluate random forest using test_data
    # assume test_data is the last fold from a given stratified cross fold validation set

    # need valid k
    if k == 0 or k > (len(training_data) - 1):
        print("k must be greater than 0 but less than the number of training data points")
        return -1

    if k % 2 == 0:
        print("k must be odd to avoid ties")
        return -1

    # setup labels if not preset
    if labels is None:
        labels = get_labels(training_labels)

    # ensure only valid labels are used
    valid_labels = dict()
    for label in labels:
        valid_labels[label] = 0

    precision = 0
    accuracy = 0
    recall = 0
    # keys are 'true_class,predicted_class' of type 'int,int'
    confusion_matrix = dict()
    for true_class in valid_labels:
        for predicted_class in valid_labels:
            confusion_matrix[f"{true_class},{predicted_class}"] = 0

    # maintain number of correct guesses and total guesses to perform accuracy calculation
    correct_guesses = 0
    total_guesses = 0
    for index, row in enumerate(data_to_test):
        # get top k vectors/rows in form (index, distance)
        # only compare current vector only to training data
        top_k = knn(row, training_data, k)
        predicted_label_count = 0
        predicted_label = None
        # loop through each k row and find predicted/majority label
        # ensure starting counts are all 0
        for label in labels:
            valid_labels[label] = 0


        for data_point in top_k:
            # the index of the current data point out of the k data points is that current data point's (a tuple)
            # first index
            label_index = data_point[0]
            # ensure only valid labels are used
            try:
                # increase label counts and see if the predicted/majority holder has changed
                valid_labels[training_labels[label_index,0]] += 1
                if valid_labels[training_labels[label_index,0]] > predicted_label_count:
                    predicted_label_count = valid_labels[training_labels[label_index,0]]
                    predicted_label = training_labels[label_index,0]
            except KeyError:
                print("invalid label")
        # make sure vote count is legal (number of total votes == num data points in top_k (i.e. k))
        check_sum = sum(valid_labels.values())
        assert(check_sum == k)
        predicted_class = predicted_label
        true_class = data_to_test_labels[index][0]
        confusion_matrix[f"{true_class},{predicted_class}"] += 1
        # check if guess was accurate
        if predicted_label == data_to_test_labels[index]:
            correct_guesses += 1
        total_guesses += 1

    # get statistics from confusion matrix 
    # handle easy case with 2 classes for cancer in particular
    if (binary and len(valid_labels)==2):
        valid_labels = np.sort(valid_labels)
        class_1 = valid_labels[1]
        class_0 =valid_labels[0]
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

    # loop over all classes/labels to desired statistics (averaging at the end)
    for true_class in valid_labels:
        true_positive = 0
        false_positive = 0
        false_negative = 0
        for predicted_class in valid_labels:
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
    num_instances = data_to_test.shape[0]
    num_classes = len(valid_labels)
    # accuracy is just the sum of diagonals of total entries
    accuracy = accuracy/num_instances
    # average precision and recall over all classes
    precision = precision/num_classes
    recall = recall/num_classes
    # f1 score from class
    f1_score = (2*(precision*recall))/(precision + recall)
    accuracy_1 = correct_guesses/total_guesses
    return accuracy_1, accuracy, precision, recall, f1_score


def plot_accuracy(data: np.ndarray):
    """
    Plot accuracy of knn for values of k odd from 1 to 51, on both training and test data

    :param data: np.ndarray all data to work with (create training, test data from)
    :return: (int) - 1 on success
    """

    # on training_data
    training_accuracy = []
    for k in range(1, 52, 2):
        k_training_accuracy = []
        for i in range(20):
            training_data, training_labels, test_data, test_labels = preprocess(data)
            k_training_accuracy.append(train(training_data, training_labels, training_data, training_labels, k=k))
        training_accuracy.append(k_training_accuracy)

    # on test data
    test_accuracy = []
    for k in range(1, 52, 2):
        k_test_accuracy = []
        for i in range(20):
            training_data, training_labels, test_data, test_labels = preprocess(data)
            k_test_accuracy.append(train(training_data, training_labels, test_data, test_labels, k=k))
        test_accuracy.append(k_test_accuracy)

    # plot based on training data
    fig, ax = plt.subplots(2)
    x_training = np.arange(1, 52, 2)
    y_training = [np.mean(accuracies) for accuracies in training_accuracy]
    y_training_std = [np.std(accuracies) for accuracies in training_accuracy]
    ax[0].errorbar(x_training, y_training, y_training_std)
    ax[0].set_title("Accuracy on Training Data")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("Accuracy on Training Data")

    # plot based on test data
    x_test = np.arange(1, 52, 2)
    y_test = [np.mean(accuracies) for accuracies in test_accuracy]
    y_test_std = [np.std(accuracies) for accuracies in test_accuracy]
    ax[1].errorbar(x_test, y_test, y_test_std)
    ax[1].set_title("Accuracy on Test Data")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Accuracy on Test Data")

    plt.tight_layout()
    plt.show()
    return 1


# main function
def main():
    file = "data/iris.csv"
    data = load(file)
    plot_accuracy(data)


if __name__ == '__main__':
    main()