import neural_networks as nn
import RandomForest as rf
import knn as knn
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

def get_mnist():
    digits = datasets.load_digits(return_X_y=True)
    digits_dataset_X = digits[0]
    digits_dataset_Y = digits[1]
    # N = len(digits_dataset_X)
    # digit_to_show = np.random.choice(range(N), 1)[0]
    # print(digits_dataset_X)
    # print(digits_dataset_Y[digit_to_show])
    # plt.imshow(np.reshape(digits_dataset_X[digit_to_show], (8,8)))
    # plt.show()
    return digits_dataset_X, digits_dataset_Y


def load(filename: str, delimiter: str) -> np.ndarray:
    """
    Takes in name of file assumed to be csv. The file can be deliminited with tab or comma returns np.ndarray of data

    :param filename: (str) either full or local path to data file
    :return: np.ndarray of data
    """
    data = pd.read_csv(filename, delimiter=delimiter)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    data = data.to_numpy()
    return data

def run_neural_net(inputs, outputs, hidden_layers, learning_rate, lambda_param, delta=1e-9, max_training_runs=15000, categorical_output=True):
    input, output = shuffle(inputs, outputs)
    lambda_param = lambda_param
    learning_rate = learning_rate
    layers = np.append(np.array([input.shape[1]]),hidden_layers)
    input_folds, output_folds = nn.run_cross_fold_validation(input, output, 10)
    # folds are dictionary labeled 1,...,num_folds
    if categorical_output:
        output_folds = nn.OneHot(output_folds=output_folds)
    layers = np.append(layers,np.array([output_folds[1].shape[1]]))
    # cross fold handled in neural net function
    nn.train_test(input_folds, output_folds, learning_rate, layers, lambda_param, delta=delta, max_training_runs=max_training_runs)

def run_decision_tree(dataset, label_column):
    # cross fold handled in random forest rf_plot function
    rf.rf_plot(dataset, label_column)

def run_knn(input, output, k):
    # do cross fold for knn, cross fold wasn't part of original function
    input, output = shuffle(input, output)
    input_folds, output_folds = nn.run_cross_fold_validation(input, output, 10)
    num_folds = len(input_folds)
    accuracy = 0
    averaged_f1_score = 0
    for fold in range(len(input_folds.keys())):
        testing_input = input_folds[fold+1]
        testing_output = output_folds[fold+1]
        # construct training input and output
        for i in range(len(input_folds)):
            if i == fold:
                continue
            else:
                try:
                    training_input = np.concatenate(training_input, input_folds[i+1])
                    training_output = np.concatenate(training_output, output_folds[i+1])
                except:
                    training_input = input_folds[i+1]
                    training_output = output_folds[i+1]
        cur_accuracy1, cur_accuracy, precision, recall, f1_score = knn.train(training_input, training_output, testing_input, testing_output, k=k)
        accuracy += cur_accuracy
        averaged_f1_score += f1_score
        # plt.plot(num_instances, loss)
        # plt.show()
        # plt.plot(statst_instances, accuracies)
        # plt.show()
        # plt.plot(statst_instances, f1_scores)
        # plt.show()
    accuracy = accuracy/num_folds
    averaged_f1_score = averaged_f1_score/num_folds
    print(f"Average Accuracy: {accuracy}")
    print(f"Average F1-Score: {averaged_f1_score}")
    return accuracy, f1_score

def process_titanic():
    # get titanic dataset
    titanic_df = load("titanic.csv", ",")
    titanic_label_column = 0
    titanic_input = titanic_df[:, range(1, titanic_df.shape[1])]
    old_column_num = titanic_input.shape[1]-1
    # column indices 4 and on are numerical, so that is column indices 3 and on in input data array - normalize them
    titanic_input_numerical = titanic_input[:, 3:]
    titanic_input_numerical = nn.normalize_data(titanic_input_numerical)
    titanic_input = titanic_input[:, :3]
    titanic_input = np.append(titanic_input, titanic_input_numerical, axis=1)
    titanic_output = titanic_df[:, 0]
    # delete name column
    titanic_input = np.delete(titanic_input, 1, axis=1)
    # make full data include normalized input
    titanic_df = np.append(titanic_input, titanic_output[:, np.newaxis], axis=1)
    assert(old_column_num == titanic_input.shape[1])
    return titanic_df, titanic_input, titanic_output, titanic_label_column

def process_loans():
    loan_df = load("loan.csv", ",")
    # drop loan id column
    loan_df = loan_df[:, 1:]
    old_loan_input = loan_df[:, range(0, loan_df.shape[1]-1)]
    old_column_num = old_loan_input.shape[1]
    # column indices 6-9 inclusive are numerical - normalize them
    loan_input_numerical = old_loan_input[:, [5,6,7,8]]
    loan_input_numerical = nn.normalize_data(loan_input_numerical)
    loan_input = old_loan_input
    loan_input[:, [5,6,7,8]] = loan_input_numerical.astype(float)
    assert(old_column_num == loan_input.shape[1])
    loan_output = loan_df[:, -1]
    # make full data include normalized input
    loan_df = np.append(loan_input, loan_output[:, np.newaxis], axis=1)
    loan_label_column = loan_df.shape[1] - 1
    return loan_df, loan_input, loan_output, loan_label_column

def process_parkinsons():
    parkinsons_df = load("parkinsons.csv", ",")
    parkinsons_label_column = parkinsons_df.shape[1] - 1
    parkinsons_input = parkinsons_df[:, range(0, parkinsons_df.shape[1]-1)]
    parkinsons_input = nn.normalize_data(parkinsons_input)
    parkinsons_output = parkinsons_df[:, -1]
    # make full data include normalized input
    parkinsons_df = np.append(parkinsons_input, parkinsons_output[:, np.newaxis], axis=1)
    return parkinsons_df, parkinsons_input, parkinsons_output, parkinsons_label_column

def process_mnist():
    mnist_input, mnist_output = get_mnist()
    mnist_input, mnist_output = shuffle(mnist_input, mnist_output)
    mnist_input = nn.normalize_data(mnist_input)
    mnist_df = np.append(mnist_input, mnist_output[:, np.newaxis], axis=1)
    msnist_label_column = mnist_df.shape[1]-1
    return mnist_df, mnist_input, mnist_output, msnist_label_column

def main():
# ********************************************************
    delta = 1e-10
    max_training_runs = 15000
    # get mnist dataset
    mnist_df, mnist_input, mnist_output, mnist_label_column = process_mnist()
    # print(mnist_input.shape)

    # run neural network on mnist
    # arch 1 **************************
    # layers = [6, 4]
    # learning_rate = 0.0001
    # lambda_param = 0.5
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(mnist_input, mnist_output, layers, learning_rate, lambda_param)
    # accuracy: 0.102702, f1-score: 0.018627
    # arch 2 **************************
    # layers = [8]
    # learning_rate = 0.1
    # lambda_param = 0.5
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(mnist_input, mnist_output, layers, learning_rate, lambda_param, delta, max_training_runs)
    # f1-score: 0.9152965245007867
    # arch 3 *****************************
    # layers = [20, 8]
    # learning_rate = 0.5
    # lambda_param = 0.25
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(mnist_input, mnist_output, layers, learning_rate, lambda_param, delta, max_training_runs)
    # arch 4 *****************************
    # layers = [20, 8]
    # learning_rate = 1
    # lambda_param = 1
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(mnist_input, mnist_output, layers, learning_rate, lambda_param, delta, max_training_runs)
    # arch 4 *****************************
    # layers = [10]
    # learning_rate = 1
    # lambda_param = 0.5
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(mnist_input, mnist_output, layers, learning_rate, lambda_param, delta, max_training_runs)

    # run descision tree on mnist
    # numerical columns = all columns
    run_decision_tree(mnist_df, mnist_label_column)

    # run knn on mnist
    # print("knn-mnist")
    # run_knn(mnist_input, mnist_output, 1)
    # run_knn(mnist_input, mnist_output, 3)
    # run_knn(mnist_input, mnist_output, 5)
    # run_knn(mnist_input, mnist_output, 7)
    # x = np.array([1,3,5,7])
    # y1 = np.array([])
    # y1 = np.append(y1, run_knn(mnist_input, mnist_output, 1)[0])
    # y1 = np.append(y1, run_knn(mnist_input, mnist_output, 3)[0])
    # y1 = np.append(y1, run_knn(mnist_input, mnist_output, 5)[0])
    # y1 = np.append(y1, run_knn(mnist_input, mnist_output, 7)[0])
    # y2 = np.array([])
    # y2 = np.append(y2, run_knn(mnist_input, mnist_output, 1)[1])
    # y2 = np.append(y2, run_knn(mnist_input, mnist_output, 3)[1])
    # y2 = np.append(y2, run_knn(mnist_input, mnist_output, 5)[1])
    # y2 = np.append(y2, run_knn(mnist_input, mnist_output, 7)[1])
    # plt.plot(x, y1, label="accuracy")
    # plt.plot(x, y2, label="f1-score")
    # plt.xlabel("Values for k")
    # plt.ylabel("Percentage")
    # plt.legend()
    # plt.show()
# ********************************************************
    # get titanic dataset
    titanic_df, titanic_input, titanic_output, titanic_label_column = process_titanic()
    enc = OneHotEncoder(handle_unknown='ignore')
    titanic_input_one_hot = titanic_input
    # sex could be only categorical variable we being tested - order matters for pclass so we can keep it numerical
    # however here it is considered categorical
    titanic_input_one_hot = np.delete(titanic_input_one_hot, [0,1], axis=1)
    titanic_input_one_hot_inputs = np.array(enc.fit_transform(titanic_input[:, [0,1]]).toarray())
    titanic_input_one_hot = np.append(titanic_input_one_hot_inputs, titanic_input_one_hot, axis=1)
    # run neural network on titanic
    # arch 1 **************************
    # layers = [5, 4]
    # learning_rate = 1
    # lambda_param = 0.5
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(titanic_input_one_hot, titanic_output, layers, learning_rate, lambda_param, delta, max_training_runs)
    # arch 2 **************************
    # layers = [5]
    # learning_rate = 1
    # lambda_param = 0.25
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(titanic_input_one_hot, titanic_output, layers, learning_rate, lambda_param, delta, max_training_runs)
    # arch 3 **************************
    # layers = [6, 4]
    # learning_rate = 1
    # lambda_param = 0.5
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(titanic_input_one_hot, titanic_output, layers, learning_rate, lambda_param, delta, max_training_runs)

    # run descision tree on titanic
    # columns by type: (0, cat), (1, cat), all the rest continous/numerical
    run_decision_tree(titanic_df, titanic_label_column)

    # run knn on titanic
    # print("knn-titanic")
    # run_knn(titanic_input_one_hot, titanic_output, 1)
    # run_knn(titanic_input_one_hot, titanic_output, 3)
    # run_knn(titanic_input_one_hot, titanic_output, 5)
    # run_knn(titanic_input_one_hot, titanic_output, 7)
    # x = np.array([1,3,5,7])
    # y1 = np.array([])
    # y1 = np.append(y1, run_knn(titanic_input_one_hot, titanic_output, 1)[0])
    # y1 = np.append(y1, run_knn(titanic_input_one_hot, titanic_output, 1)[0])
    # y1 = np.append(y1, run_knn(titanic_input_one_hot, titanic_output, 1)[0])
    # y1 = np.append(y1, run_knn(titanic_input_one_hot, titanic_output, 1)[0])
    # y2 = np.array([])
    # y2 = np.append(y2, run_knn(titanic_input_one_hot, titanic_output, 1)[1])
    # y2 = np.append(y2, run_knn(titanic_input_one_hot, titanic_output, 1)[1])
    # y2 = np.append(y2, run_knn(titanic_input_one_hot, titanic_output, 1)[1])
    # y2 = np.append(y2, run_knn(titanic_input_one_hot, titanic_output, 1)[1])
    # plt.plot(x, y1, label="accuracy")
    # plt.plot(x, y2, label="f1-score")
    # plt.xlabel("Values for k")
    # plt.ylabel("Percentage")
    # plt.legend()
    # plt.show()
# ********************************************************
    # get loan dataset
    loan_df, loan_input, loan_output, loan_label_column = process_loans()
    enc = OneHotEncoder(handle_unknown='ignore')
    loan_input_one_hot = loan_input
    loan_input_one_hot = np.delete(loan_input_one_hot, [0,1,2,3,4,9,10], axis=1)
    loan_input_one_hot_inputs = np.array(enc.fit_transform(loan_input[:, [0,1,2,3,4,9,10]]).toarray())
    loan_input_one_hot = np.append(loan_input_one_hot_inputs, loan_input_one_hot, axis=1)
    # run neural network on loan
    # arch 1 **************************
    # layers = [6, 4]
    # learning_rate = 1
    # lambda_param = 0.5
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(loan_input_one_hot, loan_output, layers, learning_rate, lambda_param, delta, max_training_runs)
    # arch 2 **************************
    # layers = [6]
    # learning_rate = 1
    # lambda_param = 0.25
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(loan_input_one_hot, loan_output, layers, learning_rate, lambda_param, delta, max_training_runs)
    # arch 3 **************************
    # layers = [6]
    # learning_rate = 0.1
    # lambda_param = 0.5
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(loan_input_one_hot, loan_output, layers, learning_rate, lambda_param, delta, max_training_runs)
    # arch 4 **************************
    # layers = [6,4,3]
    # learning_rate = 1
    # lambda_param = 0.5
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(loan_input_one_hot, loan_output, layers, learning_rate, lambda_param, delta, max_training_runs)
    # run descision tree on loan
    # [0,1,2,3,4,9,10] are cat, all the rest continous
    loan_decision_tree_df = loan_df
    # def convert_no_yes_to_0_1(x):
    #     if x[-1] == 'N':
    #         x[-1] = 0
    #         return x
    #     elif x[-1] == 'Y':
    #         x[-1] = 1
    #         return x
    #     else:
    #         print("bad data")
    #         quit()
    # loan_decision_tree_df = np.apply_along_axis(convert_no_yes_to_0_1, arr=loan_decision_tree_df, axis=1) 
    # print(loan_decision_tree_df)
    # run_decision_tree(loan_decision_tree_df, loan_label_column)

    # run knn on loan
    # print("knn-loans")
    # x = np.array([1,3,5,7])
    # y1 = np.array([])
    # y1 = np.append(y1, run_knn(loan_input_one_hot, loan_output, 1)[0])
    # y1 = np.append(y1, run_knn(loan_input_one_hot, loan_output, 1)[0])
    # y1 = np.append(y1, run_knn(loan_input_one_hot, loan_output, 1)[0])
    # y1 = np.append(y1, run_knn(loan_input_one_hot, loan_output, 1)[0])
    # y2 = np.array([])
    # y2 = np.append(y2, run_knn(loan_input_one_hot, loan_output, 1)[1])
    # y2 = np.append(y2, run_knn(loan_input_one_hot, loan_output, 1)[1])
    # y2 = np.append(y2, run_knn(loan_input_one_hot, loan_output, 1)[1])
    # y2 = np.append(y2, run_knn(loan_input_one_hot, loan_output, 1)[1])
    # plt.plot(x, y1, label="accuracy")
    # plt.plot(x, y2, label="f1-score")
    # plt.xlabel("Values for k")
    # plt.ylabel("Percentage")
    # plt.legend()
    # plt.show()
# ********************************************************
    # get parkinsons dataset
    parkinsons_df, parkinsons_input, parkinsons_output, parkinsons_label_column = process_parkinsons()
    # run neural network on parkinsons
    # layers = [6, 4]
    # learning_rate = 1 
    # lambda_param = 0.5
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(parkinsons_input, parkinsons_output, layers, learning_rate, lambda_param, delta, max_training_runs)

     # run neural network on parkinsons
    # layers = [10, 6]
    # learning_rate = 0.1
    # lambda_param = 0.5
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(parkinsons_input, parkinsons_output, layers, learning_rate, lambda_param, delta, max_training_runs)

     # run neural network on parkinsons
    # layers = [9]
    # learning_rate = 1
    # lambda_param = 0.5
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(parkinsons_input, parkinsons_output, layers, learning_rate, lambda_param, delta, max_training_runs)

    # run neural network on parkinsons
    # layers = [9,7]
    # learning_rate = 1
    # lambda_param = 0.25
    # print(f"layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_neural_net(parkinsons_input, parkinsons_output, layers, learning_rate, lambda_param, delta, max_training_runs)

    # run descision tree on parkinsons - all data continous
    run_decision_tree(parkinsons_df, parkinsons_label_column)

    # run knn on parkinsons
    # print("knn parkinson's")
    # x = np.array([1,3,5,7])
    # y1 = np.array([])
    # y1 = np.append(y1, run_knn(parkinsons_input, parkinsons_output, 1)[0])
    # y1 = np.append(y1, run_knn(parkinsons_input, parkinsons_output, 3)[0])
    # y1 = np.append(y1, run_knn(parkinsons_input, parkinsons_output, 5)[0])
    # y1 = np.append(y1, run_knn(parkinsons_input, parkinsons_output, 7)[0])
    # y2 = np.array([])
    # y2 = np.append(y2, run_knn(parkinsons_input, parkinsons_output, 1)[1])
    # y2 = np.append(y2, run_knn(parkinsons_input, parkinsons_output, 3)[1])
    # y2 = np.append(y2, run_knn(parkinsons_input, parkinsons_output, 5)[1])
    # y2 = np.append(y2, run_knn(parkinsons_input, parkinsons_output, 7)[1])
    # plt.plot(x, y1, label="accuracy")
    # plt.plot(x, y2, label="f1-score")
    # plt.xlabel("Values for k")
    # plt.ylabel("Percentage")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()

