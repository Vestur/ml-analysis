import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import math


class Model:
    def __init__(self, input: np.ndarray, output: np.ndarray, layers: np.ndarray, learning_rate: float, lambda_param: float, step_by_step=False):
        """This is the main class to create and run a neural network

        Args:
            input (np.ndarray): input data
            output (np.ndarray): output data

            layers (np.ndarray): iterable whose elements are the number of neurons in each layer. 
            For example, [6,4,3] is a network with a data input layer, a layer with 6 neurons, 
            another layer with 4 neurons, then an output layer with 3 neurons.

            learning_rate (float): model learning rate hyperparamter
        """
        self.lambda_param = lambda_param
        self.input_size = input.shape
        self.layers = layers
        self.num_layers = self.layers.shape[0]
        # weights initialized to random values from a gaussian distribution with mean 0, variance 1
        self.weights = self.initialize_weights(self.layers) # a dictionary where the key is the layer number (1,...,L)
        self.learning_rate = learning_rate # learning rate to be used
        self.layer_outputs = dict
        self.inputs = input
        self.outputs = output
        self.step_by_step = step_by_step

    # manually set weights (good for testing)
    def set_weights(self, weights):
        self.weights = weights
    
    # get weights (good for testing)
    def get_weights(self):
        return self.weights

    # manually set outputs (good for testing)
    def set_outputs(self, layer_outputs):
        self.layer_outputs = layer_outputs
    
    # get outputs (good for testing)
    def get_outputs(self):
        return self.layer_outputs

    # sigmoid activation function
    def sigmoid(self, input_vector: np.ndarray):
        return 1/(1 + np.e**(-input_vector))

    def initialize_weights(self, layers):
        weights = dict()
        for i in range(1, self.num_layers):
            # only iterate for layers 1...L-1
            if i == 0:
                continue
            # initialize weight for layer i, (i.e. the weight that gets multiplied by layer i, to be passed into the activation function of layer i+1)
            # number of columns is layers[i]+1 to add bias term
            weights[i] = np.random.normal(0,1,size=(layers[i], layers[i-1]+1))
        return weights

    def get_accuracy(self, inputs, outputs):
        total_predictions = 0
        n_correct = 0
        for i in range(inputs.shape[0]):
            input = inputs[i,:]
            output = outputs[i, :]
            model_output = np.array(self.forward(input)[0])
            prediction = np.zeros_like(model_output)
            prediction[model_output.argmax(0)] = 1
            n_correct += np.sum(prediction == output)
            assert(output.shape[0] == prediction.shape[0] )
            total_predictions += output.shape[0]
        return n_correct/total_predictions


    # forward pass
    def forward(self, data_input: np.ndarray, training=False):
        """returns a dictionary of outputs where the key is the layer number and the value is the vector of values for each neuron in that layer

        Args:
            data_input (np.ndarray): vector input

        Returns:
            (np.ndarray): model output vector
        """
        a = dict()
        # output of 1st layer (L=1) is just input to network
        a[1] = data_input
        # prepend bias to 1st layer output
        a[1] = np.insert(a[1], 0, 1)
        num_layers = self.num_layers

        # iterate from 1,...,L-1
        z_i = None
        for i in self.weights.keys():
            # already calculated output of 1st layer above
            if i == 1:
                # print for correctness verification
                if self.step_by_step and (not training):
                    print(f"z{i}: {z_i}")
                    print(f"a{i}: {a[i]}")
                continue
            else:
                z_i = self.weights[i-1].dot(a[i-1])
                # element wise activation function
                a[i] = self.sigmoid(z_i)
                # add bias term
                a[i] = np.insert(a[i], 0, 1)
                # print for correctness verification
                if self.step_by_step and (not training):
                    print(f"z{i}: {z_i}")
                    print(f"a{i}: {a[i]}")

        # do last layer separately because it has no bias term
        z_L = self.weights[num_layers-1].dot(a[num_layers-1])
        # here we want outputs between 0 and 1, but this may not always be the case
        # if it isn't the case (i.e. we have different loss function, remove next line)
        a[num_layers] = self.sigmoid(z_L)
        # print for correctness verification
        if self.step_by_step and (not training):
            print(f"z{num_layers}: {z_L}")
            print(f"a{num_layers}: {a[num_layers]}")
        # return final output
        self.layer_outputs = a
        return a[num_layers], a

    def logistical_regression_loss(self, actual_output, model_output):
        y = actual_output
        y_hat = np.array(model_output,dtype=float)
        return np.multiply(-y, np.log(y_hat)) - np.multiply((1-y), np.log(1-y_hat))

    def get_cost(self, inputs=None, outputs=None):
        lambda_param = self.lambda_param
        if not (inputs is None) and not (outputs is None):
            input_data = inputs
            output_data = outputs
        else:
            input_data = self.inputs
            output_data = self.outputs
        total_cost = 0
        for index in range(0, len(input_data)):
            actual_output = output_data[index,:]
            cur_input = input_data[index,:]
            model_output, _ = self.forward(cur_input, training=True)
            total_cost += np.sum(self.logistical_regression_loss(actual_output, model_output))

        num_instances = input_data.shape[0]
        average_cost = total_cost/num_instances
        squared_weights = sum([np.sum(np.square(weights[:, 1:])) for weights in self.weights.values()])
        regularization_term = lambda_param/(2*num_instances) * squared_weights
        return average_cost + regularization_term
    
    def train(self, batch_size=None, input_data=None, output_data=None, final_result=True, intermediate_results=True):
        # batch gradient descent by default
        # 
        if not (input_data is None) and not (output_data is None):
            if batch_size is None:
                batch_size = len(input_data)
            num_splits = math.floor(len(input_data)/batch_size)
            inputs = np.array_split(input_data, num_splits)
            outputs = np.array_split(output_data, num_splits)
        else:
            if batch_size is None:
                batch_size = len(self.inputs)
            num_splits = math.floor(len(self.inputs)/batch_size)
            inputs = np.array_split(self.inputs, num_splits)
            outputs = np.array_split(self.outputs, num_splits)
        num_instances_trained_array = np.array([0])
        num_instances_trained = 0
        losses = np.array(self.get_cost())
        for input_batch, output_batch in zip(inputs, outputs):
            aggregate_delta = dict()
            assert(len(input_batch) == len(output_batch))
            cur_num_instances = len(input_batch)
            num_instances_trained += cur_num_instances
            for input_instance, output_instance in zip(input_batch, output_batch):
                delta = dict()
                model_output, a = self.forward(input_instance, training=True)
                delta[self.num_layers] = model_output - output_instance
                # print for correctness verification
                if self.step_by_step and intermediate_results:
                        print(f"delta{self.num_layers}: {delta[self.num_layers]}")
                for layer in range(self.num_layers-1, 1, -1):
                    delta[layer] = (np.transpose(self.weights[layer])).dot(delta[layer+1])*a[layer]*(1-a[layer])
                    delta[layer] = delta[layer][1:]
                    # sum deltas/blame for each layer
                    # print for correctness verification
                    if self.step_by_step and intermediate_results:
                        print(f"delta{layer}: {delta[layer]}")
                for layer in range(self.num_layers-1, 0, -1):
                    new_theta = delta[layer+1][:,np.newaxis].dot(np.transpose(a[layer][:,np.newaxis]))
                    if self.step_by_step and intermediate_results:
                            print(f"Theta{layer}: {new_theta}")
                    try:
                        aggregate_delta[layer] += new_theta
                    except KeyError:
                        aggregate_delta[layer] = new_theta
            P = dict()
            for layer in range(self.num_layers-1, 0, -1):
                P[layer] = np.multiply(self.lambda_param,self.weights[layer])
                P[layer][:,0] = 0
                aggregate_delta[layer] = np.multiply(1/cur_num_instances, (aggregate_delta[layer] + P[layer]))
                if self.step_by_step:
                    continue
                self.weights[layer] = self.weights[layer] - self.learning_rate*(aggregate_delta[layer])
            num_instances_trained_array = np.append(num_instances_trained_array, num_instances_trained)
            losses = np.append(losses, self.get_cost())
        # print for correctness verification
        if self.step_by_step and final_result:
            for layer in aggregate_delta.keys():
                print(f"Final Regularized Theta{layer}\n: {aggregate_delta[layer]}")
        return num_instances_trained_array, losses

    def evaluate(self, test_input: np.ndarray, test_output: np.ndarray, binary=False):
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
        classes = np.unique(test_output, axis=0)
        num_instances = test_output.shape[0]
        num_classes = classes.shape[0]
        classes = [(np.where(_class==1)[0][0]+1) for _class in classes]
        for true_class in classes:
            for predicted_class in classes:
                confusion_matrix[f"{true_class},{predicted_class}"] = 0
        
        for input_instance, output_instance in zip(test_input, test_output):
            model_output = np.array(self.forward(input_instance)[0])
            prediction = np.zeros_like(model_output)
            prediction[model_output.argmax(0)] = 1
            predicted_class = (np.where(prediction==1)[0][0]+1)
            true_class = (np.where(output_instance==1)[0][0]+1)
            confusion_matrix[f"{true_class},{predicted_class}"] += 1

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

def normalize_data(data: np.ndarray):
        """normalize input data

        Args:
            data (np.ndarray): input data to normalize
        """
        maxes = np.amax(data, axis=0)
        mins = np.amin(data, axis = 0)
        cols_to_delete = []
        for col in range(len(maxes)):
            if maxes[col]==0 and mins[col]==0:
                cols_to_delete.append(col)
        data = np.delete(data, cols_to_delete, 1)
        maxes = np.delete(maxes, cols_to_delete, 0)
        mins = np.delete(mins, cols_to_delete, 0)

        def normalize(row_vector: np.ndarray):
            # add small epsilon in case both max and min are 0
            return (maxes - row_vector)/((maxes-mins))

        data = np.apply_along_axis(normalize, arr=data, axis=1)
        return data

def load(filename: str, delimiter: str) -> np.ndarray:
    """
    Takes in name of file assumed to be csv. The file can be deliminited with tab or comma returns np.ndarray of data

    :param filename: (str) either full or local path to data file
    :return: np.ndarray of data
    """
    data = pd.read_csv(filename, delimiter=delimiter)
    data = data.to_numpy()
    return data

def run_cross_fold_validation(inputs, outputs, num_folds):
    # assume input/output is preshuffled
    classes = np.unique(outputs)
    # get data split into classes
    split_data_inputs = dict()
    split_data_outputs = dict()
    for _class in classes:
        split_data_inputs[_class] = inputs[outputs[:] == _class]
        split_data_outputs[_class] = np.full_like(outputs, fill_value=_class, shape=(split_data_inputs[_class].shape[0], ))
    # get 1/k * amount of data in each class to put into k folds (to maintain proportions)
    input_folds = dict()
    output_folds = dict()
    for i in range(num_folds):
        for _class in split_data_inputs.keys():
            if len(input_folds.keys()) != i:
                input_folds[i+1] = np.concatenate((input_folds[i+1],np.array_split(split_data_inputs[_class], num_folds)[i]), axis=0)
                output_folds[i+1] = np.concatenate((output_folds[i+1],np.array_split(split_data_outputs[_class], num_folds)[i]), axis=0)
            else:
                input_folds[i+1] = np.array_split(split_data_inputs[_class], num_folds)[i]
                output_folds[i+1] = np.array_split(split_data_outputs[_class], num_folds)[i]
    # make output arrays 2d (i.e. n x 1 vectors instead of n x none)
    for i in range(num_folds):
        output_folds[i+1] = np.expand_dims(output_folds[i+1], axis=1)
    
    return input_folds, output_folds
    # unique_classes, class_counts = np.unique(outputs, return_counts=True)
    # input_by_class = dict()
    # class_info = dict()
    # folds_input = dict()
    # folds_output = dict()
    # for i, cat in enumerate(unique_classes):
    #     class_info[cat] = dict()
    #     class_info[cat]["num_to_put_in_each_fold"] = math.floor(class_counts[i]/num_folds)
    #     input_by_class[cat] = inputs[outputs==cat, :]

    # for i in range(num_folds):
    #     for cat in unique_classes:
    #         try:
    #             folds_input[i+1] = np.concatenate((folds_input[i+1], input_by_class[cat][:class_info[cat]["num_to_put_in_each_fold"]]))
    #         except KeyError: 
    #             folds_input[i+1] = input_by_class[cat][:class_info[cat]["num_to_put_in_each_fold"]]

    #         try:
    #             folds_output[i+1] = np.concatenate((folds_output[i+1], np.full(class_info[cat]["num_to_put_in_each_fold"], cat)))
    #         except KeyError:
    #             folds_output[i+1] = np.full(class_info[cat]["num_to_put_in_each_fold"], cat)
    #         assert(len(folds_input[i+1]) == len(folds_output[i+1]))

    #         # remove used instances
    #         input_by_class[cat] = np.delete(input_by_class[cat], [0,class_info[cat]["num_to_put_in_each_fold"]], axis=0)
    # assert(len(folds_input) == len(folds_output))

    # # make output arrays 2d (i.e. n x 1 vectors instead of n x none)
    # for i in range(num_folds):
    #     folds_output[i+1] = np.expand_dims(folds_output[i+1], axis=1)
    
    # return folds_input, folds_output

def train_test(input_folds, output_folds, learning_rate, layers, lambda_param, delta, max_training_runs):
    # folds labeled from 1,..,num folds
    num_folds = len(input_folds)
    accuracy = 0
    averaged_f1_score = 0
    for fold in range(len(input_folds)):
        testing_input = input_folds[fold+1]
        testing_output = output_folds[fold+1]
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
        model = Model(training_input, training_output, layers, learning_rate, lambda_param)
        prev_loss = np.inf
        num_instances = np.array([])
        test_losses = np.array([])
        accuracies = np.array([])
        f1_scores = np.array([])
        statst_instances = np.array([])
        last_instance = 0
        for i in tqdm(range(max_training_runs)):
            num_instances_trained, losses = model.train()
            new_loss = losses[-1]
            if (prev_loss - new_loss) < 1e-9:
                break
            else:
                prev_loss = new_loss
                test_losses = np.append(test_losses, model.get_cost(testing_input, testing_output))
                num_instances_trained = num_instances_trained + last_instance
                num_instances = np.append(num_instances, num_instances_trained)
                last_instance = num_instances_trained[-1]
                cur_accuracy, precision, recall, f1_score = model.evaluate(testing_input, testing_output)
                statst_instances = np.append(statst_instances, last_instance)
                accuracies = np.append(accuracies, cur_accuracy)
                f1_scores = np.append(f1_scores, f1_score)
        cur_accuracy, precision, recall, f1_score = model.evaluate(testing_input, testing_output)
        accuracy += cur_accuracy
        averaged_f1_score += f1_score
        print(f"Cur Accuracy: {cur_accuracy}")
        print(f"Cur F1-Score: {f1_score}")
        plt.plot(statst_instances, test_losses)
        plt.title("Training Samples vs Test Loss")
        plt.xlabel(f"num_instances,\n final accuracy:{cur_accuracy}, lr: {learning_rate}")
        plt.ylabel("test_loss")
        plt.show()
        # plt.plot(statst_instances, f1_scores)
        # plt.show()
    accuracy = accuracy/num_folds
    averaged_f1_score = averaged_f1_score/num_folds
    print(f"Average Accuracy: {accuracy}")
    print(f"Average F1-Score: {averaged_f1_score}")
        
def OneHot(output_folds):
    num_folds = len(output_folds)
    # perform one hot encoding on outputs
    for fold in range(num_folds):
        enc = OneHotEncoder(handle_unknown='ignore')
        output_folds[fold+1] = np.array(enc.fit_transform(output_folds[fold+1]).toarray())
    return output_folds

def run_house_votes(layers, learning_rate, lambda_param):
    df = load("datasets/hw3_house_votes_84.csv", ',')
    label_column_index = 16
    output_columns = [16]
    input_columns = list(range(df.shape[1]-1))
    # get just output information
    output = df[:, label_column_index]
    # delete output column from input
    input = np.delete(df, label_column_index, 1)
    # normalize input
    input = normalize_data(input)
    input, output = shuffle(input, output)
    lambda_param = lambda_param
    learning_rate = learning_rate
    layers = np.append(np.array([input.shape[1]]),layers)
    input_folds, output_folds = run_cross_fold_validation(input, output, 10)
    # folds are dictionary labeled 1,...,num_folds
    output_folds = OneHot(output_folds=output_folds)
    layers = np.append(layers,np.array([output_folds[1].shape[1]]))
    train_test(input_folds, output_folds, learning_rate, layers, lambda_param, max_training_runs=15000)

def run_wine_dataset(layers, learning_rate, lambda_param):
    df = load("datasets/hw3_wine.csv", '\t')
    label_column_name = 0
    output_columns = [0]
    input_columns = list(range(1,df.shape[1]))
    # get just output information
    output = df[:, label_column_name]
    # delete output column from input
    input = np.delete(df, label_column_name, 1)
    # normalize input
    input = normalize_data(input)
    input, output = shuffle(input, output)
    lambda_param = lambda_param
    learning_rate = learning_rate
    # add input column size to layers (it is size of first layer)
    layers = np.append(np.array([input.shape[1]]),layers)
    num_folds = 10
    input_folds, output_folds = run_cross_fold_validation(input, output, num_folds)
    # folds are dictionary labeled 1,...,num_folds
    output_folds = OneHot(output_folds=output_folds)
    # add output column size to layers (it is size of last layer)
    layers = np.append(layers,np.array([output_folds[1].shape[1]]))
    train_test(input_folds, output_folds, learning_rate, layers, lambda_param, max_training_runs=15000)


def run_solution_verification_code(instance1):
    # backprop_example1
    if (instance1):
        model = Model(np.array([[0.13],[0.42]]), np.array([[0.9],[0.23]]), layers=np.array([1, 2, 1]), 
        learning_rate=1, lambda_param=0, step_by_step=True)
        model.weights = dict()
        model.weights[1] = np.array([[0.4, 0.1], [0.3, 0.2]])
        model.weights[2] = np.array([[0.7, 0.5, 0.6]])
        print("instance1")
        model.forward([0.13])
        print("instance1 cost")
        print(model.get_cost(np.array([[0.13]]), np.array([[0.9]])))
        print("instance2")
        model.forward([0.42])
        print("instance2 cost")
        print(model.get_cost(np.array([[0.42]]), np.array([[0.23]])))
        print("total cost")
        print(model.get_cost())
        print("instance1 backprop")
        model.train(input_data=np.array([[0.13]]), output_data=np.array([[0.9]]), final_result=False)
        print("instance2 backprop")
        model.train(input_data=np.array([[0.42]]), output_data=np.array([[0.23]]), final_result=False)
        print("all together")
        model.train(intermediate_results=False)
    # ******************************************
    # backprop_example2
    else:
        model = Model(input=np.array([[0.32000, 0.68000], [0.83000, 0.02000]]), 
        output=np.array([[0.75000, 0.98000], [0.75000, 0.28000]]), layers=np.array([2,4,3,2]), lambda_param = 0.250, 
        learning_rate = 1, step_by_step=True)
        model.weights = dict()
        model.weights[1] = np.array([[0.42000,0.15000,0.40000], 
                                    [0.72000,0.10000,0.54000],
                                    [0.01000,0.19000,0.42000], 
                                    [0.30000,0.35000,0.68000]])
        model.weights[2] = np.array([[0.21000,0.67000,0.14000,0.96000,0.87000],
                                    [0.87000,0.42000,0.20000,0.32000,0.89000],  
                                    [0.03000,0.56000,0.80000,0.69000,0.09000]])
        model.weights[3] = np.array([[0.04000,0.87000,0.42000,0.53000],
                                    [0.17000,0.10000,0.95000,0.69000]])
        print("instance1")
        model.forward([0.32000, 0.68000])
        print("instance1 cost")
        model.lambda_param = 0
        print(model.get_cost(np.array([[0.32000, 0.68000]]), np.array([[0.75000, 0.98000]])))
        model.lambda_param = 0.25
        print("instance2")
        model.forward([0.83000, 0.02000])
        print("instance2 cost")
        model.lambda_param = 0
        print(model.get_cost(np.array([[0.83000, 0.02000]]), np.array([[0.75000, 0.28000]])))
        model.lambda_param = 0.25
        print("total cost")
        print(model.get_cost())
        print("instance1 backprop")
        model.train(input_data=np.array([[0.32000, 0.68000]]), output_data=np.array([[0.75000, 0.98000]]), final_result=False)
        print("instance2 backprop")
        model.train(input_data=np.array([[0.83000, 0.02000]]), output_data=np.array([[0.75000, 0.28000]]), final_result=False)
        print("all together")
        model.train(intermediate_results=False)

def neural_main():
    print("nothing for testing")
    # ***************************************************************
    # VERIFICATION SECTION
    # set instance1 = False to get solution verification code printout for instance2
    # run_solution_verification_code(instance1=True)
    # ***************************************************************
    # WINE ARCHITECTURES
    # define hidden layers - 1st/input and last/output layers are handled inside run_wine_dataset function
    # ****ex 1
    # layers = np.array([6,4])
    # learning_rate = 0.5
    # lambda_param = 0.5
    # print(f"hidden layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_wine_dataset(layers=layers, learning_rate=learning_rate, lambda_param=lambda_param)
    # layers = np.array([8])
    # learning_rate = 0.00001
    # lambda_param = 0.2
    # print(f"hidden layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_wine_dataset(layers=layers, learning_rate=learning_rate, lambda_param=lambda_param)
    # layers = np.array([8])
    # learning_rate = 1
    # lambda_param = 0.5
    # print(f"hidden layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_wine_dataset(layers=layers, learning_rate=learning_rate, lambda_param=lambda_param)
    # layers = np.array([10,8])
    # learning_rate = 0.1
    # lambda_param = 0.2
    # print(f"hidden layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_wine_dataset(layers=layers, learning_rate=learning_rate, lambda_param=lambda_param)
    # layers = np.array([10,8])
    # learning_rate = 1
    # lambda_param = 0.2
    # print(f"hidden layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_wine_dataset(layers=layers, learning_rate=learning_rate, lambda_param=lambda_param)
    # layers = np.array([8])
    # learning_rate = 1
    # lambda_param = 10
    # print(f"hidden layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_wine_dataset(layers=layers, learning_rate=learning_rate, lambda_param=lambda_param)
    # ***************************************************************
    # WINE ARCHITECTURES
    # define hidden layers - 1st/input and last/output layers are handled inside run_house_votes function
    # ****ex 2
    # layers = np.array([8])
    # learning_rate = 1
    # lambda_param = 10
    # print(f"hidden layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_house_votes(layers=layers, learning_rate=learning_rate, lambda_param=lambda_param)
    # layers = np.array([8])
    # learning_rate = 1
    # lambda_param = 0.5
    # print(f"hidden layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_house_votes(layers=layers, learning_rate=learning_rate, lambda_param=lambda_param)
    # layers = np.array([8])
    # learning_rate = 1
    # lambda_param = 0
    # print(f"hidden layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_house_votes(layers=layers, learning_rate=learning_rate, lambda_param=lambda_param)
    # layers = np.array([9,6,4])
    # learning_rate = 1e-5
    # lambda_param = 0.5
    # print(f"hidden layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_house_votes(layers=layers, learning_rate=learning_rate, lambda_param=lambda_param)
    # layers = np.array([9,6,4])
    # learning_rate = 1e-2
    # lambda_param = 0.5
    # print(f"hidden layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_house_votes(layers=layers, learning_rate=learning_rate, lambda_param=lambda_param)
    # layers = np.array([10,4])
    # learning_rate = 1
    # lambda_param = 0.5
    # print(f"hidden layers: {layers}")
    # print(f"learning_rate: {learning_rate}")
    # print(f"lambda_param: {lambda_param}")
    # run_house_votes(layers=layers, learning_rate=learning_rate, lambda_param=lambda_param)