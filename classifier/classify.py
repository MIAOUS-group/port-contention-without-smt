import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
import tikzplotlib
import json
from os import listdir
from os.path import isfile, join
import statistics
import random
import time
import argparse

# CONSTANTS




'''
Threshold to distinguish when the ratio(interleaved/grouped) reveals that both instructions use different port or not.
It may depend on browser vendor or version so we need to fine tune a value that work for most.
If the ratio is above the threshold, we consider that there is a difference on sequential port contention between interleaved and grouped.
'''
RATIO_THRESHOLD = 1.09

'''
Number of repetitions in the dataset.
Generally 10 as the website we created (https://fp-cpu-gen.github.io/fp-cpu-gen/index.html) gathers 10 traces.
'''
REP = 10

'''
True if you want to remove microarchitectures outside of the training set
'''
only_training_set = True


# Groups of generations
group_zen = ['zen', 'zen 2', 'zen 3']
group_coffee = ['coffee lake', 'whiskey lake', 'comet lake']
group_well = ['haswell','broadwell']
group_skylake = ['skylake', 'cascade lake sp']
group_bridge = ['sandy bridge', 'ivy bridge']


#################################### Utils #####################################

def get_dataset_path(grouped,balanced):
    '''     Picks the proper trainung dataset according to parameters.
    We have several training sets based on the same results from the framework.
    A balanced set includes the same number of traces from all generations / groups of generations.
    A grouped set categorizes similar generations (eg Haswell and Broadwell) into a single class as they are indistinguishable using sequential port contention.

    Parameters:
    grouped(bool): If True, selects a grouped dataset.
    balanced(bool): If True, selects a balanced dataset.

    Returns:
    path(str): The path to the chosen dataset.
    '''
    if grouped:
        if balanced:
            path = "traces_grouped_balanced/"
        else:
            path = "traces_grouped/"
    else:
        if balanced:
            path = "traces_ungrouped_balanced/"
        else:
            path = "traces_ungrouped/"
    return path



def get_classes(dataset_path):
    ''' Dynamically extracts classes from the dataset.
    Basically, we load all files then extracts the first part of the formatted name.
    It allows to dynamically change the classes if we restrict/extend our training set.

    Parameters:
    path(str): The path to the chosen dataset.

    Returns:
    classes(list): List of string containing classes for the classifier.
    '''
    traces = [ f for f in listdir(dataset_path) if isfile(join(dataset_path, f)) and ".txt" in f ]
    classes = []
    for trace in traces:
        gen = trace.split('__')[0]
        if gen not in classes:
            classes.append(gen)
    return classes



def read_result_file_list(path,classes,grouped):
    '''  Read all traces for the evaluation.
    Traces must be formatted JSON format similar to the output of https://fp-cpu-gen.github.io/fp-cpu-gen

    Parameters:
    path(str): The path to the evaluation dataset.
    classes(list): List of string containing classes for the classifier.
    grouped(bool): If True, selects a grouped dataset.

    Returns:
    X(list): Binary traces for a specific run. For each instruciton, holds 1 if the instructions use different ports and 0 otherwise. An element of the List is a list containing results for all tested instructions.
    y(list): Ground truth, generation associated with the trace.
    '''
    rep = REP
    traces = [[] for i in range(rep)]
    with open(path,'r') as f:
        data = json.load(f)
    for instruction in data['timings']:
        dic = data['timings'][instruction]
        for i in range(rep):
            try:
                ratio = dic['pcm'][i] / dic['npcm'][i]
            except:
                ratio =  0
            # Check whether or not both instrucitons use different ports.
            traces[i].append(int(ratio >= RATIO_THRESHOLD))
    sequences = list()
    for i in range(rep):
        np_values = np.array(traces[i][1:],dtype=np.double)
        sequences.append(np_values)
    X = sequences
    gt = data['ground_truth']
    if grouped: # If we use the group classifier, we merge the classes
        if gt in group_zen:
            gt = 'zen'
        elif gt in group_coffee:
            gt = 'coffee_whiskey_comet'
        elif gt in group_well:
            gt = 'haswell_broadwell'
        elif gt in group_skylake:
            gt = 'cascade_skylake'
        elif gt in group_bridge:
            gt = 'ivy_sandy_bridge'
    y = [gt for _ in range(rep)]
    if only_training_set: # Sometimes, we have weird generations on which we didn't train, so we exclude them
    # This would be great to redo training on more generations!
        if gt not in classes:
            X = []
            y = []
    return(X, y)



def get_results(path, classes,grouped):
    ''' Gather traces for evaluation and format them to be fed to the knn.

    Parameters:
    path(str): The path to the evaluation dataset.
    classes(list): List of string containing classes for the classifier.
    grouped(bool): If True, selects a grouped dataset.

    Returns:
    X(pandas.DataFrame): DataFrame containing binary traces for the whole evaluation dataset
    y(pandas.Series): Contains ground truth associated with X
    X_ll(List): Two dimensionnal array containing binary traces for the whole evaluation dataset
    y_ll(list): Contains ground truth associated with X
    '''
    files = [ f for f in listdir(path) if isfile(join(path, f)) and ".json" in f ]
    X_l = []
    y_l = []
    X_ll = []
    y_ll = []
    for f in files:
        (X, y) = read_result_file_list(path+f,classes,grouped)
        X_l = X_l + X
        y_l = y_l + y
        if (X != []):
            X_ll.append(X)
            y_ll.append(y)
    X = pd.DataFrame({'col1': X_l})
    y = pd.Series(y_l)
    return (X,y,X_ll,y_ll)



def get_files(path):
    ''' List all file names in a folder.
    Top notch engineering wahoo.

    Parameters:
    path(str): Path to the folder.

    Returns:
    (list): List of filenames as strings.
    '''
    return [ f for f in listdir(path) if isfile(join(path, f)) and ".txt" in f ]



def get_knn(n_neighbors, distance):
    ''' Instantiates a knn classifier.

    Parameters:
    n_neighbors(int): Number of neighbors used for knn classification.
    distance(str): Distance used by the classifier. Refer to scikitlearn doc for available metrics.

    Returns:
    knn(sklearn.KNeighborsTimeSeriesClassifier): K-NN classifier instance.
    '''
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=3, distance="euclidean", n_jobs=-1)
    return knn



def get_training_set(training_set_path, files):
    ''' Loads and formats the training set.

    Parameters:
    training_set_path(str): Path of the training set.
    files(list): List of files in the training set.

    Returns:
    X_train(pandas.DataFrame): Binary data traces for the training set.
    y_train(pands.Series): Classes of the associated trace.
    '''
    sequences = list()
    target = list()
    maxlen = 0
    for f in files:
        df = pd.read_csv(training_set_path + f, header=0)
        values = df.values.flatten()
        maxlen = max(len(values), maxlen)
    for f in files:
        df = pd.read_csv(training_set_path + f, header=0)
        values = df.values.flatten()
        if len(values) > maxlen:
            values = values[:maxlen]
        np_values = np.array(values, dtype=np.double)
        np_values = np.pad(np_values, (0, maxlen - len(values)), 'constant')
        sequences.append(np_values)
        target.append(f.split("__")[0])
    X_train = pd.DataFrame({'col1': sequences })
    y_train = pd.Series(target)
    return (X_train,y_train)



def self_fit(X_train, y_train,knn):
    ''' Here, we evaluate our knn classifier on a subset of our training set and not on
    actual evaluation data. This lets us quickly test the efficiency of parameters.

    Parameters:
    X_train(pandas.DataFrame): Binary data traces for the training set.
    y_train(pands.Series): Classes of the associated trace.
    knn(sklearn.KNeighborsTimeSeriesClassifier): K-NN classifier instance.

    Returns:
    y_test: Ground truth of test subset.
    y_pred: Predicted values of test subset.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X_train.iloc[:], y_train) #Remove subset of the training set for evaluation.
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    labels, counts = np.unique(y_train, return_counts=True)
    knn.fit(X_train, y_train) # Train our model
    y_pred = knn.predict(X_test) #Evaluates it on the part we removed
    return (y_test, y_pred)



def evaluation_fit(X_train, y_train,X_test, y_test, X_ll, y_ll, knn, mv_size):
    ''' Here, we evaluate our knn classifier on evaluation data.
    This repository holds the dataset we used for the paper, gathered with  https://fp-cpu-gen.github.io/fp-cpu-gen.
    It holds 50 traces and 12 generations but feel free to enrich it with new data!

    This function uses majority voting.
    Basically, sometimes the first experiment is "noisy", probably due to warming up or javascript black magic.
    To avoid that, we computed several traces for each instruction (10 for the paper) and implemented majority voting to determine the class.
    For a majority voting window of n, we classify the first n experiments, and the value most present in the results is the chosen class.
    The higher the window, the more stable the result but it requires more traces in a real-world attack.

    Parameters:
    X_train(pandas.DataFrame): Binary data traces for the training set.
    y_train(pands.Series): Classes of the associated trace.
    X_test(pandas.DataFrame): DataFrame containing binary traces for the whole evaluation dataset.
    y_test(pandas.Series): Contains ground truth associated with X_test.
    X_ll(List): Two dimensionnal array containing binary traces for the whole evaluation dataset.
    y_ll(list): Contains ground truth associated with X_ll.
    knn(sklearn.KNeighborsTimeSeriesClassifier): K-NN classifier instance.
    mv_size(int): Size of the majority voting window.

    Returns:
    y_test_mv: Ground truth of test subset after majority voting.
    y_pred_mv: Predicted values of test subset after majority voting.
    '''
    knn.fit(X_train, y_train)
    y_test_mv = list()
    y_pred_mv = list()
    for cpu in range(len(X_ll)):
        y_pred = knn.predict(pd.DataFrame({'col1': X_ll[cpu]}))
        y_test_mv.append(statistics.mode(y_ll[cpu][:mv_size]))
        y_pred_mv.append(statistics.mode(y_pred[:mv_size]))
    return (y_test_mv,y_pred_mv)



def print_data(y_test, y_pred, save_matrix):
    print("Balanced accuracy score: {}%".format(balanced_accuracy_score(y_test, y_pred)))

    print("\n")
    print("Classification Report: ")
    print(classification_report(y_test,y_pred))

    print("\n\n")
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred, normalize = 'true')
    print(cm)
    for line in cm:
        lsum = max(1, sum(line.tolist()))
        print(",".join([str(100.0 * x / lsum) for x in line.tolist()]))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred,normalize = 'true')
    plt.xticks(rotation=45)
    if save_matrix:
        tikzplotlib.save("confusion_matrix.tex")
    plt.show()



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--neighbors', help='Number of knn neighbors', type=int, default = 3)
    parser.add_argument('-d', '--distance', help='Distance used for knn', type=str, default = 'euclidean')
    parser.add_argument('-m', '--mv_size', help='Size of the majority-voting window', type=int, default = 10)
    parser.add_argument('--grouped', dest='grouped',help='Classify with groups of similar generations. This is default', action='store_true')
    parser.add_argument('--ungrouped', dest='grouped',help='Disable groups of generations', action='store_false')
    parser.set_defaults(grouped=True)
    parser.add_argument('--balanced', dest='balanced', help='Used balanced dataset. This is default',action='store_true')
    parser.add_argument('--unbalanced', dest='balanced',help='Uses unbalanced dataset, may introduce bias.', action='store_false')
    parser.set_defaults(balanced=True)
    parser.add_argument('--self_fit', dest='self_fit', help='Evaluate the knn on data from the training set.',action='store_true')
    parser.add_argument('--no-self_fit', dest='balanced',help='Skip testing the knn on data from the training set. This is default.', action='store_false')
    parser.set_defaults(self_fit=False)
    parser.add_argument('--evaluation_fit', dest='evaluation_fit', help='Evaluate the knn on evaluation data. This is default.',action='store_true')
    parser.add_argument('--no-evaluation_fit', dest='balanced',help='Skip testing the knn on evaluation data.', action='store_false')
    parser.set_defaults(evaluation_fit=True)
    parser.add_argument('-s', '--save_matrix', help="Save confusion matrix as png", action='store_true', default=False)
    parser.add_argument('--evaluation_path', help="Path for evaluation data", type=str, default="./evaluation_data/")
    args = parser.parse_args()
    return args



def main():
    args = parse_arguments()
    training_dataset_path = get_dataset_path(args.grouped,args.balanced)
    classes = get_classes(training_dataset_path)
    training_files = get_files(training_dataset_path)
    (X_train, y_train) = get_training_set(training_dataset_path, training_files)
    if args.self_fit:
        knn = get_knn(args.neighbors, args.distance)
        (y_test, y_pred) = self_fit(X_train,y_train,knn)
        print_data(y_test, y_pred,args.save_matrix)
    if args.evaluation_fit:
        knn = get_knn(args.neighbors, args.distance)
        (X_test, y_test, X_ll, y_ll) = get_results(args.evaluation_path,classes,args.grouped)
        (y_test, y_pred) = evaluation_fit(X_train, y_train,X_test, y_test, X_ll, y_ll, knn, args.mv_size)
        print_data(y_test, y_pred,args.save_matrix)


if __name__ == '__main__':
    main()
