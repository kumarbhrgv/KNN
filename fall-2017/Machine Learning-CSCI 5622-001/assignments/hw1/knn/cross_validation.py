import argparse
import random
from collections import namedtuple
import numpy as np
from knn import Knearest, Numbers
import math
random.seed(20170830)

SplitIndices = namedtuple("SplitIndices", ["train", "test"])


def split_cv(length, num_folds):
    splits = []
    indices = list(range(length))
    random.shuffle(indices)
    k =0
    number_in_fold = length/num_folds
    for i in range(0,num_folds):
        random.shuffle(indices)
        train= []
        test = indices[int(i* number_in_fold) : int((i+1)* number_in_fold) ]
        for j in range(length):
            if(j not in test):
                train.append(j)
        random.shuffle(train)
        split = SplitIndices(train,test)
        splits.append(split)
    return splits


def cv_performance(x, y, num_folds, k):
    length = len(y)
    splits = split_cv(length, num_folds)
    accuracy_array = []
    for split in splits:
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for i in split.train:
            x_train.append(x[i])
            y_train.append(y[i])
        for i in split.test:
            y_test.append(y[i])
            x_test.append(x[i])
        knn = Knearest(np.array(x_train),np.array(y_train), k)
        confusion = knn.confusion_matrix(np.array(x_test),np.array(y_test))
        accuracy_array.append(knn.accuracy(confusion))
    return np.mean(accuracy_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    parser.add_argument('--num_folds', type=int, default=-1,
                        help="Number of folds for cross validations")
    args = parser.parse_args()
    data = Numbers("../data/mnist.pkl.gz")
    x, y = data.train_x, data.train_y
    
    if args.limit > 0:
        x, y = x[:args.limit], y[:args.limit]
    best_k, best_accuracy = -1, 0
    num_folds = args.num_folds
    if num_folds >0 : 
        print ("Number of folds chosen: ",num_folds)
    else:
        print(" Default Number of folds: ",5)
        num_folds =5
    for k in [1,3,5,7,9,11,15]:
        accuracy = cv_performance(x, y, num_folds, k)
        print("%d-nearest neighber accuracy: %f" % (k, accuracy))
        if accuracy > best_accuracy:
            best_accuracy, best_k = accuracy, k
    print ("best_accuracy : %f, best_k : %d" %(best_accuracy,best_k))
    knn = Knearest(x, y, best_k)
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    accuracy = knn.accuracy(confusion)
    print("Accuracy for chosen best k= %d: %f" % (best_k, accuracy))