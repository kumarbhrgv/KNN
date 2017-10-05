import argparse
import random
from collections import namedtuple
import numpy as np
from knn import Knearest, Numbers
import math
from sklearn.model_selection import StratifiedShuffleSplit
random.seed(20170830)

SplitData = namedtuple("SplitData", ["xtrain", "xtest","ytrain","ytest"])



def split_crossvalidate(X,y,num_folds) : 
	splits = []
	#using scikit library to split the data in stratified manner
	#StratifiedShuffleSplit means all the data of similar kind are grouped and distributed equally in all the k folds 
	# in a randomized order.
	sss = StratifiedShuffleSplit(num_folds)
	for train_index, test_index in sss.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		splits.append(SplitData(X_train,X_test,y_train,y_test))
	return splits


def cv_performance(x, y, num_folds, k):
    length = len(y)
    splits = split_crossvalidate(x, y, num_folds)
    accuracy_array = []
    for split in splits:
        knn = Knearest(split.xtrain, split.ytrain, k)
        confusion = knn.confusion_matrix(split.xtest,split.ytest)
        accuracy_array.append(knn.accuracy(confusion))
    return np.mean(accuracy_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()
    data = Numbers("../data/mnist.pkl.gz")
    x, y = data.train_x, data.train_y
    if args.limit > 0:
        x, y = x[:args.limit], y[:args.limit]
    best_k, best_accuracy = -1, 0
    for k in [1,3,5,7,9]:
        accuracy = cv_performance(x, y, 10, k)
        print("%d-nearest neighber accuracy: %f" % (k, accuracy))
        if accuracy > best_accuracy:
            best_accuracy, best_k = accuracy, k
    print ("best_accuracy : %f, best_k : %d" %(best_accuracy,best_k))
    knn = Knearest(x, y, best_k)
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    accuracy = knn.accuracy(confusion)
    print("Accuracy for chosen best k= %d: %f" % (best_k, accuracy))