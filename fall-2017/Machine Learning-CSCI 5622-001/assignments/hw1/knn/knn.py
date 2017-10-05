import argparse
import pickle
import gzip
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import random
import numpy
from numpy import median
from sklearn.neighbors import KDTree,DistanceMetric
import time

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.


        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set


class Knearest:
    def __init__(self, x, y, k=5,algorithm='euclidean'):
        self._kdtree =KDTree(x,40, DistanceMetric.get_metric(algorithm))
        self._y = y
        self._k = k
        self._start_time = time.time()


    def majority(self, item_indices,distance):

        assert len(item_indices) == self._k, "Did not get k inputs"
        counter = defaultdict(int)
        label_predicted = []
        for label in self._y[item_indices]:
            counter[label] += 1
            label_predicted.append(label)
        majority_count = max(counter.values())
        sorted_label_predicted=[]
        for key, value in counter.items():
            if value == majority_count:
                return key
                #for mean add all the keys to sorted_label_predicted and use numpy.median
                #sorted_label_predicted.append(key)
        #sorted_label_predicted = sorted(sorted_label_predicted)
        #median = numpy.median(sorted_label_predicted)
        #return int(median)

    def classify(self, example):
        distance,indices = self._kdtree.query(example.reshape(1, -1),self._k)
        indices_new = indices.tolist()
        distance_new =distance.tolist()
        return self.majority(indices_new[0],distance_new[0])

    def computeconfusionmatrix(self,yactualList,ypredictedList):
        confusionmatrix = numpy.zeros((10, 10))
        for actual, predicted in zip(yactualList, ypredictedList):
            confusionmatrix[actual][predicted] += 1
        #self.plot_confusion_matrix(confusionmatrix)
        return confusionmatrix

    def confusion_matrix(self, test_x, test_y, debug=False):
        self._start_time =time.time()
        ypredictedList = []
        yactualList=[]
        for xx, yy in zip(test_x, test_y):
            yactualList.append(yy)
            ypred = self.classify(xx)
            ypredictedList.append(ypred)
        #print("--- %s seconds ---" % (time.time() - self._start_time))
        return self.computeconfusionmatrix(yactualList, ypredictedList)
    
    def plot_confusion_matrix(confusionmatrix, title='Confusion matrix', cmap=plt.cm.gray_r):
        dataframe_confusionMatrix = pd.DataFrame(confusionmatrix, 
        index = [i for i in range(0,10)],columns = [i for i in range(0,10)])
        plt.figure(figsize = (10,10))
        sn.heatmap(dataframe_confusionMatrix, annot=True)
        
    @staticmethod
    def accuracy(confusion_matrix):
        total = sum(sum(i) for i in confusion_matrix)
        correct =  sum(confusion_matrix[j][j] for j in range(len(confusion_matrix)))
        if total > 0:
            return float(correct) / float(total)
        else:
        	return 0.0

if __name__ == "__main__":
    print("KNN classifier")
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    parser.add_argument('--algorithm', type=str, default='euclidean',
                        help="Distance metric")
    args = parser.parse_args()
    print("Number of nearest points to use:", args.k)
    if args.algorithm is not 'null' : 
        print("Algorithm chosen to run K",args.algorithm)
    data = Numbers("../data/mnist.pkl.gz")
    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k,args.algorithm)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k,args.algorithm)
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    confusion_matrix = confusion.astype(int)
    print("\t" + "\t".join(str(x) for x in range(10)))
    print("".join(["-"] * 90))
    for ii in range(10):
        print("%i:\t" % ii + "\t".join(str(confusion_matrix[ii][x])
                                       for x in range(10)))
    print("Accuracy: %f" % knn.accuracy(confusion))

