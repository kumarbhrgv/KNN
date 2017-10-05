
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
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set


class Knearest:
    def __init__(self, x, y, k=5):
        self._kdtree =KDTree(x,40, DistanceMetric.get_metric('euclidean'))
        self._y = y
        self._k = k
        self._start_time = time.time()

    def majority(self, item_indices,distance):

        assert len(item_indices) == self._k, "Did not get k inputs"
        counter = defaultdict(int)
        for label in self._y[item_indices]:
            counter[label] += 1
        majority_count = max(counter.values())
        sorted_label_predicted=[]
        for key, value in counter.items():
            if value == majority_count:
                #sorted_label_predicted.append(key)
                return key
        
        #sorted_label_predicted = sorted(sorted_label_predicted)
        #median = numpy.median(sorted_label_predicted)
        #print(sorted_label_predicted,median)
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
        dataframe_confusionMatrix = pd.DataFrame(confusionmatrix, index = [i for i in range(0,10)],
                  columns = [i for i in range(0,10)])
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
    parser.add_argument('--limit', type=int, default=500,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")
    accuracy_array = []
    karray = [1,3,5,7,9,11]
    for i in karray:
        print("Number of nearest points to use: %i" % i)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                           i)
        confusion = knn.confusion_matrix(data.test_x, data.test_y)
        accuracy_for_limit = knn.accuracy(confusion)
        print("Accuracy: %f" % accuracy_for_limit)
        accuracy_array.append(accuracy_for_limit)
    plt.plot(karray,accuracy_array)
    plt.ylabel('accuracy')
    plt.xlabel('k values')
    plt.show()

